use crate::api::{
    Answer, ApiError, BoundingBox, DecisionBackend, Question, Request, Response, DEFAULT_MODEL,
};
use crate::privacy::{
    content_region_offset_and_scale_frac, prepare_content_zoom_2x, prepare_corner_zoom_2x,
    CornerRegion, PreparedImage,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

pub const NOTHING_TO_CLICK: &str = "nothing to click";
pub const LOW_CONFIDENCE_THRESHOLD: f64 = 0.2;
pub const MAX_POPUP_BOX_WIDTH_FRAC: f64 = 0.15;
pub const PRIMARY_LOW_BOX_CONF_EXCLUDE_STEPS: usize = 3;
pub const CARD_REASK_FIRST_CARD_IMAGE_DESC: &str = "熱門活動區第一張活動卡片的圖片";
pub const CARD_REASK_TITLE_TEXT_DESC: &str = "活動卡片標題文字";
pub const POPUP_REASK_COOKIE_GOT_IT_DESC: &str = "Cookie 橫幅的 我知道了 按鈕";
pub const POPUP_REASK_ACCEPT_DESC: &str = "接受 按鈕";
pub const POPUP_REASK_AGREE_DESC: &str = "同意 按鈕";
pub const POPUP_REASK_COOKIE_CLOSE_X_DESC: &str = "Cookie 說明右上角的 ×";
pub const POPUP_REASK_COOKIE_DESC: &str =
    "Cookie 橫幅的 接受 或 我知道了 按鈕 或 右上角 ×";
pub const POPUP_REASK_AD_CLOSE_DESC: &str = "左下角廣告的 × 關閉按鈕";
pub const POPUP_COOKIE_REASK_DESCRIPTIONS: &[&str] = &[
    POPUP_REASK_COOKIE_GOT_IT_DESC,
    POPUP_REASK_ACCEPT_DESC,
    POPUP_REASK_AGREE_DESC,
    POPUP_REASK_COOKIE_CLOSE_X_DESC,
];
pub const RESULTS_QUESTION_INSTRUCTIONS: &str =
    "Does the image show a Google Flights results list with flight prices?";
pub const CHECKOUT_QUESTION_INSTRUCTIONS: &str =
    "Does the image show a ticket checkout or ticket selection page with ticket prices visible?";
pub const CHECKOUT_ZH_QUESTION_INSTRUCTIONS: &str =
    "畫面是否為顯示票價的選擇票種或結帳頁面？";
pub const SENSITIVE_QUESTION_INSTRUCTIONS: &str =
    "Is the element at this target a sign-in, payment, place-order/purchase/pay button, or a personal-info input field (name, email, phone, card)?";
pub const CLOSEISH_QUESTION_INSTRUCTIONS: &str =
    "Is this button a Close, X, Reject, Decline, Accept-cookie or Got-it button only (never a link, ad body, Subscribe or Sign up)?";
pub const POPUP_PRESENT_QUESTION_INSTRUCTIONS: &str =
    "Is a popup, modal dialog, cookie banner or overlay currently covering part of the page?";
pub const POPUP_PRESENT_THRESHOLD: f64 = 0.6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoneCondition {
    #[serde(rename = "flights-results")]
    FlightsResults,
    #[serde(rename = "checkout-page")]
    CheckoutPage,
    #[serde(rename = "model")]
    Model,
}

impl Default for DoneCondition {
    fn default() -> Self {
        Self::FlightsResults
    }
}

impl DoneCondition {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.trim().to_lowercase().as_str() {
            "flights-results" => Ok(Self::FlightsResults),
            "checkout-page" => Ok(Self::CheckoutPage),
            "model" => Ok(Self::Model),
            other => Err(format!(
                "Invalid --done-when '{}': expected flights-results, checkout-page, or model",
                other
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FlightsResults => "flights-results",
            Self::CheckoutPage => "checkout-page",
            Self::Model => "model",
        }
    }

    pub fn noul_instructions(&self) -> Option<&'static str> {
        self.noul_instructions_for_preset(None)
    }

    pub fn noul_instructions_for_preset(&self, preset: Option<Preset>) -> Option<&'static str> {
        match self {
            Self::FlightsResults => Some(RESULTS_QUESTION_INSTRUCTIONS),
            Self::CheckoutPage => {
                if preset == Some(Preset::EventsZh) {
                    Some(CHECKOUT_ZH_QUESTION_INSTRUCTIONS)
                } else {
                    Some(CHECKOUT_QUESTION_INSTRUCTIONS)
                }
            }
            Self::Model => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetSpec {
    pub label: String,
    pub description: Option<String>,
}

impl TargetSpec {
    pub fn new(label: impl Into<String>, description: Option<String>) -> Self {
        Self {
            label: label.into(),
            description,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Preset {
    #[serde(rename = "flights")]
    Flights,
    #[serde(rename = "events")]
    Events,
    #[serde(rename = "events-zh")]
    EventsZh,
    #[serde(rename = "generic")]
    Generic,
}

impl Default for Preset {
    fn default() -> Self {
        Self::Flights
    }
}

impl Preset {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.trim().to_lowercase().as_str() {
            "flights" => Ok(Self::Flights),
            "events" => Ok(Self::Events),
            "events-zh" => Ok(Self::EventsZh),
            "generic" => Ok(Self::Generic),
            other => Err(format!(
                "Invalid --preset '{}': expected flights, events, events-zh, or generic",
                other
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Flights => "flights",
            Self::Events => "events",
            Self::EventsZh => "events-zh",
            Self::Generic => "generic",
        }
    }

    pub fn default_targets(&self) -> Vec<TargetSpec> {
        match self {
            Self::Flights => flights_preset_targets(),
            Self::Events => events_preset_targets(),
            Self::EventsZh => events_zh_preset_targets(),
            Self::Generic => generic_preset_targets(),
        }
    }

    pub fn default_done_condition(&self) -> DoneCondition {
        match self {
            Self::Flights => DoneCondition::FlightsResults,
            Self::Events | Self::EventsZh => DoneCondition::CheckoutPage,
            Self::Generic => DoneCondition::Model,
        }
    }

    pub fn popup_close_label(&self) -> &'static str {
        match self {
            Self::EventsZh => "關閉彈窗",
            _ => "popup close",
        }
    }

    pub fn checkout_once_label(&self) -> &'static str {
        match self {
            Self::EventsZh => "下一步按鈕",
            _ => "checkout button",
        }
    }
}

pub fn flights_preset_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec::new(
            "trip type dropdown",
            Some(
                "the small dropdown labelled 'Round trip' at the top left of the search form"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "one way option",
            Some("the menu item that reads 'One way' in the open trip-type list".to_string()),
        ),
        TargetSpec::new(
            "origin field",
            Some("the left text field of the search form, showing the origin city".to_string()),
        ),
        TargetSpec::new(
            "origin suggestion",
            Some(
                "the suggestion row reading 'Kansai International Airport' with code KIX"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "destination field",
            Some(
                "the text field labelled 'Where to?' to the right of the origin field".to_string(),
            ),
        ),
        TargetSpec::new(
            "destination suggestion",
            Some(
                "the suggestion row reading 'Taiwan Taoyuan International Airport' with code TPE"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "departure date",
            Some("the date field labelled 'Departure' at the right of the search form".to_string()),
        ),
        TargetSpec::new(
            "departure date option",
            Some("the number 3 under November 2026 in the open date picker".to_string()),
        ),
        TargetSpec::new(
            "done button",
            Some("the blue 'Done' button at the bottom right of the date picker".to_string()),
        ),
        TargetSpec::new(
            "search button",
            Some("the blue 'Search' button under the search form".to_string()),
        ),
    ]
}

pub fn events_preset_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec::new(
            "events menu",
            Some("the 'Events' link in the site header/navigation".to_string()),
        ),
        TargetSpec::new(
            "upcoming event card",
            Some(
                "the card or link for the next upcoming event, nearest future date".to_string(),
            ),
        ),
        TargetSpec::new(
            "tickets link",
            Some(
                "a 'Tickets', 'Buy tickets', 'Get tickets' or 'Register' button/link for that event"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "cheapest ticket option",
            Some("the lowest-priced ticket type row or its select/add button".to_string()),
        ),
        TargetSpec::new(
            "quantity increase",
            Some("the '+' button to set ticket quantity to 1".to_string()),
        ),
        TargetSpec::new(
            "checkout button",
            Some("a 'Checkout', 'Continue' or 'Next' button that proceeds to checkout".to_string()),
        ),
        TargetSpec::new(
            "popup close",
            Some(
                "Close/X/Reject/Decline button of a cookie, newsletter or promo popup".to_string(),
            ),
        ),
    ]
}

pub fn events_zh_preset_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec::new(
            "搜尋或活動分類",
            Some("首頁的活動分類或「近期活動」連結".to_string()),
        ),
        TargetSpec::new(
            "活動卡片",
            Some("列表中顯示日期與標題的即將舉辦活動卡片".to_string()),
        ),
        TargetSpec::new(
            "立即報名按鈕",
            Some("活動頁上的「立即報名」「購票」「報名」按鈕".to_string()),
        ),
        TargetSpec::new(
            "票種選項",
            Some("票種列表中價格最低的那一列或其選擇按鈕".to_string()),
        ),
        TargetSpec::new(
            "數量加號",
            Some("把票數設為 1 的「+」按鈕".to_string()),
        ),
        TargetSpec::new(
            "下一步按鈕",
            Some("前往結帳的「下一步」「結帳」「確認」按鈕".to_string()),
        ),
        TargetSpec::new(
            "關閉彈窗",
            Some("彈窗的「關閉」「X」「不用了」「拒絕」按鈕".to_string()),
        ),
    ]
}

pub fn generic_preset_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec::new(
            "navigation link toward the goal",
            Some("a navigation link or menu item that leads toward the goal".to_string()),
        ),
        TargetSpec::new(
            "primary call-to-action button toward the goal",
            Some(
                "the primary call-to-action button or control that progresses toward the goal"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "popup close",
            Some(
                "Close/X/Reject/Decline button of a cookie, newsletter or promo popup".to_string(),
            ),
        ),
    ]
}

pub fn load_targets_file(path: &std::path::Path) -> Result<Vec<TargetSpec>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read targets file '{}': {}", path.display(), e))?;
    parse_targets_json(&raw)
        .map_err(|e| format!("Invalid targets JSON in '{}': {}", path.display(), e))
}

pub fn parse_targets_json(raw: &str) -> Result<Vec<TargetSpec>, String> {
    let specs: Vec<TargetSpec> =
        serde_json::from_str(raw).map_err(|e| format!("JSON parse error: {}", e))?;
    if specs.is_empty() {
        return Err("Targets list cannot be empty".to_string());
    }
    for s in &specs {
        if s.label.trim().is_empty() {
            return Err("Target label cannot be empty".to_string());
        }
    }
    Ok(specs)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionKind {
    #[serde(rename = "click")]
    Click,
    #[serde(rename = "dismiss popup")]
    DismissPopup,
    #[serde(rename = "type")]
    Type,
    #[serde(rename = "press enter")]
    PressEnter,
    #[serde(rename = "scroll")]
    Scroll,
    #[serde(rename = "wait")]
    Wait,
    #[serde(rename = "done")]
    Done,
}

impl ActionKind {
    pub fn from_choice(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "click" => Some(Self::Click),
            "dismiss popup" | "dismiss_popup" => Some(Self::DismissPopup),
            "type" => Some(Self::Type),
            "press enter" | "press_enter" => Some(Self::PressEnter),
            "scroll" => Some(Self::Scroll),
            "wait" => Some(Self::Wait),
            "done" => Some(Self::Done),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Click => "click",
            Self::DismissPopup => "dismiss popup",
            Self::Type => "type",
            Self::PressEnter => "press enter",
            Self::Scroll => "scroll",
            Self::Wait => "wait",
            Self::Done => "done",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub kind: ActionKind,
    pub kind_confidence: f64,
    pub target: Option<String>,
    pub click_point_screen_pt: Option<(f64, f64)>,
    pub bbox: Option<BoundingBox>,
    pub box_confidence: Option<f64>,
    pub confidence: f64,
    pub reasked: bool,
    pub text_to_type: Option<String>,
    pub results_noul: f64,
    pub sensitive_noul: Option<f64>,
    pub closeish_noul: Option<f64>,
    pub popup_present_noul: Option<f64>,
}

#[derive(Debug, Error)]
pub enum DeciderError {
    #[error("API error: {0}")]
    Api(#[from] ApiError),
    #[error("Missing answer for question '{0}'")]
    MissingAnswer(String),
    #[error("Unexpected answer variant for question '{0}'")]
    UnexpectedAnswerType(String),
    #[error("Unknown action kind '{0}'")]
    UnknownActionKind(String),
}

pub fn map_box_center_to_screen_pt(bbox: &BoundingBox, img: &PreparedImage) -> (f64, f64) {
    let (cx_frac, cy_frac) = bbox.center();
    let scale = if img.scale_factor <= 0.0 {
        1.0
    } else {
        img.scale_factor
    };
    let (origin_x_px, origin_y_px) = img.crop_origin_px;
    let (crop_w_px, crop_h_px) = img.crop_size_px;

    let screen_x_px = (origin_x_px as f64) + cx_frac * (crop_w_px as f64);
    let screen_y_px = (origin_y_px as f64) + cy_frac * (crop_h_px as f64);

    (screen_x_px / scale, screen_y_px / scale)
}

pub fn map_corner_box_to_window_box(bbox: &BoundingBox, corner: CornerRegion) -> BoundingBox {
    let (left_off, top_off, w_scale, h_scale) = corner.offset_and_scale_frac();
    BoundingBox {
        top: top_off + bbox.top * h_scale,
        left: left_off + bbox.left * w_scale,
        bottom: top_off + bbox.bottom * h_scale,
        right: left_off + bbox.right * w_scale,
    }
}

pub fn map_content_box_to_window_box(
    bbox: &BoundingBox,
    window_img: &PreparedImage,
) -> BoundingBox {
    let (left_off, top_off, w_scale, h_scale) = content_region_offset_and_scale_frac(window_img);
    BoundingBox {
        top: top_off + bbox.top * h_scale,
        left: left_off + bbox.left * w_scale,
        bottom: top_off + bbox.bottom * h_scale,
        right: left_off + bbox.right * w_scale,
    }
}

pub fn is_popup_box_width_ok(bbox: &BoundingBox) -> bool {
    let width_frac = bbox.right - bbox.left;
    width_frac >= 0.0 && width_frac <= MAX_POPUP_BOX_WIDTH_FRAC + 1e-9
}

pub fn is_auxiliary_target(label: &str) -> bool {
    let trimmed = label.trim();
    trimmed == "關閉彈窗"
        || trimmed == "搜尋或活動分類"
        || trimmed.eq_ignore_ascii_case("popup close")
        || trimmed.eq_ignore_ascii_case("close popup")
        || trimmed.eq_ignore_ascii_case("events menu")
}

pub fn is_primary_target(label: &str) -> bool {
    !is_auxiliary_target(label)
}

pub fn low_box_conf_exclude_steps(label: &str) -> usize {
    if is_auxiliary_target(label) {
        1000
    } else {
        PRIMARY_LOW_BOX_CONF_EXCLUDE_STEPS
    }
}

pub fn card_primary_reask_descriptions(label: &str) -> Option<&'static [&'static str]> {
    match label.trim() {
        "活動卡片" => Some(&[CARD_REASK_FIRST_CARD_IMAGE_DESC, CARD_REASK_TITLE_TEXT_DESC]),
        "立即報名按鈕" => Some(&[
            "活動頁面右側或下方的「立即報名」按鈕",
            "「購票」或「報名」按鈕",
        ]),
        "票種選項" => Some(&[
            "票種列表中第一列或價格最低的票種選項",
            "票種選擇按鈕或票券列",
        ]),
        "數量加號" => Some(&[
            "票數右側把數量設為 1 的「+」加號按鈕",
            "增加數量的「+」按鈕",
        ]),
        "下一步按鈕" => Some(&[
            "頁面底部前往結帳的「下一步」或「確認」按鈕",
            "結帳或下一步按鈕",
        ]),
        _ => None,
    }
}

pub fn extract_clicked_target(history_entry: &str) -> Option<String> {
    if let Some(rest) = history_entry.strip_prefix("click '") {
        let end = rest.find('\'')?;
        return Some(rest[..end].to_string());
    }
    if let Some(pos) = history_entry.find("' into ") {
        if history_entry.starts_with("clicked and typed '") {
            return Some(history_entry[pos + 7..].trim().to_string());
        }
    }
    None
}

pub fn clicked_targets_from_history(history: &[String]) -> Vec<String> {
    history
        .iter()
        .filter_map(|h| extract_clicked_target(h))
        .collect()
}

pub fn extract_low_box_conf_target(entry: &str) -> Option<String> {
    if let Some(rest) = entry.strip_prefix("low_box_conf:") {
        let trimmed = rest.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Some(rest) = entry.strip_prefix("low_box_conf '") {
        let end = rest.find('\'')?;
        return Some(rest[..end].to_string());
    }
    None
}

pub fn compute_excluded_targets(step_clicks: &[Option<String>]) -> Vec<String> {
    let mut excluded = Vec::new();
    if step_clicks.len() >= 2 {
        if let (Some(prev1), Some(prev2)) = (
            &step_clicks[step_clicks.len() - 2],
            &step_clicks[step_clicks.len() - 1],
        ) {
            if prev1 == prev2 && extract_low_box_conf_target(prev1).is_none() {
                excluded.push(prev1.clone());
            }
        }
    }
    for (idx, entry_opt) in step_clicks.iter().enumerate() {
        if let Some(entry) = entry_opt {
            if let Some(tgt) = extract_low_box_conf_target(entry) {
                let steps_after = step_clicks.len() - idx;
                if steps_after <= low_box_conf_exclude_steps(&tgt) && !excluded.contains(&tgt) {
                    excluded.push(tgt);
                }
            }
        }
    }
    excluded
}

pub fn format_state(goal: &str, history: &[String]) -> String {
    let recent: Vec<&String> = history
        .iter()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let clicked = clicked_targets_from_history(history);
    let clicked_str = if clicked.is_empty() {
        "(none)".to_string()
    } else {
        clicked.join(" -> ")
    };
    if recent.is_empty() {
        format!(
            "Goal: {}\nTargets already clicked: {}\nHistory: (none)",
            goal, clicked_str
        )
    } else {
        let joined = recent
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{}. {}", i + 1, h))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Goal: {}\nTargets already clicked: {}\nHistory:\n{}",
            goal, clicked_str, joined
        )
    }
}

fn pending_fill_target<'a>(
    history: &[String],
    type_values: &'a BTreeMap<String, String>,
) -> Option<&'a str> {
    if type_values.is_empty() {
        return None;
    }
    let last = history.last()?;
    if !last.starts_with("click '") {
        return None;
    }
    for key in type_values.keys() {
        let prefix = format!("click '{}'", key);
        if last.starts_with(&prefix) {
            return Some(key.as_str());
        }
    }
    None
}

pub struct VisionDecider<B: DecisionBackend> {
    backend: B,
    model: String,
    targets: Vec<TargetSpec>,
    preset: Option<Preset>,
    done_condition: DoneCondition,
    pub type_values: BTreeMap<String, String>,
    requests_sent: AtomicUsize,
    images_sent: AtomicUsize,
    last_popup_present: std::sync::Mutex<Option<f64>>,
}

impl<B: DecisionBackend> VisionDecider<B> {
    pub fn new(backend: B, targets: Vec<TargetSpec>) -> Self {
        Self {
            backend,
            model: DEFAULT_MODEL.to_string(),
            targets,
            preset: None,
            done_condition: DoneCondition::FlightsResults,
            type_values: BTreeMap::new(),
            requests_sent: AtomicUsize::new(0),
            images_sent: AtomicUsize::new(0),
            last_popup_present: std::sync::Mutex::new(None),
        }
    }

    pub fn with_preset(mut self, preset: Preset) -> Self {
        self.preset = Some(preset);
        self
    }

    pub fn preset(&self) -> Option<Preset> {
        self.preset
    }

    pub fn allow_dismiss_popup(&self) -> bool {
        match self.preset {
            Some(Preset::Events | Preset::EventsZh | Preset::Generic) => true,
            Some(Preset::Flights) => false,
            None => self
                .targets
                .iter()
                .any(|t| t.label == "popup close" || t.label == "關閉彈窗"),
        }
    }

    pub fn popup_close_label(&self) -> &'static str {
        match self.preset {
            Some(p) => p.popup_close_label(),
            None => {
                if self.targets.iter().any(|t| t.label == "關閉彈窗") {
                    "關閉彈窗"
                } else {
                    "popup close"
                }
            }
        }
    }

    pub fn is_popup_close_target(&self, label: &str) -> bool {
        let trimmed = label.trim();
        trimmed.eq_ignore_ascii_case("popup close")
            || trimmed.eq_ignore_ascii_case("close popup")
            || trimmed == "關閉彈窗"
    }

    pub fn scroll_target_label(&self) -> &'static str {
        match self.preset {
            Some(Preset::EventsZh) => "向下捲動",
            Some(_) => "scroll down",
            None => {
                if self
                    .targets
                    .iter()
                    .any(|t| t.label == "關閉彈窗" || t.label == "活動卡片")
                {
                    "向下捲動"
                } else {
                    "scroll down"
                }
            }
        }
    }

    pub fn is_checkout_once_target(&self, label: &str) -> bool {
        let trimmed = label.trim();
        trimmed.eq_ignore_ascii_case("checkout button") || trimmed == "下一步按鈕"
    }

    pub fn with_done_condition(mut self, done_condition: DoneCondition) -> Self {
        self.done_condition = done_condition;
        self
    }

    pub fn done_condition(&self) -> DoneCondition {
        self.done_condition
    }

    pub fn with_type_values(mut self, type_values: BTreeMap<String, String>) -> Self {
        self.type_values = type_values;
        self
    }

    pub fn type_values(&self) -> &BTreeMap<String, String> {
        &self.type_values
    }

    pub fn fill_for(&self, target_label: &str) -> Option<&str> {
        self.type_values.get(target_label).map(|s| s.as_str())
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn targets(&self) -> &[TargetSpec] {
        &self.targets
    }

    pub fn requests_sent(&self) -> usize {
        self.requests_sent
            .load(Ordering::SeqCst)
            .max(self.backend.requests_sent())
    }

    pub fn images_sent(&self) -> usize {
        self.images_sent.load(Ordering::SeqCst)
    }

    fn ask_backend(&self, req: &Request) -> Result<Response, DeciderError> {
        self.requests_sent.fetch_add(1, Ordering::SeqCst);
        self.images_sent
            .fetch_add(req.images.len(), Ordering::SeqCst);
        Ok(self.backend.ask(req)?)
    }

    fn find_target_description(&self, label: &str) -> Option<&str> {
        self.targets
            .iter()
            .find(|t| t.label == label)
            .and_then(|t| t.description.as_deref())
    }

    pub fn has_available_targets(&self, excluded_targets: &[String]) -> bool {
        let prev_popup_present = self
            .last_popup_present
            .lock()
            .unwrap()
            .unwrap_or(0.0);
        self.targets.iter().any(|t| {
            if self.is_popup_close_target(&t.label) && prev_popup_present < POPUP_PRESENT_THRESHOLD {
                return false;
            }
            !excluded_targets.iter().any(|ex| ex == &t.label)
        })
    }

    pub fn resolve_click_box(
        &self,
        state: &str,
        img: &PreparedImage,
        tgt: &str,
        mut audit_log: Option<(&mut crate::privacy::AuditLog, usize)>,
    ) -> Result<(BoundingBox, f64, Option<f64>, bool), DeciderError> {
        let (mut bbox, mut box_conf, mut sensitive_noul) = self.ask_box(state, img, tgt)?;
        let mut reasked = false;

        if box_conf < LOW_CONFIDENCE_THRESHOLD {
            reasked = true;
            if let Some(specific_descs) = card_primary_reask_descriptions(tgt) {
                for desc in specific_descs {
                    let (b, c, s) = self.ask_box(state, img, desc)?;
                    bbox = b;
                    box_conf = c;
                    sensitive_noul = s.or(sensitive_noul);
                    if box_conf >= LOW_CONFIDENCE_THRESHOLD {
                        break;
                    }
                }
                if box_conf < LOW_CONFIDENCE_THRESHOLD {
                    if let Ok(content_img) = prepare_content_zoom_2x(img) {
                        if let Some((ref mut log, step)) = audit_log {
                            log.record(&content_img, format!("step_{}_content_2x", step));
                        }
                        let crop_desc = specific_descs.first().copied().unwrap_or(tgt);
                        let (cb, cc, cs) = self.ask_box(state, &content_img, crop_desc)?;
                        let mapped_box = map_content_box_to_window_box(&cb, img);
                        bbox = mapped_box;
                        box_conf = cc;
                        sensitive_noul = cs.or(sensitive_noul);
                    }
                }
            } else {
                let longer = self
                    .find_target_description(tgt)
                    .map(|desc| format!("{} ({})", tgt, desc))
                    .unwrap_or_else(|| format!("Locate the interactive element: {}", tgt));
                let (bbox2, conf2, sens2) = self.ask_box(state, img, &longer)?;
                bbox = bbox2;
                box_conf = conf2;
                sensitive_noul = sens2.or(sensitive_noul);
            }
        }

        Ok((bbox, box_conf, sensitive_noul, reasked))
    }

    pub fn decide(
        &self,
        goal: &str,
        history: &[String],
        img: &PreparedImage,
    ) -> Result<Decision, DeciderError> {
        let step_clicks: Vec<Option<String>> = history
            .iter()
            .map(|h| {
                extract_low_box_conf_target(h)
                    .map(|t| format!("low_box_conf:{}", t))
                    .or_else(|| extract_clicked_target(h))
            })
            .collect();
        let auto_excluded = compute_excluded_targets(&step_clicks);
        self.decide_with_excluded(goal, history, img, &auto_excluded)
    }

    pub fn decide_with_excluded_defer_popup(
        &self,
        goal: &str,
        history: &[String],
        img: &PreparedImage,
        excluded_targets: &[String],
    ) -> Result<Decision, DeciderError> {
        self.decide_with_excluded_defer_popup_with_audit(goal, history, img, excluded_targets, None)
    }

    pub fn decide_with_excluded_defer_popup_with_audit(
        &self,
        goal: &str,
        history: &[String],
        img: &PreparedImage,
        excluded_targets: &[String],
        audit_log: Option<(&mut crate::privacy::AuditLog, usize)>,
    ) -> Result<Decision, DeciderError> {
        let state = format_state(goal, history);
        let active_fill_target = pending_fill_target(history, &self.type_values);
        let prev_popup_present = self
            .last_popup_present
            .lock()
            .unwrap()
            .unwrap_or(0.0);

        let popup_close = self.popup_close_label();
        let popup_close_excluded = excluded_targets.iter().any(|ex| ex == popup_close);

        let mut kind_criteria = BTreeMap::new();
        kind_criteria.insert(
            "click".to_string(),
            Some("Click an interactive element on the page".to_string()),
        );
        if self.allow_dismiss_popup()
            && prev_popup_present >= POPUP_PRESENT_THRESHOLD
            && !popup_close_excluded
        {
            kind_criteria.insert(
                "dismiss popup".to_string(),
                Some("Close a cookie, newsletter or promo popup".to_string()),
            );
        }
        if active_fill_target.is_some() {
            kind_criteria.insert(
                "type".to_string(),
                Some("Type text into the currently focused input field".to_string()),
            );
        }
        kind_criteria.insert(
            "press enter".to_string(),
            Some("Press Return to confirm the current input".to_string()),
        );
        kind_criteria.insert(
            "scroll".to_string(),
            Some("Scroll down the page to reveal more content".to_string()),
        );
        let last_two_are_waits =
            history.len() >= 2 && history.iter().rev().take(2).all(|h| h == "wait");
        if !last_two_are_waits {
            kind_criteria.insert(
                "wait".to_string(),
                Some("Wait for the page to finish loading".to_string()),
            );
        }
        kind_criteria.insert(
            "done".to_string(),
            Some("The goal has been achieved or no further action is needed".to_string()),
        );

        let mut target_criteria = BTreeMap::new();
        for t in &self.targets {
            if self.is_popup_close_target(&t.label) && prev_popup_present < POPUP_PRESENT_THRESHOLD {
                continue;
            }
            if !excluded_targets.iter().any(|ex| ex == &t.label) {
                target_criteria.insert(t.label.clone(), t.description.clone());
            }
        }
        target_criteria.insert(
            NOTHING_TO_CLICK.to_string(),
            Some("No relevant target is visible to click".to_string()),
        );

        let mut questions = BTreeMap::new();
        questions.insert(
            "kind".to_string(),
            Question::Choice {
                instructions: "Choose the next action kind required to progress toward the goal."
                    .to_string(),
                criteria: kind_criteria,
            },
        );
        questions.insert(
            "target".to_string(),
            Question::Choice {
                instructions:
                    "Choose the UI element to interact with, or 'nothing to click' if none applies."
                        .to_string(),
                criteria: target_criteria,
            },
        );
        if let Some(instructions) = self.done_condition.noul_instructions_for_preset(self.preset) {
            questions.insert(
                "results".to_string(),
                Question::Noul {
                    instructions: instructions.to_string(),
                },
            );
        }
        if self.allow_dismiss_popup() {
            questions.insert(
                "popup_present".to_string(),
                Question::Noul {
                    instructions: POPUP_PRESENT_QUESTION_INSTRUCTIONS.to_string(),
                },
            );
        }

        let req1 = Request {
            model: self.model.clone(),
            state: state.clone(),
            images: vec![img.data_url.clone()],
            questions,
        };

        let resp1 = self.ask_backend(&req1)?;

        let (kind_choice, kind_conf) = match resp1.answers.get("kind") {
            Some(Answer::Choice {
                choice, confidence, ..
            }) => (choice.clone(), *confidence),
            Some(_) => return Err(DeciderError::UnexpectedAnswerType("kind".to_string())),
            None => return Err(DeciderError::MissingAnswer("kind".to_string())),
        };

        let target_choice = match resp1.answers.get("target") {
            Some(Answer::Choice { choice, .. }) => Some(choice.clone()),
            Some(_) => return Err(DeciderError::UnexpectedAnswerType("target".to_string())),
            None => None,
        };

        let results_noul = match resp1.answers.get("results") {
            Some(Answer::Noul { noul }) => *noul,
            Some(_) => return Err(DeciderError::UnexpectedAnswerType("results".to_string())),
            None => 0.0,
        };

        let popup_present_noul = match resp1.answers.get("popup_present") {
            Some(Answer::Noul { noul }) => Some(*noul),
            Some(_) => return Err(DeciderError::UnexpectedAnswerType("popup_present".to_string())),
            None => None,
        };
        *self.last_popup_present.lock().unwrap() = popup_present_noul;

        let action_kind = ActionKind::from_choice(&kind_choice)
            .ok_or_else(|| DeciderError::UnknownActionKind(kind_choice.clone()))?;

        if action_kind == ActionKind::DismissPopup {
            return Ok(Decision {
                kind: ActionKind::DismissPopup,
                kind_confidence: kind_conf,
                target: Some(self.popup_close_label().to_string()),
                click_point_screen_pt: None,
                bbox: None,
                box_confidence: None,
                confidence: kind_conf,
                reasked: false,
                text_to_type: None,
                results_noul,
                sensitive_noul: None,
                closeish_noul: None,
                popup_present_noul,
            });
        }

        if action_kind == ActionKind::Click {
            let tgt = target_choice
                .clone()
                .unwrap_or_else(|| NOTHING_TO_CLICK.to_string());
            if tgt == NOTHING_TO_CLICK {
                return Ok(Decision {
                    kind: ActionKind::Wait,
                    kind_confidence: kind_conf,
                    target: Some(tgt),
                    click_point_screen_pt: None,
                    bbox: None,
                    box_confidence: None,
                    confidence: kind_conf,
                    reasked: false,
                    text_to_type: None,
                    results_noul,
                    sensitive_noul: None,
                    closeish_noul: None,
                    popup_present_noul,
                });
            }

            let (bbox, box_conf, sensitive_noul, reasked) =
                self.resolve_click_box(&state, img, &tgt, audit_log)?;

            let pt = if box_conf < LOW_CONFIDENCE_THRESHOLD {
                None
            } else {
                Some(map_box_center_to_screen_pt(&bbox, img))
            };
            let fill_text = if box_conf < LOW_CONFIDENCE_THRESHOLD {
                None
            } else {
                self.type_values.get(&tgt).cloned()
            };
            return Ok(Decision {
                kind: ActionKind::Click,
                kind_confidence: kind_conf,
                target: Some(tgt),
                click_point_screen_pt: pt,
                bbox: Some(bbox),
                box_confidence: Some(box_conf),
                confidence: box_conf,
                reasked,
                text_to_type: fill_text,
                results_noul,
                sensitive_noul,
                closeish_noul: None,
                popup_present_noul,
            });
        }

        let text_to_type = if action_kind == ActionKind::Type {
            active_fill_target
                .and_then(|t| self.type_values.get(t).cloned())
                .or_else(|| {
                    target_choice
                        .as_ref()
                        .and_then(|t| self.type_values.get(t).cloned())
                })
        } else {
            None
        };

        let final_target = if action_kind == ActionKind::Scroll {
            Some(self.scroll_target_label().to_string())
        } else {
            target_choice
        };

        Ok(Decision {
            kind: action_kind,
            kind_confidence: kind_conf,
            target: final_target,
            click_point_screen_pt: None,
            bbox: None,
            box_confidence: None,
            confidence: kind_conf,
            reasked: false,
            text_to_type,
            results_noul,
            sensitive_noul: None,
            closeish_noul: None,
            popup_present_noul,
        })
    }

    pub fn resolve_dismiss_popup(
        &self,
        state: &str,
        img: &PreparedImage,
        decision: &mut Decision,
        mut audit_log: Option<(&mut crate::privacy::AuditLog, usize)>,
    ) -> Result<(), DeciderError> {
        let tgt = decision
            .target
            .clone()
            .unwrap_or_else(|| self.popup_close_label().to_string());
        let (mut bbox, mut box_conf, mut closeish_noul) =
            self.ask_dismiss_popup_box(state, img, &tgt)?;
        let mut reasked = false;

        if box_conf < LOW_CONFIDENCE_THRESHOLD {
            reasked = true;
            for desc in [
                POPUP_REASK_COOKIE_GOT_IT_DESC,
                POPUP_REASK_ACCEPT_DESC,
                POPUP_REASK_AGREE_DESC,
                POPUP_REASK_COOKIE_CLOSE_X_DESC,
                POPUP_REASK_COOKIE_DESC,
                POPUP_REASK_AD_CLOSE_DESC,
            ] {
                let (b, c, cl) = self.ask_dismiss_popup_box(state, img, desc)?;
                bbox = b;
                box_conf = c;
                closeish_noul = cl.or(closeish_noul);
                if box_conf >= LOW_CONFIDENCE_THRESHOLD {
                    break;
                }
            }
        }

        if box_conf < LOW_CONFIDENCE_THRESHOLD {
            for (corner, desc, suffix) in [
                (
                    CornerRegion::BottomRight,
                    POPUP_REASK_COOKIE_GOT_IT_DESC,
                    "bottom_right_2x",
                ),
                (
                    CornerRegion::BottomLeft,
                    POPUP_REASK_AD_CLOSE_DESC,
                    "bottom_left_2x",
                ),
            ] {
                if let Ok(corner_img) = prepare_corner_zoom_2x(img, corner) {
                    if let Some((ref mut log, step)) = audit_log {
                        log.record(&corner_img, format!("step_{}_popup_{}", step, suffix));
                    }
                    let (cb, cc, cl) = self.ask_dismiss_popup_box(state, &corner_img, desc)?;
                    let mapped_box = map_corner_box_to_window_box(&cb, corner);
                    bbox = mapped_box;
                    box_conf = cc;
                    closeish_noul = cl.or(closeish_noul);
                    if box_conf >= LOW_CONFIDENCE_THRESHOLD {
                        break;
                    }
                }
            }
        }

        let valid = box_conf >= LOW_CONFIDENCE_THRESHOLD && is_popup_box_width_ok(&bbox);
        if !valid && box_conf >= LOW_CONFIDENCE_THRESHOLD {
            // Reject boxes larger than 15% of window width (never click link or ad body)
            box_conf = 0.0;
        }

        let pt = if valid {
            Some(map_box_center_to_screen_pt(&bbox, img))
        } else {
            None
        };
        decision.bbox = Some(bbox);
        decision.box_confidence = Some(box_conf);
        decision.confidence = box_conf;
        decision.reasked = reasked;
        decision.click_point_screen_pt = pt;
        decision.closeish_noul = closeish_noul;
        Ok(())
    }

    pub fn decide_with_excluded(
        &self,
        goal: &str,
        history: &[String],
        img: &PreparedImage,
        excluded_targets: &[String],
    ) -> Result<Decision, DeciderError> {
        let mut decision = self.decide_with_excluded_defer_popup_with_audit(
            goal,
            history,
            img,
            excluded_targets,
            None,
        )?;
        if decision.kind == ActionKind::DismissPopup {
            let state = format_state(goal, history);
            self.resolve_dismiss_popup(&state, img, &mut decision, None)?;
        }
        Ok(decision)
    }

    fn ask_box(
        &self,
        state: &str,
        img: &PreparedImage,
        instructions: &str,
    ) -> Result<(BoundingBox, f64, Option<f64>), DeciderError> {
        let mut questions = BTreeMap::new();
        questions.insert(
            "target_box".to_string(),
            Question::Box {
                instructions: instructions.to_string(),
            },
        );
        questions.insert(
            "sensitive".to_string(),
            Question::Noul {
                instructions: SENSITIVE_QUESTION_INSTRUCTIONS.to_string(),
            },
        );
        let req = Request {
            model: self.model.clone(),
            state: state.to_string(),
            images: vec![img.data_url.clone()],
            questions,
        };
        let resp = self.ask_backend(&req)?;
        let sensitive_noul = match resp.answers.get("sensitive") {
            Some(Answer::Noul { noul }) => Some(*noul),
            Some(_) => return Err(DeciderError::UnexpectedAnswerType("sensitive".to_string())),
            None => None,
        };
        match resp.answers.get("target_box") {
            Some(Answer::Box { bbox, confidence }) => Ok((*bbox, *confidence, sensitive_noul)),
            Some(_) => Err(DeciderError::UnexpectedAnswerType("target_box".to_string())),
            None => Err(DeciderError::MissingAnswer("target_box".to_string())),
        }
    }

    fn ask_dismiss_popup_box(
        &self,
        state: &str,
        img: &PreparedImage,
        instructions: &str,
    ) -> Result<(BoundingBox, f64, Option<f64>), DeciderError> {
        let mut questions = BTreeMap::new();
        questions.insert(
            "target_box".to_string(),
            Question::Box {
                instructions: instructions.to_string(),
            },
        );
        questions.insert(
            "closeish".to_string(),
            Question::Noul {
                instructions: CLOSEISH_QUESTION_INSTRUCTIONS.to_string(),
            },
        );
        let req = Request {
            model: self.model.clone(),
            state: state.to_string(),
            images: vec![img.data_url.clone()],
            questions,
        };
        let resp = self.ask_backend(&req)?;
        let closeish_noul = match resp.answers.get("closeish") {
            Some(Answer::Noul { noul }) => Some(*noul),
            Some(_) => return Err(DeciderError::UnexpectedAnswerType("closeish".to_string())),
            None => None,
        };
        match resp.answers.get("target_box") {
            Some(Answer::Box { bbox, confidence }) => Ok((*bbox, *confidence, closeish_noul)),
            Some(_) => Err(DeciderError::UnexpectedAnswerType("target_box".to_string())),
            None => Err(DeciderError::MissingAnswer("target_box".to_string())),
        }
    }
}
