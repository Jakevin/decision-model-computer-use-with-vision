use crate::api::{
    Answer, ApiError, BoundingBox, DecisionBackend, Question, Request, Response, DEFAULT_MODEL,
};
use crate::experience::{extract_host_from_url, ExperienceEntry, ExperienceStore};
use crate::harness::{
    CandidateKind, HarnessOverrides, DEFAULT_MAX_EXTRA_REQUESTS_PER_STEP,
};
use crate::ocr::{
    candidates_for_target, is_ocr_text_target, OcrLine, TriedOcrItem, TRIED_POSITION_TOLERANCE_FRAC,
};
use crate::privacy::{
    content_region_offset_and_scale_frac, corner_offset_and_scale_frac, is_sensitive_text,
    prepare_content_zoom_2x, prepare_corner_zoom_2x, CornerRegion, PreparedImage,
};
use crate::safety::SafetyPolicy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

pub const NOTHING_TO_CLICK: &str = "nothing to click";
pub const LOW_CONFIDENCE_THRESHOLD: f64 = 0.2;
pub const SELF_FIX_OCR_BOX_BONUS: f64 = 0.10;
pub const SELF_FIX_SENSITIVE_THRESHOLD: f64 = 0.5;
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
pub const EVENT_PAGE_REGISTER_DESC: &str = "立即報名 或 購票 或 報名 按鈕";
pub const EVENT_PAGE_TICKET_DESC: &str = "票種列表中第一列或價格最低的票種選項";
pub const FOOTER_QUESTION_INSTRUCTIONS: &str = "頁面底部的頁尾區塊（關於我們/客服中心）";
pub const FOOTER_DETECTED_THRESHOLD: f64 = 0.5;
pub const LOGIN_FORM_QUESTION_INSTRUCTIONS: &str = "登入表單（電子信箱與密碼欄位）";
pub const LOGIN_FORM_DETECTED_THRESHOLD: f64 = 0.5;
pub const PRICE_TEXT_BOX_DESC: &str = "頁面上的票價文字（NT$ 或 元）";
pub const CARD_LINK_RETRY_TITLE_DESC: &str = "活動卡片的標題文字連結";
pub const CARD_LINK_RETRY_IMAGE_DESC: &str = "活動卡片的圖片";
pub const CARD_LINK_RETRY_DESCRIPTIONS: &[&str] = &[
    CARD_LINK_RETRY_TITLE_DESC,
    CARD_LINK_RETRY_IMAGE_DESC,
];
pub const MAX_POPUP_CLOSE_COUNT: usize = 2;
pub const POPUP_LOCATION_PROXIMITY_FRAC: f64 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScrollDirection {
    #[serde(rename = "down")]
    Down,
    #[serde(rename = "up")]
    Up,
}

impl Default for ScrollDirection {
    fn default() -> Self {
        Self::Down
    }
}

impl ScrollDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Down => "down",
            Self::Up => "up",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosedPopupMemory {
    pub description: String,
    pub center_frac: Option<(f64, f64)>,
    pub count: usize,
}

pub fn is_event_page_url(url: Option<&str>) -> bool {
    url.map(|u| u.contains("/event/")).unwrap_or(false)
}

pub fn is_login_url(url: Option<&str>) -> bool {
    url.map(|u| {
        let lower = u.to_lowercase();
        lower.contains("/login") || lower.contains("/signin")
    })
    .unwrap_or(false)
}
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
pub struct SelfFixTriedEntry {
    pub candidate: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelfFixProbeResult {
    pub target: String,
    pub tried: Vec<SelfFixTriedEntry>,
    pub chosen: Option<String>,
    pub chosen_lesson: Option<String>,
    pub base_conf: f64,
    pub budget_used: usize,
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
    #[serde(default)]
    pub footer_noul: Option<f64>,
    #[serde(default)]
    pub login_form_noul: Option<f64>,
    #[serde(default)]
    pub click_source: Option<String>,
    #[serde(default)]
    pub ocr_item_text: Option<String>,
    #[serde(default)]
    pub ocr_item_center_frac: Option<(f64, f64)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub self_fix: Option<SelfFixProbeResult>,
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
    let roi_x = img.crop_origin_px.0 as f64;
    let roi_y = img.crop_origin_px.1 as f64;
    let roi_w = img.crop_size_px.0 as f64;
    let roi_h = img.crop_size_px.1 as f64;

    if let Some(win) = img.window_rect {
        let cap_w = (if img.capture_size_px.0 > 0 {
            img.capture_size_px.0
        } else {
            img.crop_size_px.0
        }
        .max(1)) as f64;
        let cap_h = (if img.capture_size_px.1 > 0 {
            img.capture_size_px.1
        } else {
            img.crop_size_px.1
        }
        .max(1)) as f64;
        let x_pt = win.x + (roi_x + cx_frac * roi_w) * win.width / cap_w;
        let y_pt = win.y + (roi_y + cy_frac * roi_h) * win.height / cap_h;
        (x_pt, y_pt)
    } else {
        let scale = if img.scale_factor <= 0.0 {
            1.0
        } else {
            img.scale_factor
        };
        ((roi_x + cx_frac * roi_w) / scale, (roi_y + cy_frac * roi_h) / scale)
    }
}

pub fn map_corner_box_to_window_box(
    bbox: &BoundingBox,
    window_img: &PreparedImage,
    corner: CornerRegion,
) -> BoundingBox {
    let (left_off, top_off, w_scale, h_scale) = corner_offset_and_scale_frac(window_img, corner);
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
    current_url: std::sync::Mutex<Option<String>>,
    scroll_direction: std::sync::Mutex<ScrollDirection>,
    closed_popups: std::sync::Mutex<Vec<ClosedPopupMemory>>,
    ocr_candidates: std::sync::Mutex<Vec<OcrLine>>,
    tried_ocr_items: std::sync::Mutex<Vec<TriedOcrItem>>,
    harness: Option<HarnessOverrides>,
    experience_store: Option<ExperienceStore>,
    self_fix_enabled: bool,
    safety_policy: SafetyPolicy,
    last_self_fix: std::sync::Mutex<Option<SelfFixProbeResult>>,
    last_ocr_item_choice: std::sync::Mutex<Option<(String, (f64, f64))>>,
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
            current_url: std::sync::Mutex::new(None),
            scroll_direction: std::sync::Mutex::new(ScrollDirection::Down),
            closed_popups: std::sync::Mutex::new(Vec::new()),
            ocr_candidates: std::sync::Mutex::new(Vec::new()),
            tried_ocr_items: std::sync::Mutex::new(Vec::new()),
            harness: None,
            experience_store: None,
            self_fix_enabled: true,
            safety_policy: SafetyPolicy::default(),
            last_self_fix: std::sync::Mutex::new(None),
            last_ocr_item_choice: std::sync::Mutex::new(None),
        }
    }

    pub fn with_harness(mut self, harness: HarnessOverrides) -> Self {
        self.targets = harness.apply_to_targets(&self.targets);
        self.harness = Some(harness);
        self
    }

    pub fn with_harness_opt(self, harness: Option<HarnessOverrides>) -> Self {
        match harness {
            Some(h) => self.with_harness(h),
            None => self,
        }
    }

    pub fn harness(&self) -> Option<&HarnessOverrides> {
        self.harness.as_ref()
    }

    pub fn with_experience_store(mut self, store: ExperienceStore) -> Self {
        self.experience_store = Some(store);
        self
    }

    pub fn with_experience_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.experience_store = Some(ExperienceStore::new(dir));
        self
    }

    pub fn set_experience_store(&mut self, store: Option<ExperienceStore>) {
        self.experience_store = store;
    }

    pub fn experience_store(&self) -> Option<&ExperienceStore> {
        self.experience_store.as_ref()
    }

    pub fn with_self_fix(mut self, enabled: bool) -> Self {
        self.self_fix_enabled = enabled;
        self
    }

    pub fn set_self_fix(&mut self, enabled: bool) {
        self.self_fix_enabled = enabled;
    }

    pub fn self_fix_enabled(&self) -> bool {
        self.self_fix_enabled
    }

    pub fn with_safety_policy(mut self, policy: SafetyPolicy) -> Self {
        self.safety_policy = policy;
        self
    }

    pub fn set_safety_policy(&mut self, policy: SafetyPolicy) {
        self.safety_policy = policy;
    }

    pub fn last_self_fix(&self) -> Option<SelfFixProbeResult> {
        self.last_self_fix.lock().unwrap().clone()
    }

    pub fn take_last_self_fix(&self) -> Option<SelfFixProbeResult> {
        self.last_self_fix.lock().unwrap().take()
    }

    pub fn preset_str(&self) -> &'static str {
        self.preset.unwrap_or_default().as_str()
    }

    pub fn current_host(&self) -> String {
        let u = self.current_url();
        extract_host_from_url(u.as_deref())
    }

    pub fn lookup_experience(&self, target_label: &str) -> Option<ExperienceEntry> {
        let store = self.experience_store.as_ref()?;
        store.lookup(self.preset_str(), &self.current_host(), target_label)
    }

    pub fn ocr_candidates_for_self_fix(&self, tgt: &str) -> Vec<OcrLine> {
        let all = self.ocr_candidates();
        if all.is_empty() {
            return Vec::new();
        }
        let url = self.current_url();
        let tried = self.tried_ocr_items();
        let targeted = candidates_for_target(&all, tgt, url.as_deref(), &tried);
        if !targeted.is_empty() {
            return targeted;
        }
        all.into_iter()
            .filter(|c| !tried.iter().any(|tr| tr.matches(c, url.as_deref())))
            .collect()
    }

    fn is_candidate_safe(
        &self,
        tgt: &str,
        desc_or_text: &str,
        bbox: &BoundingBox,
        sensitive_noul: Option<f64>,
        img: &PreparedImage,
    ) -> bool {
        if self.safety_policy.check_target(tgt).is_err() {
            return false;
        }
        if self.safety_policy.check_target(desc_or_text).is_err() {
            return false;
        }
        if is_sensitive_text(desc_or_text) {
            return false;
        }
        if sensitive_noul.unwrap_or(0.0) >= SELF_FIX_SENSITIVE_THRESHOLD {
            return false;
        }
        if self.is_popup_close_target(tgt) && !is_popup_box_width_ok(bbox) {
            return false;
        }
        let pt = map_box_center_to_screen_pt(bbox, img);
        if let Some(win_rect) = img.window_rect {
            if self.safety_policy.check_click(pt, win_rect).is_err() {
                return false;
            }
        } else {
            let (_lo, content_top_frac, _ws, _hs) = content_region_offset_and_scale_frac(img);
            let (_cx, cy) = bbox.center();
            if cy < content_top_frac
                || bbox.left < 0.0
                || bbox.right > 1.0
                || bbox.top < 0.0
                || bbox.bottom > 1.0
            {
                return false;
            }
        }
        true
    }

    pub fn set_ocr_candidates(&self, candidates: Vec<OcrLine>) {
        *self.ocr_candidates.lock().unwrap() = candidates;
    }

    pub fn ocr_candidates(&self) -> Vec<OcrLine> {
        self.ocr_candidates.lock().unwrap().clone()
    }

    pub fn record_tried_ocr_item(
        &self,
        text: impl Into<String>,
        center_frac: (f64, f64),
        url: Option<String>,
    ) {
        let item = TriedOcrItem {
            text: text.into(),
            center_frac,
            url,
        };
        let mut guard = self.tried_ocr_items.lock().unwrap();
        if !guard.iter().any(|existing| {
            existing.url == item.url
                && existing.text.trim() == item.text.trim()
                && (existing.center_frac.0 - item.center_frac.0).abs() <= 0.08
                && (existing.center_frac.1 - item.center_frac.1).abs() <= 0.08
        }) {
            guard.push(item);
        }
    }

    pub fn tried_ocr_items(&self) -> Vec<TriedOcrItem> {
        self.tried_ocr_items.lock().unwrap().clone()
    }

    pub fn build_item_criteria(candidates: &[OcrLine]) -> BTreeMap<String, Option<String>> {
        let mut criteria = BTreeMap::new();
        for (idx, line) in candidates.iter().enumerate() {
            criteria.insert(idx.to_string(), Some(line.format_criterion()));
        }
        criteria.insert(
            "none".to_string(),
            Some("None of the listed text items match the target".to_string()),
        );
        criteria
    }

    pub fn ask_ocr_item(
        &self,
        state: &str,
        img: &PreparedImage,
        target_label: &str,
        candidates: &[OcrLine],
    ) -> Result<Option<(OcrLine, f64)>, DeciderError> {
        if candidates.is_empty() {
            return Ok(None);
        }
        let criteria = Self::build_item_criteria(candidates);
        let mut questions = BTreeMap::new();
        questions.insert(
            "item".to_string(),
            Question::Choice {
                instructions: format!(
                    "Choose the OCR text item that matches '{}', or 'none' if none matches.",
                    target_label
                ),
                criteria,
            },
        );
        let req = Request {
            model: self.model.clone(),
            state: state.to_string(),
            images: vec![img.data_url.clone()],
            questions,
        };
        let resp = match self.ask_backend(&req) {
            Ok(r) => r,
            Err(_) => return Ok(None),
        };
        match resp.answers.get("item") {
            Some(Answer::Choice {
                choice, confidence, ..
            }) => {
                let c = choice.trim();
                if c.eq_ignore_ascii_case("none") || *confidence < LOW_CONFIDENCE_THRESHOLD {
                    return Ok(None);
                }
                if let Ok(idx) = c.parse::<usize>() {
                    if let Some(chosen) = candidates.get(idx) {
                        return Ok(Some((chosen.clone(), *confidence)));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    pub fn set_current_url(&self, url: Option<String>) {
        *self.current_url.lock().unwrap() = url;
    }

    pub fn current_url(&self) -> Option<String> {
        self.current_url.lock().unwrap().clone()
    }

    pub fn is_on_event_page(&self) -> bool {
        let guard = self.current_url.lock().unwrap();
        is_event_page_url(guard.as_deref())
    }

    pub fn scroll_direction(&self) -> ScrollDirection {
        *self.scroll_direction.lock().unwrap()
    }

    pub fn set_scroll_direction(&self, dir: ScrollDirection) {
        *self.scroll_direction.lock().unwrap() = dir;
    }

    pub fn closed_popups(&self) -> Vec<ClosedPopupMemory> {
        self.closed_popups.lock().unwrap().clone()
    }

    pub fn record_closed_popup(
        &self,
        description: impl Into<String>,
        center_frac: Option<(f64, f64)>,
    ) -> usize {
        let desc = description.into();
        let mut guard = self.closed_popups.lock().unwrap();
        for entry in guard.iter_mut() {
            let loc_match = match (entry.center_frac, center_frac) {
                (Some((x1, y1)), Some((x2, y2))) => {
                    (x1 - x2).abs() <= POPUP_LOCATION_PROXIMITY_FRAC
                        && (y1 - y2).abs() <= POPUP_LOCATION_PROXIMITY_FRAC
                }
                _ => false,
            };
            if loc_match || entry.description == desc {
                entry.count += 1;
                if entry.center_frac.is_none() {
                    entry.center_frac = center_frac;
                }
                return entry.count;
            }
        }
        guard.push(ClosedPopupMemory {
            description: desc,
            center_frac,
            count: 1,
        });
        1
    }

    pub fn is_popup_exhausted(
        &self,
        description: &str,
        center_frac: Option<(f64, f64)>,
    ) -> bool {
        let guard = self.closed_popups.lock().unwrap();
        guard.iter().any(|entry| {
            if entry.count < MAX_POPUP_CLOSE_COUNT {
                return false;
            }
            let loc_match = match (entry.center_frac, center_frac) {
                (Some((x1, y1)), Some((x2, y2))) => {
                    (x1 - x2).abs() <= POPUP_LOCATION_PROXIMITY_FRAC
                        && (y1 - y2).abs() <= POPUP_LOCATION_PROXIMITY_FRAC
                }
                _ => false,
            };
            loc_match || entry.description == description
        })
    }

    pub fn is_any_popup_exhausted(&self) -> bool {
        self.closed_popups
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.count >= MAX_POPUP_CLOSE_COUNT)
    }

    pub fn resolve_card_link_retry_box(
        &self,
        state: &str,
        img: &PreparedImage,
    ) -> Result<Option<(BoundingBox, f64, Option<f64>)>, DeciderError> {
        for desc in CARD_LINK_RETRY_DESCRIPTIONS {
            let (bbox, conf, sens) = self.ask_box(state, img, desc)?;
            if conf >= LOW_CONFIDENCE_THRESHOLD {
                return Ok(Some((bbox, conf, sens)));
            }
        }
        Ok(None)
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
        let up = self.scroll_direction() == ScrollDirection::Up;
        let zh = match self.preset {
            Some(Preset::EventsZh) => true,
            Some(_) => false,
            None => self
                .targets
                .iter()
                .any(|t| t.label == "關閉彈窗" || t.label == "活動卡片"),
        };
        match (zh, up) {
            (true, false) => "向下捲動",
            (true, true) => "向上捲動",
            (false, false) => "scroll down",
            (false, true) => "scroll up",
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
}

pub fn centers_within_tolerance(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() <= TRIED_POSITION_TOLERANCE_FRAC
        && (a.1 - b.1).abs() <= TRIED_POSITION_TOLERANCE_FRAC
}

#[derive(Debug, Clone)]
struct ProbedCandidate {
    candidate: String,
    bbox: BoundingBox,
    raw_conf: f64,
    effective_conf: f64,
    sensitive_noul: Option<f64>,
    safe: bool,
    lesson: String,
    ocr_item: Option<(String, (f64, f64))>,
}

impl<B: DecisionBackend> VisionDecider<B> {
    pub fn has_available_targets(&self, excluded_targets: &[String]) -> bool {
        let prev_popup_present = self
            .last_popup_present
            .lock()
            .unwrap()
            .unwrap_or(0.0);
        let popup_exhausted = self.is_any_popup_exhausted();
        self.targets.iter().any(|t| {
            if self.is_popup_close_target(&t.label)
                && (prev_popup_present < POPUP_PRESENT_THRESHOLD || popup_exhausted)
            {
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
        *self.last_self_fix.lock().unwrap() = None;
        *self.last_ocr_item_choice.lock().unwrap() = None;

        let (mut bbox, mut box_conf, mut sensitive_noul) = self.ask_box(state, img, tgt)?;
        let base_bbox = bbox;
        let base_conf = box_conf;
        let mut reasked = false;

        if box_conf < LOW_CONFIDENCE_THRESHOLD && self.self_fix_enabled {
            reasked = true;
            let budget = self
                .harness
                .as_ref()
                .map(|h| h.max_extra_requests_per_step)
                .unwrap_or(DEFAULT_MAX_EXTRA_REQUESTS_PER_STEP);
            let default_order = crate::harness::default_candidate_order();
            let order = self
                .harness
                .as_ref()
                .map(|h| h.candidate_order.as_slice())
                .unwrap_or(&default_order);

            let exp_entry = self.lookup_experience(tgt);
            let ocr_cands = self.ocr_candidates_for_self_fix(tgt);
            let is_legacy_only =
                self.harness.is_none() && exp_entry.is_none() && ocr_cands.is_empty();

            let mut budget_used = 0usize;
            let mut probed: Vec<ProbedCandidate> = Vec::new();

            for kind in order {
                if budget_used >= budget {
                    break;
                }
                match kind {
                    CandidateKind::Reask => {
                        let harness_reasks: Vec<String> = self
                            .harness
                            .as_ref()
                            .and_then(|h| h.reask_for(tgt).map(|s| s.to_vec()))
                            .unwrap_or_default();

                        if !harness_reasks.is_empty() {
                            for prompt in &harness_reasks {
                                if budget_used >= budget {
                                    break;
                                }
                                let (b, c, s) = self.ask_box(state, img, prompt)?;
                                budget_used += 1;
                                let safe = self.is_candidate_safe(tgt, prompt, &b, s, img);
                                probed.push(ProbedCandidate {
                                    candidate: "reask".to_string(),
                                    bbox: b,
                                    raw_conf: c,
                                    effective_conf: c,
                                    sensitive_noul: s,
                                    safe,
                                    lesson: prompt.clone(),
                                    ocr_item: None,
                                });
                                if harness_reasks.len() > 1
                                    && safe
                                    && c >= LOW_CONFIDENCE_THRESHOLD
                                {
                                    break;
                                }
                            }
                        } else if let Some(specific_descs) = card_primary_reask_descriptions(tgt) {
                            if is_legacy_only {
                                for desc in specific_descs {
                                    if budget_used >= budget {
                                        break;
                                    }
                                    let (b, c, s) = self.ask_box(state, img, desc)?;
                                    budget_used += 1;
                                    let safe = self.is_candidate_safe(tgt, desc, &b, s, img);
                                    probed.push(ProbedCandidate {
                                        candidate: "reask".to_string(),
                                        bbox: b,
                                        raw_conf: c,
                                        effective_conf: c,
                                        sensitive_noul: s,
                                        safe,
                                        lesson: (*desc).to_string(),
                                        ocr_item: None,
                                    });
                                    if safe && c >= LOW_CONFIDENCE_THRESHOLD {
                                        break;
                                    }
                                }
                            } else {
                                let desc = specific_descs.first().copied().unwrap_or(tgt);
                                let (b, c, s) = self.ask_box(state, img, desc)?;
                                budget_used += 1;
                                let safe = self.is_candidate_safe(tgt, desc, &b, s, img);
                                probed.push(ProbedCandidate {
                                    candidate: "reask".to_string(),
                                    bbox: b,
                                    raw_conf: c,
                                    effective_conf: c,
                                    sensitive_noul: s,
                                    safe,
                                    lesson: desc.to_string(),
                                    ocr_item: None,
                                });
                            }
                        } else {
                            let longer = self
                                .find_target_description(tgt)
                                .map(|desc| format!("{} ({})", tgt, desc))
                                .unwrap_or_else(|| {
                                    format!("Locate the interactive element: {}", tgt)
                                });
                            let (b, c, s) = self.ask_box(state, img, &longer)?;
                            budget_used += 1;
                            let safe = self.is_candidate_safe(tgt, &longer, &b, s, img);
                            probed.push(ProbedCandidate {
                                candidate: "reask".to_string(),
                                bbox: b,
                                raw_conf: c,
                                effective_conf: c,
                                sensitive_noul: s,
                                safe,
                                lesson: longer,
                                ocr_item: None,
                            });
                        }
                    }
                    CandidateKind::Experience => {
                        if let Some(ref entry) = exp_entry {
                            if !entry.lesson.trim().is_empty() && budget_used < budget {
                                let exp_prompt = if entry.lesson.contains(tgt) {
                                    entry.lesson.clone()
                                } else {
                                    format!("{} ({})", tgt, entry.lesson)
                                };
                                let (b, c, s) = self.ask_box(state, img, &exp_prompt)?;
                                budget_used += 1;
                                let safe = self.is_candidate_safe(tgt, &entry.lesson, &b, s, img);
                                probed.push(ProbedCandidate {
                                    candidate: "experience".to_string(),
                                    bbox: b,
                                    raw_conf: c,
                                    effective_conf: c,
                                    sensitive_noul: s,
                                    safe,
                                    lesson: entry.lesson.clone(),
                                    ocr_item: None,
                                });
                            }
                        }
                    }
                    CandidateKind::Zoom => {
                        let legacy_already_resolved = is_legacy_only
                            && probed
                                .iter()
                                .any(|p| p.safe && p.raw_conf >= LOW_CONFIDENCE_THRESHOLD);
                        let skip_aux = is_auxiliary_target(tgt) && self.harness.is_none();
                        // Legacy (no harness / no experience / no OCR) only zoomed for card
                        // targets; keep that so runs without a harness send the same requests.
                        let legacy_no_zoom =
                            is_legacy_only && card_primary_reask_descriptions(tgt).is_none();
                        if !legacy_already_resolved
                            && !skip_aux
                            && !legacy_no_zoom
                            && budget_used < budget
                        {
                            if self.is_popup_close_target(tgt) {
                                if let Ok(corner_img) =
                                    prepare_corner_zoom_2x(img, CornerRegion::BottomRight)
                                {
                                    if let Some((ref mut log, step)) = audit_log {
                                        log.record(
                                            &corner_img,
                                            format!("step_{}_popup_bottom_right_2x", step),
                                        );
                                    }
                                    let crop_desc = POPUP_REASK_COOKIE_GOT_IT_DESC;
                                    let (cb, cc, cs) =
                                        self.ask_box(state, &corner_img, crop_desc)?;
                                    budget_used += 1;
                                    let mapped_box = map_corner_box_to_window_box(
                                        &cb,
                                        img,
                                        CornerRegion::BottomRight,
                                    );
                                    let safe = self.is_candidate_safe(
                                        tgt,
                                        crop_desc,
                                        &mapped_box,
                                        cs,
                                        img,
                                    );
                                    probed.push(ProbedCandidate {
                                        candidate: "zoom".to_string(),
                                        bbox: mapped_box,
                                        raw_conf: cc,
                                        effective_conf: cc,
                                        sensitive_noul: cs,
                                        safe,
                                        lesson: crop_desc.to_string(),
                                        ocr_item: None,
                                    });
                                }
                            } else if let Ok(content_img) = prepare_content_zoom_2x(img) {
                                if let Some((ref mut log, step)) = audit_log {
                                    log.record(&content_img, format!("step_{}_content_2x", step));
                                }
                                let crop_desc = self
                                    .harness
                                    .as_ref()
                                    .and_then(|h| h.reask_for(tgt)?.first().map(|s| s.as_str()))
                                    .or_else(|| {
                                        card_primary_reask_descriptions(tgt)
                                            .and_then(|d| d.first().copied())
                                    })
                                    .or_else(|| self.find_target_description(tgt))
                                    .unwrap_or(tgt)
                                    .to_string();
                                let (cb, cc, cs) =
                                    self.ask_box(state, &content_img, &crop_desc)?;
                                budget_used += 1;
                                let mapped_box = map_content_box_to_window_box(&cb, img);
                                let safe = self.is_candidate_safe(
                                    tgt,
                                    &crop_desc,
                                    &mapped_box,
                                    cs,
                                    img,
                                );
                                probed.push(ProbedCandidate {
                                    candidate: "zoom".to_string(),
                                    bbox: mapped_box,
                                    raw_conf: cc,
                                    effective_conf: cc,
                                    sensitive_noul: cs,
                                    safe,
                                    lesson: crop_desc,
                                    ocr_item: None,
                                });
                            }
                        }
                    }
                    CandidateKind::Ocr => {
                        if !ocr_cands.is_empty() && budget_used < budget {
                            let chosen_opt = self.ask_ocr_item(state, img, tgt, &ocr_cands)?;
                            budget_used += 1;
                            if let Some((chosen_line, ocr_conf)) = chosen_opt {
                                let ocr_bbox = chosen_line.bbox();
                                let safe = self.is_candidate_safe(
                                    tgt,
                                    &chosen_line.text,
                                    &ocr_bbox,
                                    None,
                                    img,
                                );
                                probed.push(ProbedCandidate {
                                    candidate: "ocr".to_string(),
                                    bbox: ocr_bbox,
                                    raw_conf: ocr_conf,
                                    effective_conf: ocr_conf,
                                    sensitive_noul: None,
                                    safe,
                                    lesson: chosen_line.text.clone(),
                                    ocr_item: Some((chosen_line.text.clone(), chosen_line.center())),
                                });
                            } else {
                                probed.push(ProbedCandidate {
                                    candidate: "ocr".to_string(),
                                    bbox: base_bbox,
                                    raw_conf: 0.0,
                                    effective_conf: 0.0,
                                    sensitive_noul: None,
                                    safe: false,
                                    lesson: String::new(),
                                    ocr_item: None,
                                });
                            }
                        }
                    }
                }
            }

            // Compute OCR center & box center agreement bonus within TRIED_POSITION_TOLERANCE_FRAC
            let mut ocr_centers: Vec<(f64, f64)> = probed
                .iter()
                .filter(|p| p.candidate == "ocr" && p.ocr_item.is_some())
                .map(|p| p.bbox.center())
                .collect();
            if ocr_centers.is_empty() {
                for line in &ocr_cands {
                    ocr_centers.push(line.center());
                }
            }

            let mut box_centers: Vec<(f64, f64)> = probed
                .iter()
                .filter(|p| p.candidate != "ocr" && p.raw_conf > 0.0)
                .map(|p| p.bbox.center())
                .collect();
            if base_conf > 0.0 {
                box_centers.push(base_bbox.center());
            }

            for p in &mut probed {
                if p.raw_conf <= 0.0 {
                    continue;
                }
                let c_center = p.bbox.center();
                let agrees = if p.candidate == "ocr" {
                    box_centers
                        .iter()
                        .any(|&bc| centers_within_tolerance(c_center, bc))
                } else {
                    ocr_centers
                        .iter()
                        .any(|&oc| centers_within_tolerance(c_center, oc))
                };
                if agrees {
                    p.effective_conf = (p.raw_conf + SELF_FIX_OCR_BOX_BONUS).min(1.0);
                }
            }

            let tried: Vec<SelfFixTriedEntry> = probed
                .iter()
                .map(|p| SelfFixTriedEntry {
                    candidate: p.candidate.clone(),
                    confidence: p.effective_conf,
                })
                .collect();

            let best_safe = probed
                .iter()
                .filter(|p| p.safe)
                .max_by(|a, b| {
                    a.effective_conf
                        .partial_cmp(&b.effective_conf)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned();

            let mut chosen = None;
            let mut chosen_lesson = None;

            if let Some(best) = best_safe {
                if best.effective_conf >= LOW_CONFIDENCE_THRESHOLD {
                    bbox = best.bbox;
                    box_conf = best.effective_conf;
                    sensitive_noul = best.sensitive_noul.or(sensitive_noul);
                    chosen = Some(best.candidate.clone());
                    chosen_lesson = Some(best.lesson.clone());
                    if let Some((text, center)) = best.ocr_item {
                        *self.last_ocr_item_choice.lock().unwrap() = Some((text, center));
                    }
                } else {
                    bbox = best.bbox;
                    box_conf = best.effective_conf;
                    sensitive_noul = best.sensitive_noul.or(sensitive_noul);
                }
            } else if let Some(any_unsafe) = probed.last() {
                // Keep sensitive_noul if a sensitive candidate was probed so safety checks can observe it if needed,
                // while keeping box_conf at base_conf (< LOW_CONFIDENCE_THRESHOLD) so it is never adopted.
                sensitive_noul = sensitive_noul.or(any_unsafe.sensitive_noul);
            }

            *self.last_self_fix.lock().unwrap() = Some(SelfFixProbeResult {
                target: tgt.to_string(),
                tried,
                chosen,
                chosen_lesson,
                base_conf,
                budget_used,
            });
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
        let popup_close_excluded =
            excluded_targets.iter().any(|ex| ex == popup_close) || self.is_any_popup_exhausted();

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
        let scroll_desc = if self.scroll_direction() == ScrollDirection::Up {
            "Scroll up the page to reveal content above"
        } else {
            "Scroll down the page to reveal more content"
        };
        kind_criteria.insert("scroll".to_string(), Some(scroll_desc.to_string()));
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
        let popup_exhausted = self.is_any_popup_exhausted();
        for t in &self.targets {
            if self.is_popup_close_target(&t.label)
                && (prev_popup_present < POPUP_PRESENT_THRESHOLD || popup_exhausted)
            {
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
        if self.preset == Some(Preset::EventsZh) || self.is_on_event_page() {
            questions.insert(
                "footer".to_string(),
                Question::Noul {
                    instructions: FOOTER_QUESTION_INSTRUCTIONS.to_string(),
                },
            );
        }
        questions.insert(
            "login_form".to_string(),
            Question::Noul {
                instructions: LOGIN_FORM_QUESTION_INSTRUCTIONS.to_string(),
            },
        );

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

        let footer_noul = match resp1.answers.get("footer") {
            Some(Answer::Noul { noul }) => Some(*noul),
            Some(_) => return Err(DeciderError::UnexpectedAnswerType("footer".to_string())),
            None => None,
        };
        if footer_noul.unwrap_or(0.0) >= FOOTER_DETECTED_THRESHOLD {
            self.set_scroll_direction(ScrollDirection::Up);
        }

        let login_form_noul = match resp1.answers.get("login_form") {
            Some(Answer::Noul { noul }) => Some(*noul),
            Some(_) => return Err(DeciderError::UnexpectedAnswerType("login_form".to_string())),
            None => None,
        };

        let action_kind = ActionKind::from_choice(&kind_choice)
            .ok_or_else(|| DeciderError::UnknownActionKind(kind_choice.clone()))?;

        if login_form_noul.unwrap_or(0.0) >= LOGIN_FORM_DETECTED_THRESHOLD {
            return Ok(Decision {
                kind: ActionKind::Wait,
                kind_confidence: kind_conf,
                target: target_choice,
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
                footer_noul,
                login_form_noul,
                click_source: None,
                ocr_item_text: None,
                ocr_item_center_frac: None,
                self_fix: None,
            });
        }

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
                footer_noul,
                login_form_noul,
                click_source: None,
                ocr_item_text: None,
                ocr_item_center_frac: None,
                self_fix: None,
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
                    footer_noul,
                    login_form_noul,
                    click_source: None,
                    ocr_item_text: None,
                    ocr_item_center_frac: None,
                    self_fix: None,
                });
            }

            if is_ocr_text_target(&tgt) && self.harness.is_none() {
                let candidates = candidates_for_target(
                    &self.ocr_candidates(),
                    &tgt,
                    self.current_url().as_deref(),
                    &self.tried_ocr_items(),
                );
                if !candidates.is_empty() {
                    let chosen_from_resp1 = match resp1.answers.get("item") {
                        Some(Answer::Choice {
                            choice, confidence, ..
                        }) => {
                            let c = choice.trim();
                            if !c.eq_ignore_ascii_case("none")
                                && *confidence >= LOW_CONFIDENCE_THRESHOLD
                            {
                                c.parse::<usize>()
                                    .ok()
                                    .and_then(|idx| candidates.get(idx).cloned())
                                    .map(|line| (line, *confidence))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    let chosen_item = if resp1.answers.contains_key("item") {
                        chosen_from_resp1
                    } else {
                        self.ask_ocr_item(&state, img, &tgt, &candidates)
                            .ok()
                            .flatten()
                    };

                    if let Some((chosen, item_conf)) = chosen_item {
                        let bbox = chosen.bbox();
                        let pt = Some(map_box_center_to_screen_pt(&bbox, img));
                        let fill_text = self.type_values.get(&tgt).cloned();
                        return Ok(Decision {
                            kind: ActionKind::Click,
                            kind_confidence: kind_conf,
                            target: Some(tgt),
                            click_point_screen_pt: pt,
                            bbox: Some(bbox),
                            box_confidence: Some(item_conf),
                            confidence: item_conf,
                            reasked: false,
                            text_to_type: fill_text,
                            results_noul,
                            sensitive_noul: None,
                            closeish_noul: None,
                            popup_present_noul,
                            footer_noul,
                            login_form_noul,
                            click_source: Some("ocr".to_string()),
                            ocr_item_text: Some(chosen.text.clone()),
                            ocr_item_center_frac: Some(chosen.center()),
                            self_fix: None,
                        });
                    }
                }
            }

            let (bbox, box_conf, sensitive_noul, reasked) =
                self.resolve_click_box(&state, img, &tgt, audit_log)?;
            let self_fix = self.take_last_self_fix();
            let ocr_choice = self.last_ocr_item_choice.lock().unwrap().take();

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
            let (click_source, ocr_item_text, ocr_item_center_frac) =
                if let Some((text, center)) = ocr_choice {
                    (Some("ocr".to_string()), Some(text), Some(center))
                } else {
                    (Some("box".to_string()), None, None)
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
                footer_noul,
                login_form_noul,
                click_source,
                ocr_item_text,
                ocr_item_center_frac,
                self_fix,
            });
        }

        if action_kind == ActionKind::Scroll && self.is_on_event_page() {
            for (priority_label, priority_desc) in [
                ("立即報名按鈕", EVENT_PAGE_REGISTER_DESC),
                ("票種選項", EVENT_PAGE_TICKET_DESC),
            ] {
                let has_target = self.targets.iter().any(|t| t.label == priority_label);
                let is_excluded = excluded_targets.iter().any(|ex| ex == priority_label);
                if has_target && !is_excluded {
                    let candidates = candidates_for_target(
                        &self.ocr_candidates(),
                        priority_label,
                        self.current_url().as_deref(),
                        &self.tried_ocr_items(),
                    );
                    if !candidates.is_empty() {
                        if let Ok(Some((chosen, item_conf))) =
                            self.ask_ocr_item(&state, img, priority_label, &candidates)
                        {
                            let bbox = chosen.bbox();
                            let pt = Some(map_box_center_to_screen_pt(&bbox, img));
                            return Ok(Decision {
                                kind: ActionKind::Click,
                                kind_confidence: kind_conf,
                                target: Some(priority_label.to_string()),
                                click_point_screen_pt: pt,
                                bbox: Some(bbox),
                                box_confidence: Some(item_conf),
                                confidence: item_conf,
                                reasked: true,
                                text_to_type: None,
                                results_noul,
                                sensitive_noul: None,
                                closeish_noul: None,
                                popup_present_noul,
                                footer_noul,
                                login_form_noul,
                                click_source: Some("ocr".to_string()),
                                ocr_item_text: Some(chosen.text.clone()),
                                ocr_item_center_frac: Some(chosen.center()),
                                self_fix: None,
                            });
                        }
                    }
                    let (bbox, box_conf, sensitive_noul) =
                        self.ask_box(&state, img, priority_desc)?;
                    if box_conf >= LOW_CONFIDENCE_THRESHOLD {
                        let pt = Some(map_box_center_to_screen_pt(&bbox, img));
                        return Ok(Decision {
                            kind: ActionKind::Click,
                            kind_confidence: kind_conf,
                            target: Some(priority_label.to_string()),
                            click_point_screen_pt: pt,
                            bbox: Some(bbox),
                            box_confidence: Some(box_conf),
                            confidence: box_conf,
                            reasked: true,
                            text_to_type: None,
                            results_noul,
                            sensitive_noul,
                            closeish_noul: None,
                            popup_present_noul,
                            footer_noul,
                            login_form_noul,
                            click_source: Some("box".to_string()),
                            ocr_item_text: None,
                            ocr_item_center_frac: None,
                            self_fix: None,
                        });
                    }
                }
            }
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
            footer_noul,
            login_form_noul,
            click_source: None,
            ocr_item_text: None,
            ocr_item_center_frac: None,
            self_fix: None,
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
        let mut matched_desc = tgt.clone();
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
                matched_desc = desc.to_string();
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
                    let mapped_box = map_corner_box_to_window_box(&cb, img, corner);
                    bbox = mapped_box;
                    box_conf = cc;
                    closeish_noul = cl.or(closeish_noul);
                    matched_desc = desc.to_string();
                    if box_conf >= LOW_CONFIDENCE_THRESHOLD {
                        break;
                    }
                }
            }
        }

        let mut valid = box_conf >= LOW_CONFIDENCE_THRESHOLD && is_popup_box_width_ok(&bbox);
        if !valid && box_conf >= LOW_CONFIDENCE_THRESHOLD {
            // Reject boxes larger than 15% of window width (never click link or ad body)
            box_conf = 0.0;
        }

        let center = bbox.center();
        if valid && self.is_popup_exhausted(&matched_desc, Some(center)) {
            valid = false;
            box_conf = 0.0;
        }

        if valid
            && closeish_noul.unwrap_or(0.0) >= 0.6
            && decision.popup_present_noul.unwrap_or(1.0) >= POPUP_PRESENT_THRESHOLD
        {
            self.record_closed_popup(matched_desc, Some(center));
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
        decision.click_source = Some("box".to_string());
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

    pub fn ask_price_box(
        &self,
        state: &str,
        img: &PreparedImage,
    ) -> Result<(BoundingBox, f64), DeciderError> {
        let mut questions = BTreeMap::new();
        questions.insert(
            "price_box".to_string(),
            Question::Box {
                instructions: PRICE_TEXT_BOX_DESC.to_string(),
            },
        );
        let req = Request {
            model: self.model.clone(),
            state: state.to_string(),
            images: vec![img.data_url.clone()],
            questions,
        };
        let resp = self.ask_backend(&req)?;
        match resp.answers.get("price_box") {
            Some(Answer::Box { bbox, confidence }) => Ok((*bbox, *confidence)),
            Some(_) => Err(DeciderError::UnexpectedAnswerType("price_box".to_string())),
            None => Err(DeciderError::MissingAnswer("price_box".to_string())),
        }
    }
}
