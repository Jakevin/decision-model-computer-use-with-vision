use crate::api::BoundingBox;
use crate::capture::Rect;
use crate::decider::map_box_center_to_screen_pt;
use crate::logging::redact;
use crate::privacy::{content_region_offset_and_scale_frac, is_sensitive_text, PreparedImage};
use image::RgbaImage;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use thiserror::Error;

pub const MIN_OCR_CONFIDENCE: f64 = 0.3;
pub const MAX_OCR_CANDIDATES: usize = 60;
pub const TRIED_POSITION_TOLERANCE_FRAC: f64 = 0.08;

static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrLine {
    pub text: String,
    pub conf: f64,
    pub top: f64,
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
}

impl OcrLine {
    pub fn new(
        text: impl Into<String>,
        conf: f64,
        top: f64,
        left: f64,
        bottom: f64,
        right: f64,
    ) -> Self {
        Self {
            text: text.into(),
            conf,
            top,
            left,
            bottom,
            right,
        }
    }

    pub fn from_vision(
        text: impl Into<String>,
        conf: f64,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
    ) -> Self {
        let (top, left, bottom, right) = vision_to_topleft(x, y, w, h);
        Self {
            text: text.into(),
            conf,
            top,
            left,
            bottom,
            right,
        }
    }

    pub fn bbox(&self) -> BoundingBox {
        BoundingBox {
            top: self.top,
            left: self.left,
            bottom: self.bottom,
            right: self.right,
        }
    }

    pub fn center(&self) -> (f64, f64) {
        ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)
    }

    pub fn height(&self) -> f64 {
        (self.bottom - self.top).max(0.0)
    }

    pub fn region(&self) -> &'static str {
        let (cx, cy) = self.center();
        region_label(cx, cy)
    }

    pub fn format_criterion(&self) -> String {
        let masked = mask_sensitive_text(&self.text);
        format!("'{}' ({})", masked, self.region())
    }
}

pub fn vision_to_topleft(x: f64, y: f64, w: f64, h: f64) -> (f64, f64, f64, f64) {
    let top = 1.0 - y - h;
    let left = x;
    let bottom = 1.0 - y;
    let right = x + w;
    (top, left, bottom, right)
}

pub fn region_label(cx: f64, cy: f64) -> &'static str {
    let row = if cy < 1.0 / 3.0 {
        0
    } else if cy < 2.0 / 3.0 {
        1
    } else {
        2
    };
    let col = if cx < 1.0 / 3.0 {
        0
    } else if cx < 2.0 / 3.0 {
        1
    } else {
        2
    };
    match (row, col) {
        (0, 0) => "top-left",
        (0, 1) => "top-center",
        (0, 2) => "top-right",
        (1, 0) => "middle-left",
        (1, 1) => "center",
        (1, 2) => "middle-right",
        (2, 0) => "bottom-left",
        (2, 1) => "bottom-center",
        _ => "bottom-right",
    }
}

pub fn mask_sensitive_text(text: &str) -> String {
    if is_sensitive_text(text) {
        "***".to_string()
    } else {
        redact(text, None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrError {
    #[error("OCR helper unavailable: {0}")]
    HelperUnavailable(String),
    #[error("OCR IO error: {0}")]
    Io(String),
    #[error("OCR process failed: {0}")]
    Process(String),
    #[error("OCR parse error: {0}")]
    Parse(String),
}

pub trait Ocr {
    fn recognize(&self, img: &RgbaImage) -> Result<Vec<OcrLine>, OcrError>;
}

impl<T: Ocr + ?Sized> Ocr for Box<T> {
    fn recognize(&self, img: &RgbaImage) -> Result<Vec<OcrLine>, OcrError> {
        (**self).recognize(img)
    }
}

pub struct FakeOcr {
    queue: Mutex<VecDeque<Result<Vec<OcrLine>, OcrError>>>,
    default_result: Result<Vec<OcrLine>, OcrError>,
    calls: AtomicUsize,
}

impl Default for FakeOcr {
    fn default() -> Self {
        Self::empty()
    }
}

impl FakeOcr {
    pub fn empty() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            default_result: Ok(Vec::new()),
            calls: AtomicUsize::new(0),
        }
    }

    pub fn new(lines: Vec<OcrLine>) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            default_result: Ok(lines),
            calls: AtomicUsize::new(0),
        }
    }

    pub fn sequence(steps: Vec<Vec<OcrLine>>) -> Self {
        let last = steps.last().cloned().unwrap_or_default();
        let queue = steps.into_iter().map(Ok).collect();
        Self {
            queue: Mutex::new(queue),
            default_result: Ok(last),
            calls: AtomicUsize::new(0),
        }
    }

    pub fn from_results(results: Vec<Result<Vec<OcrLine>, OcrError>>) -> Self {
        let last = results.last().cloned().unwrap_or_else(|| Ok(Vec::new()));
        Self {
            queue: Mutex::new(VecDeque::from(results)),
            default_result: last,
            calls: AtomicUsize::new(0),
        }
    }

    pub fn failing(msg: impl Into<String>) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            default_result: Err(OcrError::Process(msg.into())),
            calls: AtomicUsize::new(0),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Ocr for FakeOcr {
    fn recognize(&self, _img: &RgbaImage) -> Result<Vec<OcrLine>, OcrError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut guard = self.queue.lock().unwrap();
        if let Some(next) = guard.pop_front() {
            next
        } else {
            self.default_result.clone()
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct VisionRawLine {
    text: String,
    confidence: f64,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

pub fn parse_vision_jsonl(raw_output: &str) -> Result<Vec<OcrLine>, OcrError> {
    let mut lines = Vec::new();
    for raw_line in raw_output.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed: VisionRawLine =
            serde_json::from_str(trimmed).map_err(|e| OcrError::Parse(e.to_string()))?;
        let (top, left, bottom, right) =
            vision_to_topleft(parsed.x, parsed.y, parsed.w, parsed.h);
        lines.push(OcrLine {
            text: parsed.text,
            conf: parsed.confidence,
            top,
            left,
            bottom,
            right,
        });
    }
    Ok(lines)
}

#[derive(Debug, Clone)]
pub struct VisionOcr {
    helper_path: Option<PathBuf>,
}

impl Default for VisionOcr {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionOcr {
    pub fn new() -> Self {
        let compiled_bin = option_env!("VISION_OCR_BIN").unwrap_or("");
        let helper_path = if !compiled_bin.trim().is_empty() {
            let p = PathBuf::from(compiled_bin);
            if p.exists() {
                Some(p)
            } else {
                None
            }
        } else {
            None
        };
        Self { helper_path }
    }

    pub fn with_helper_path(path: impl Into<PathBuf>) -> Self {
        Self {
            helper_path: Some(path.into()),
        }
    }

    pub fn helper_path(&self) -> Option<&std::path::Path> {
        self.helper_path.as_deref()
    }
}

impl Ocr for VisionOcr {
    fn recognize(&self, img: &RgbaImage) -> Result<Vec<OcrLine>, OcrError> {
        let Some(ref helper) = self.helper_path else {
            return Err(OcrError::HelperUnavailable(
                "vision_ocr helper binary was not built".to_string(),
            ));
        };
        if !helper.exists() {
            return Err(OcrError::HelperUnavailable(format!(
                "vision_ocr helper binary not found at {}",
                helper.display()
            )));
        }

        let seq = TEMP_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let temp_path = std::env::temp_dir().join(format!("decision_cu_ocr_{}_{}.png", pid, seq));

        img.save(&temp_path)
            .map_err(|e| OcrError::Io(e.to_string()))?;

        let output_res = Command::new(helper).arg(&temp_path).output();
        let _ = std::fs::remove_file(&temp_path);

        let output = output_res.map_err(|e| OcrError::Io(e.to_string()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(OcrError::Process(format!(
                "status {}: {}",
                output.status, stderr
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_vision_jsonl(&stdout)
    }
}

pub fn reading_order(lines: &mut [OcrLine]) {
    if lines.len() <= 1 {
        return;
    }
    let mut heights: Vec<f64> = lines
        .iter()
        .map(|l| l.height())
        .filter(|h| *h > 0.0)
        .collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_h = if heights.is_empty() {
        0.03
    } else {
        heights[heights.len() / 2].max(0.01)
    };
    let row_tol = med_h * 0.5;

    lines.sort_by(|a, b| {
        let cy_a = a.center().1;
        let cy_b = b.center().1;
        cy_a.partial_cmp(&cy_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut start = 0;
    while start < lines.len() {
        let row_cy = lines[start].center().1;
        let mut end = start + 1;
        while end < lines.len() && (lines[end].center().1 - row_cy).abs() <= row_tol {
            end += 1;
        }
        lines[start..end].sort_by(|a, b| {
            a.left
                .partial_cmp(&b.left)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        start = end;
    }
}

pub fn filter_ocr_candidates<F>(
    lines: &[OcrLine],
    window_img: &PreparedImage,
    win_rect: Rect,
    tab_strip_pt: f64,
    mut is_chrome_at_pt: F,
) -> Vec<OcrLine>
where
    F: FnMut((f64, f64)) -> bool,
{
    let (_left_off, content_top_frac, _w_scale, _h_scale) =
        content_region_offset_and_scale_frac(window_img);
    let strip_limit_pt = win_rect.y + tab_strip_pt;

    let mut kept = Vec::new();
    for line in lines {
        if line.conf < MIN_OCR_CONFIDENCE {
            continue;
        }
        let trimmed = line.text.trim();
        if trimmed.is_empty() {
            continue;
        }
        if line.left < 0.0
            || line.right > 1.0
            || line.top < 0.0
            || line.bottom > 1.0
            || line.left >= line.right
            || line.top >= line.bottom
        {
            continue;
        }
        let (_cx, cy) = line.center();
        if line.top <= content_top_frac || cy <= content_top_frac {
            continue;
        }

        let bbox = line.bbox();
        let pt = map_box_center_to_screen_pt(&bbox, window_img);
        if pt.0 < win_rect.x
            || pt.0 > win_rect.x + win_rect.width
            || pt.1 <= strip_limit_pt
            || pt.1 > win_rect.y + win_rect.height
        {
            continue;
        }

        if !is_chrome_at_pt(pt) {
            continue;
        }

        let mut masked_line = line.clone();
        masked_line.text = mask_sensitive_text(trimmed);
        kept.push(masked_line);
    }

    reading_order(&mut kept);
    if kept.len() > MAX_OCR_CANDIDATES {
        kept.truncate(MAX_OCR_CANDIDATES);
    }
    kept
}

pub fn has_login_ocr_lines(lines: &[OcrLine]) -> bool {
    let mut has_email_or_account = false;
    let mut has_password = false;
    let mut has_login_word = false;
    let mut has_explicit_login_form = false;

    for line in lines {
        let t = line.text.trim();
        let lower = t.to_lowercase();
        if t.contains("電子信箱")
            || t.contains("信箱")
            || t.contains("帳號")
            || lower.contains("email")
        {
            has_email_or_account = true;
        }
        if t.contains("密碼") || lower.contains("password") {
            has_password = true;
        }
        if t.contains("登入表單")
            || t.contains("會員登入")
            || t.contains("請先登入")
            || t.contains("登入帳號")
            || lower.contains("login form")
            || lower.contains("sign in to")
        {
            has_explicit_login_form = true;
        }
        if t.contains("登入") || lower.contains("login") || lower.contains("sign in") {
            has_login_word = true;
        }
    }

    (has_email_or_account && has_password)
        || has_explicit_login_form
        || (has_login_word && (has_email_or_account || has_password))
}

pub fn is_ocr_text_target(target_label: &str) -> bool {
    let t = target_label.trim();
    t == "活動卡片"
        || t == "立即報名按鈕"
        || t == "票種選項"
        || t.eq_ignore_ascii_case("upcoming event card")
        || t.eq_ignore_ascii_case("tickets link")
        || t.eq_ignore_ascii_case("cheapest ticket option")
}

fn is_register_button_text(text: &str) -> bool {
    let lower = text.to_lowercase();
    text.contains("立即報名")
        || text.contains("報名")
        || text.contains("購票")
        || text.contains("搶票")
        || lower.contains("register")
        || lower.contains("buy tickets")
        || lower.contains("get tickets")
        || lower.contains("tickets")
}

fn is_ticket_option_text(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("nt$")
        || text.contains('$')
        || text.contains('元')
        || text.contains('票')
        || text.contains("免費")
        || lower.contains("ticket")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriedOcrItem {
    pub text: String,
    pub center_frac: (f64, f64),
    pub url: Option<String>,
}

impl TriedOcrItem {
    pub fn matches(&self, line: &OcrLine, current_url: Option<&str>) -> bool {
        if self.url.as_deref() != current_url {
            return false;
        }
        if self.text.trim() != line.text.trim() {
            return false;
        }
        let (cx, cy) = line.center();
        (self.center_frac.0 - cx).abs() <= TRIED_POSITION_TOLERANCE_FRAC
            && (self.center_frac.1 - cy).abs() <= TRIED_POSITION_TOLERANCE_FRAC
    }
}

pub fn candidates_for_target(
    candidates: &[OcrLine],
    target_label: &str,
    current_url: Option<&str>,
    tried: &[TriedOcrItem],
) -> Vec<OcrLine> {
    if !is_ocr_text_target(target_label) {
        return Vec::new();
    }
    let untried: Vec<OcrLine> = candidates
        .iter()
        .filter(|c| !tried.iter().any(|tr| tr.matches(c, current_url)))
        .cloned()
        .collect();

    let t = target_label.trim();
    if t == "立即報名按鈕" || t.eq_ignore_ascii_case("tickets link") {
        untried
            .into_iter()
            .filter(|c| is_register_button_text(&c.text))
            .collect()
    } else if t == "票種選項" || t.eq_ignore_ascii_case("cheapest ticket option") {
        untried
            .into_iter()
            .filter(|c| is_ticket_option_text(&c.text))
            .collect()
    } else if t == "活動卡片" || t.eq_ignore_ascii_case("upcoming event card") {
        let non_button: Vec<OcrLine> = untried
            .iter()
            .filter(|c| {
                let s = c.text.trim();
                !(s == "立即報名" || s == "報名" || s == "購票" || s == "搶票")
            })
            .cloned()
            .collect();
        if non_button.is_empty() {
            untried
        } else {
            non_button
        }
    } else {
        Vec::new()
    }
}
