use crate::api::{BoundingBox, DecisionBackend};
use crate::browser::{Browser, DefaultAllowedBrowser};
use crate::capture::{changed_fraction, is_changed, CaptureError, Rect, Screen};
use crate::decider::{
    compute_excluded_targets, format_state, is_login_url, low_box_conf_exclude_steps,
    map_box_center_to_screen_pt, ActionKind, DeciderError, Decision, DoneCondition,
    ScrollDirection, VisionDecider, FOOTER_DETECTED_THRESHOLD, LOGIN_FORM_DETECTED_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD, POPUP_PRESENT_THRESHOLD,
};
use crate::input::{
    Input, InputError, KeyCode, DEFAULT_SCROLL_DELTA_PX, DEFAULT_SCROLL_UP_DELTA_PX,
};
use crate::logging::RunLogger;
use crate::ocr::{filter_ocr_candidates, has_login_ocr_lines, Ocr, OcrLine};
use crate::privacy::{
    content_region_offset_and_scale_frac, encode_jpeg, is_sensitive_text, prepare_image, AuditLog,
    PreparedImage, PrivacyError, JPEG_QUALITY,
};
use crate::safety::{
    is_sensitive_fill_label, resolve_allowed_url_prefixes, SafetyError, SafetyPolicy,
};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

pub const FLIGHTS_DEFAULT_URL_PREFIX: &str = "https://www.google.com/travel/flights";
pub const RESULTS_DONE_CONSECUTIVE_THRESHOLD: f64 = 0.8;
pub const RESULTS_DONE_CONSECUTIVE_COUNT: usize = 2;
pub const RESULTS_MODEL_DONE_THRESHOLD: f64 = 0.5;
pub const SENSITIVE_BLOCK_THRESHOLD: f64 = 0.5;
pub const SENSITIVE_EXCLUDE_STEPS: usize = 3;
pub const LOW_BOX_CONF_EXCLUDE_STEPS: usize = 1000;
pub const MAX_CONSECUTIVE_NOOP_STEPS: usize = 3;
pub const DISMISS_CLOSEISH_THRESHOLD: f64 = 0.6;
pub const MAX_TAB_EVENTS: usize = 2;
pub const MAX_FRONTMOST_FAILURES: usize = 2;
pub const ACTIVATE_CHROME_SLEEP_MS: u64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timing {
    pub start_url_wait: Duration,
    pub activate_chrome_wait: Duration,
    pub before_fill_wait: Duration,
    pub after_fill_wait: Duration,
    pub post_action_wait: Duration,
    pub dropdown_reclick_wait: Duration,
    pub step_wait: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            start_url_wait: Duration::from_secs(5),
            activate_chrome_wait: Duration::from_millis(ACTIVATE_CHROME_SLEEP_MS),
            before_fill_wait: Duration::from_millis(800),
            after_fill_wait: Duration::from_millis(1000),
            post_action_wait: Duration::from_millis(1000),
            dropdown_reclick_wait: Duration::from_millis(700),
            step_wait: Duration::from_millis(2500),
        }
    }
}

impl Timing {
    pub fn zero() -> Self {
        Self {
            start_url_wait: Duration::ZERO,
            activate_chrome_wait: Duration::ZERO,
            before_fill_wait: Duration::ZERO,
            after_fill_wait: Duration::ZERO,
            post_action_wait: Duration::ZERO,
            dropdown_reclick_wait: Duration::ZERO,
            step_wait: Duration::ZERO,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    Done,
    ResultsPageWithPrices,
    CheckoutPageWithPrices,
    StepLimitReached(usize),
    NoCrop,
    SafetyBlocked(String),
    MaxWaitsExceeded,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Capture error: {0}")]
    Capture(#[from] CaptureError),
    #[error("Privacy error: {0}")]
    Privacy(#[from] PrivacyError),
    #[error("Decider error: {0}")]
    Decider(#[from] DeciderError),
    #[error("Safety error: {0}")]
    Safety(#[from] SafetyError),
    #[error("Input error: {0}")]
    Input(#[from] InputError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CropBoxPx {
    pub top: u32,
    pub left: u32,
    pub bottom: u32,
    pub right: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: usize,
    pub url: Option<String>,
    pub kind: String,
    pub kind_confidence: f64,
    pub target: Option<String>,
    pub box_crop_px: Option<CropBoxPx>,
    pub box_confidence: Option<f64>,
    pub reasked: bool,
    pub click_point: Option<(f64, f64)>,
    pub seconds: f64,
    pub results_noul: f64,
    pub done_noul: f64,
    pub sent_image_crop_rect: Rect,
    pub sent_size: (u32, u32),
    pub full_screen_size: (u32, u32),
    pub decision: Decision,
    pub executed: bool,
    #[serde(default)]
    pub click_source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TabEventRecord {
    pub event: String,
    pub step: usize,
    pub url: Option<String>,
    pub restored_to: Option<String>,
    pub tab_events: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewTabEventRecord {
    pub event: String,
    pub step: usize,
    pub url: Option<String>,
    pub tab_count: Option<usize>,
    pub front_window_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OccludedEventRecord {
    pub event: String,
    pub step: usize,
    pub app: String,
    pub point: (f64, f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReclickEventRecord {
    pub event: String,
    pub step: usize,
    pub target: String,
    pub point: (f64, f64),
    pub changed_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockedSensitiveEventRecord {
    pub event: String,
    pub step: usize,
    pub target: String,
    pub point: (f64, f64),
    pub sensitive_noul: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckoutOnceEventRecord {
    pub event: String,
    pub step: usize,
    pub target: String,
    pub point: (f64, f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DismissRefusedEventRecord {
    pub event: String,
    pub step: usize,
    pub target: String,
    pub point: Option<(f64, f64)>,
    pub closeish_noul: f64,
    pub popup_present_noul: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LowBoxConfEventRecord {
    pub event: String,
    pub step: usize,
    pub target: String,
    pub point: Option<(f64, f64)>,
    pub confidence: f64,
    pub box_confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScrollEventRecord {
    pub event: String,
    pub step: usize,
    pub direction: String,
    pub method: String,
    pub scroll_point: (f64, f64),
    pub focus_click_point: Option<(f64, f64)>,
    pub screen_changed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopupEscapeEventRecord {
    pub event: String,
    pub step: usize,
    pub screen_changed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceSeenEventRecord {
    pub event: String,
    pub step: usize,
    pub bbox: BoundingBox,
    pub box_crop_px: CropBoxPx,
    pub confidence: f64,
    pub crop_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OcrEventRecord {
    pub event: String,
    pub step: usize,
    pub lines_total: usize,
    pub candidates: usize,
    pub ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelfFixEventRecord {
    pub event: String,
    pub step: usize,
    pub target: String,
    pub tried: Vec<crate::decider::SelfFixTriedEntry>,
    pub chosen: Option<String>,
    pub base_conf: f64,
    pub budget_used: usize,
}

struct PendingSelfFixClick {
    step: usize,
    preset: String,
    host: String,
    target: String,
    lesson: String,
    chosen: String,
    confidence: f64,
    prev_url: Option<String>,
    prev_img: image::RgbaImage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummaryJson {
    pub status: String,
    pub reason: String,
    pub done_condition: String,
    pub steps: usize,
    pub clicks: usize,
    pub requests_sent: usize,
    pub images_sent: usize,
    pub avg_seconds_per_step: f64,
    pub all_images_cropped: bool,
    pub tab_events: usize,
    #[serde(default)]
    pub price_crops: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunSummary {
    pub steps_taken: usize,
    pub stop_reason: StopReason,
    pub done_condition: String,
    pub history: Vec<String>,
    pub clicks: usize,
    pub requests_sent: usize,
    pub images_sent: usize,
    pub avg_seconds_per_step: f64,
    pub all_images_cropped: bool,
    pub tab_events: usize,
    pub price_crops: Vec<String>,
}

impl AgentRunSummary {
    pub fn to_summary_json(&self) -> RunSummaryJson {
        let (status, reason) = match &self.stop_reason {
            StopReason::Done => (
                "done".to_string(),
                "Task completed ('done' returned by model)".to_string(),
            ),
            StopReason::ResultsPageWithPrices => (
                "done".to_string(),
                "results page with prices".to_string(),
            ),
            StopReason::CheckoutPageWithPrices => (
                "done".to_string(),
                "checkout page with ticket prices".to_string(),
            ),
            StopReason::StepLimitReached(n) => (
                "step_limit".to_string(),
                format!("Reached maximum step limit ({})", n),
            ),
            StopReason::MaxWaitsExceeded => (
                "wait_limit".to_string(),
                "Exceeded maximum consecutive wait actions".to_string(),
            ),
            StopReason::NoCrop => (
                "blocked".to_string(),
                "No active Google Chrome window crop available; nothing sent".to_string(),
            ),
            StopReason::SafetyBlocked(msg) => ("blocked".to_string(), msg.clone()),
        };

        RunSummaryJson {
            status,
            reason,
            done_condition: self.done_condition.clone(),
            steps: self.steps_taken,
            clicks: self.clicks,
            requests_sent: self.requests_sent,
            images_sent: self.images_sent,
            avg_seconds_per_step: self.avg_seconds_per_step,
            all_images_cropped: self.all_images_cropped,
            tab_events: self.tab_events,
            price_crops: self.price_crops.clone(),
        }
    }
}

pub struct Agent<S: Screen, B: DecisionBackend, I: Input, Br: Browser = DefaultAllowedBrowser> {
    screen: S,
    decider: VisionDecider<B>,
    input: I,
    policy: SafetyPolicy,
    browser: Br,
    ocr: Option<Box<dyn Ocr>>,
    start_url: Option<String>,
    flights_preset: bool,
    timing: Timing,
    audit_log: AuditLog,
    logger: Option<RunLogger>,
    dry_run: bool,
}

impl<S: Screen, B: DecisionBackend, I: Input> Agent<S, B, I, DefaultAllowedBrowser> {
    pub fn new(screen: S, mut decider: VisionDecider<B>, input: I, policy: SafetyPolicy) -> Self {
        decider.set_safety_policy(policy.clone());
        Self {
            screen,
            decider,
            input,
            policy,
            browser: DefaultAllowedBrowser::default(),
            ocr: None,
            start_url: None,
            flights_preset: true,
            timing: Timing::zero(),
            audit_log: AuditLog::new(),
            logger: None,
            dry_run: false,
        }
    }
}

impl<S: Screen, B: DecisionBackend, I: Input, Br: Browser> Agent<S, B, I, Br> {
    pub fn with_browser<Br2: Browser>(self, browser: Br2) -> Agent<S, B, I, Br2> {
        Agent {
            screen: self.screen,
            decider: self.decider,
            input: self.input,
            policy: self.policy,
            browser,
            ocr: self.ocr,
            start_url: self.start_url,
            flights_preset: self.flights_preset,
            timing: self.timing,
            audit_log: self.audit_log,
            logger: self.logger,
            dry_run: self.dry_run,
        }
    }

    pub fn with_ocr<O: Ocr + 'static>(mut self, ocr: O) -> Self {
        self.ocr = Some(Box::new(ocr));
        self
    }

    pub fn with_harness(mut self, harness: crate::harness::HarnessOverrides) -> Self {
        self.decider = self.decider.with_harness(harness);
        self
    }

    pub fn with_harness_opt(mut self, harness: Option<crate::harness::HarnessOverrides>) -> Self {
        self.decider = self.decider.with_harness_opt(harness);
        self
    }

    pub fn with_experience_store(mut self, store: crate::experience::ExperienceStore) -> Self {
        self.decider = self.decider.with_experience_store(store);
        self
    }

    pub fn with_experience_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.decider = self.decider.with_experience_dir(dir);
        self
    }

    pub fn with_self_fix(mut self, enabled: bool) -> Self {
        self.decider = self.decider.with_self_fix(enabled);
        self
    }

    pub fn with_flights_preset(mut self, flights_preset: bool) -> Self {
        self.flights_preset = flights_preset;
        self
    }

    pub fn with_done_condition(mut self, done_condition: DoneCondition) -> Self {
        self.decider = self.decider.with_done_condition(done_condition);
        self
    }

    pub fn done_condition(&self) -> DoneCondition {
        self.decider.done_condition()
    }

    pub fn with_start_url(mut self, start_url: Option<String>) -> Self {
        self.start_url = start_url;
        self
    }

    pub fn with_start_url_wait(mut self, wait: Duration) -> Self {
        self.timing.start_url_wait = wait;
        self
    }

    pub fn with_timing(mut self, timing: Timing) -> Self {
        self.timing = timing;
        self
    }

    pub fn timing(&self) -> &Timing {
        &self.timing
    }

    pub fn with_logger(mut self, logger: RunLogger) -> Self {
        self.logger = Some(logger);
        self
    }

    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    pub fn audit_log(&self) -> &AuditLog {
        &self.audit_log
    }

    pub fn policy(&self) -> &SafetyPolicy {
        &self.policy
    }

    pub fn input(&self) -> &I {
        &self.input
    }

    pub fn browser(&self) -> &Br {
        &self.browser
    }

    pub fn decider(&self) -> &VisionDecider<B> {
        &self.decider
    }

    pub fn run(&mut self, goal: &str) -> Result<AgentRunSummary, AgentError> {
        let mut history: Vec<String> = Vec::new();
        let max_steps = self.policy.max_steps;

        let mut clicks = 0usize;
        let mut tab_events = 0usize;
        let mut frontmost_failures = 0usize;
        let mut total_step_secs = 0.0f64;
        let mut steps_completed = 0usize;
        let mut all_images_cropped = true;
        let mut consecutive_results_high = 0usize;
        let mut step_clicks: Vec<Option<String>> = Vec::new();
        let mut last_click_point: Option<(f64, f64)> = None;
        let mut sensitive_excluded: Vec<(String, usize)> = Vec::new();
        let mut consecutive_noop_steps = 0usize;
        let mut checkout_clicked = false;
        let mut price_crops: Vec<String> = Vec::new();
        let mut price_box_asked = false;
        let mut pending_ocr_click: Option<(
            String,
            (f64, f64),
            Option<String>,
            Option<usize>,
            Option<String>,
            image::RgbaImage,
        )> = None;
        let mut pending_self_fix_click: Option<PendingSelfFixClick> = None;

        if self.policy.allowed_hosts.is_empty() {
            let preset_prefixes = if self.flights_preset {
                vec![FLIGHTS_DEFAULT_URL_PREFIX.to_string()]
            } else {
                Vec::new()
            };
            self.policy.allowed_hosts = resolve_allowed_url_prefixes(
                &[],
                &preset_prefixes,
                self.start_url.as_deref(),
            );
        }
        self.decider.set_safety_policy(self.policy.clone());

        if let Some(ref url) = self.start_url {
            let _ = self.browser.activate_chrome();
            let _ = self.browser.set_active_tab_url(url);
            if !self.timing.start_url_wait.is_zero() {
                thread::sleep(self.timing.start_url_wait);
            }
        }

        let initial_tab_count = self.browser.tab_count();
        let initial_window_id = self.browser.front_window_id();

        let mut stop_reason = StopReason::StepLimitReached(max_steps);
        let mut run_err: Option<AgentError> = None;

        for step in 0..max_steps {
            let step_start = Instant::now();

            // Activate Chrome, sleep activate_chrome_wait, and check frontmost app is Google Chrome
            let _ = self.browser.activate_chrome();
            if !self.timing.activate_chrome_wait.is_zero() {
                thread::sleep(self.timing.activate_chrome_wait);
            }
            if !self.browser.is_chrome_frontmost() {
                frontmost_failures += 1;
                if frontmost_failures >= MAX_FRONTMOST_FAILURES {
                    stop_reason = StopReason::SafetyBlocked(
                        "Google Chrome is not the frontmost application".to_string(),
                    );
                    break;
                }
                continue;
            }
            frontmost_failures = 0;

            // Read active tab URL and check if a new tab/window opened before capture
            let current_url = self.browser.active_tab_url();
            let cur_tab_count = self.browser.tab_count();
            let cur_window_id = self.browser.front_window_id();
            let tab_count_increased = match (initial_tab_count, cur_tab_count) {
                (Some(init), Some(cur)) => cur > init,
                _ => false,
            };
            let window_changed = match (&initial_window_id, &cur_window_id) {
                (Some(init), Some(cur)) => cur != init,
                _ => false,
            };
            if tab_count_increased || window_changed {
                if let Some(ref mut logger) = self.logger {
                    let evt = NewTabEventRecord {
                        event: "new_tab".to_string(),
                        step,
                        url: current_url.clone(),
                        tab_count: cur_tab_count,
                        front_window_id: cur_window_id,
                    };
                    let _ = logger.log_entry(&evt);
                }
                stop_reason = StopReason::SafetyBlocked("new tab opened".to_string());
                break;
            }
            if self.flights_preset
                && current_url
                    .as_deref()
                    .map(|u| u.trim().starts_with("https://www.google.com/travel/explore"))
                    .unwrap_or(false)
            {
                tab_events += 1;
                let _ = self.browser.go_back();
                if !self.timing.post_action_wait.is_zero() {
                    thread::sleep(self.timing.post_action_wait);
                }
                if let Some(ref mut logger) = self.logger {
                    let evt = TabEventRecord {
                        event: "explore_back".to_string(),
                        step,
                        url: current_url.clone(),
                        restored_to: Some("back".to_string()),
                        tab_events,
                    };
                    let _ = logger.log_entry(&evt);
                }
                history.push(
                    "search went to Explore page; set destination and date before searching"
                        .to_string(),
                );
                if tab_events > MAX_TAB_EVENTS {
                    stop_reason = StopReason::SafetyBlocked(format!(
                        "Tab URL off allowed prefix more than {} times (last URL: {:?})",
                        MAX_TAB_EVENTS, current_url
                    ));
                    break;
                }
                continue;
            }

            let url_allowed = current_url
                .as_deref()
                .map(|u| self.policy.check_url(u).is_ok())
                .unwrap_or(false);

            if !url_allowed {
                tab_events += 1;
                if let Some(ref start_url) = self.start_url {
                    let _ = self.browser.set_active_tab_url(start_url);
                }
                if let Some(ref mut logger) = self.logger {
                    let evt = TabEventRecord {
                        event: "tab_event".to_string(),
                        step,
                        url: current_url.clone(),
                        restored_to: self.start_url.clone(),
                        tab_events,
                    };
                    let _ = logger.log_entry(&evt);
                }
                if tab_events > MAX_TAB_EVENTS {
                    stop_reason = StopReason::SafetyBlocked(format!(
                        "Tab URL off allowed prefix more than {} times (last URL: {:?})",
                        MAX_TAB_EVENTS, current_url
                    ));
                    break;
                }
                continue;
            }

            if is_login_url(current_url.as_deref()) {
                stop_reason = StopReason::SafetyBlocked("login required".to_string());
                break;
            }

            let bound = match self.screen.capture_window() {
                Ok(b) => b,
                Err(e) => {
                    run_err = Some(AgentError::Capture(e));
                    break;
                }
            };
            let full = bound.image;
            let full_screen_size = (full.width(), full.height());
            let win_rect = bound.rect;
            let scale = self.screen.scale_factor();

            let prepared = match prepare_image(&full, win_rect, scale) {
                Ok(p) => p,
                Err(PrivacyError::NoCrop) => {
                    stop_reason = StopReason::NoCrop;
                    break;
                }
                Err(e) => {
                    run_err = Some(AgentError::Privacy(e));
                    break;
                }
            };

            let win_rect = win_rect.expect("prepare_image succeeded so win_rect is Some");
            let is_this_cropped = prepared.crop_size_px.0 > 0 && prepared.crop_size_px.1 > 0;
            all_images_cropped = all_images_cropped && is_this_cropped;

            self.audit_log
                .record(&prepared, format!("step_{}_decision", step));

            if let Some((prev_text, prev_center, prev_url, prev_tabs, prev_win, prev_img)) =
                pending_ocr_click.take()
            {
                let url_changed = current_url != prev_url;
                let tab_changed = cur_tab_count != prev_tabs || cur_window_id != prev_win;
                let content_changed = is_changed(&prev_img, &full);
                if !url_changed && !tab_changed && !content_changed {
                    self.decider
                        .record_tried_ocr_item(prev_text, prev_center, prev_url);
                }
            }

            if let Some(pending) = pending_self_fix_click.take() {
                self.record_pending_self_fix(pending, &current_url, Some(&full));
            }

            if let Some(ref ocr) = self.ocr {
                let ocr_start = Instant::now();
                let raw_ocr_lines = ocr.recognize(&full).unwrap_or_default();
                let lines_total = raw_ocr_lines.len();
                let front_ok = self
                    .browser
                    .frontmost_app()
                    .as_deref()
                    .map(|a| a.trim() == "Google Chrome")
                    .unwrap_or(false);
                let (_lo, content_top_frac, _ws, _hs) =
                    content_region_offset_and_scale_frac(&prepared);
                let strip_limit_pt = win_rect.y + self.policy.tab_strip_pt;
                let unoccluded_raw: Vec<OcrLine> = raw_ocr_lines
                    .iter()
                    .filter(|l| {
                        l.conf >= crate::ocr::MIN_OCR_CONFIDENCE
                            && !l.text.trim().is_empty()
                            && l.left >= 0.0
                            && l.right <= 1.0
                            && l.top > content_top_frac
                            && l.bottom <= 1.0
                            && l.left < l.right
                            && l.top < l.bottom
                            && l.center().1 > content_top_frac
                            && {
                                let pt = map_box_center_to_screen_pt(&l.bbox(), &prepared);
                                pt.0 >= win_rect.x
                                    && pt.0 <= win_rect.x + win_rect.width
                                    && pt.1 > strip_limit_pt
                                    && pt.1 <= win_rect.y + win_rect.height
                                    && front_ok
                                    && self
                                        .screen
                                        .topmost_app_at(pt.0, pt.1)
                                        .as_deref()
                                        .map(|a| a.trim() == "Google Chrome")
                                        .unwrap_or(false)
                            }
                    })
                    .cloned()
                    .collect();
                let candidates = filter_ocr_candidates(
                    &raw_ocr_lines,
                    &prepared,
                    win_rect,
                    self.policy.tab_strip_pt,
                    |pt| {
                        front_ok
                            && self
                                .screen
                                .topmost_app_at(pt.0, pt.1)
                                .as_deref()
                                .map(|a| a.trim() == "Google Chrome")
                                .unwrap_or(false)
                    },
                );
                let ocr_ms = ocr_start.elapsed().as_millis() as u64;
                if let Some(ref mut logger) = self.logger {
                    let evt = OcrEventRecord {
                        event: "ocr".to_string(),
                        step,
                        lines_total,
                        candidates: candidates.len(),
                        ms: ocr_ms,
                    };
                    let _ = logger.log_entry(&evt);
                }
                if has_login_ocr_lines(&unoccluded_raw) || has_login_ocr_lines(&candidates) {
                    stop_reason = StopReason::SafetyBlocked("login required".to_string());
                    break;
                }
                self.decider.set_ocr_candidates(candidates);
            } else {
                self.decider.set_ocr_candidates(Vec::new());
            }

            let mut excluded_targets = compute_excluded_targets(&step_clicks);
            for (lbl, remaining) in &sensitive_excluded {
                if *remaining > 0 && !excluded_targets.contains(lbl) {
                    excluded_targets.push(lbl.clone());
                }
            }
            for (_, remaining) in &mut sensitive_excluded {
                if *remaining > 0 {
                    *remaining -= 1;
                }
            }
            sensitive_excluded.retain(|(_, r)| *r > 0);

            self.decider.set_current_url(current_url.clone());
            let no_available_targets = !self.decider.has_available_targets(&excluded_targets);

            let mut decision = match self.decider.decide_with_excluded_defer_popup_with_audit(
                goal,
                &history,
                &prepared,
                &excluded_targets,
                Some((&mut self.audit_log, step)),
            ) {
                Ok(d) => d,
                Err(e) => {
                    run_err = Some(AgentError::Decider(e));
                    break;
                }
            };

            if let Some(ref sf) = decision.self_fix {
                if let Some(ref mut logger) = self.logger {
                    let evt = SelfFixEventRecord {
                        event: "self_fix".to_string(),
                        step,
                        target: sf.target.clone(),
                        tried: sf.tried.clone(),
                        chosen: sf.chosen.clone(),
                        base_conf: sf.base_conf,
                        budget_used: sf.budget_used,
                    };
                    let _ = logger.log_entry(&evt);
                }
            }

            if decision.login_form_noul.unwrap_or(0.0) >= LOGIN_FORM_DETECTED_THRESHOLD {
                let elapsed = step_start.elapsed().as_secs_f64();
                total_step_secs += elapsed;
                steps_completed += 1;
                self.log_step(
                    step,
                    current_url,
                    &decision,
                    false,
                    elapsed,
                    win_rect,
                    &prepared,
                    full_screen_size,
                );
                stop_reason = StopReason::SafetyBlocked("login required".to_string());
                break;
            }

            if let Some(ref target_label) = decision.target {
                if let Err(e) = self.policy.check_target(target_label) {
                    let elapsed = step_start.elapsed().as_secs_f64();
                    total_step_secs += elapsed;
                    steps_completed += 1;
                    self.log_step(
                        step,
                        current_url,
                        &decision,
                        false,
                        elapsed,
                        win_rect,
                        &prepared,
                        full_screen_size,
                    );
                    run_err = Some(AgentError::Safety(e));
                    break;
                }
            }

            let done_condition = self.decider.done_condition();
            if done_condition != DoneCondition::Model {
                if decision.results_noul >= RESULTS_DONE_CONSECUTIVE_THRESHOLD {
                    consecutive_results_high += 1;
                } else {
                    consecutive_results_high = 0;
                }

                if consecutive_results_high >= RESULTS_DONE_CONSECUTIVE_COUNT {
                    let elapsed = step_start.elapsed().as_secs_f64();
                    total_step_secs += elapsed;
                    steps_completed += 1;
                    let (reason_str, sr) = match done_condition {
                        DoneCondition::CheckoutPage => (
                            "checkout page with ticket prices",
                            StopReason::CheckoutPageWithPrices,
                        ),
                        _ => ("results page with prices", StopReason::ResultsPageWithPrices),
                    };
                    history.push(reason_str.to_string());
                    self.log_step(
                        step,
                        current_url,
                        &decision,
                        false,
                        elapsed,
                        win_rect,
                        &prepared,
                        full_screen_size,
                    );
                    stop_reason = sr;
                    break;
                }
            }

            let mut executed = false;
            let mut clicked_this_step: Option<String> = None;

            match decision.kind {
                ActionKind::Done => {
                    let model_done_ok = done_condition == DoneCondition::Model
                        || decision.results_noul >= RESULTS_MODEL_DONE_THRESHOLD;
                    if model_done_ok {
                        let elapsed = step_start.elapsed().as_secs_f64();
                        total_step_secs += elapsed;
                        steps_completed += 1;
                        history.push("done".to_string());
                        self.log_step(
                            step,
                            current_url,
                            &decision,
                            false,
                            elapsed,
                            win_rect,
                            &prepared,
                            full_screen_size,
                        );
                        stop_reason = StopReason::Done;
                        break;
                    } else {
                        // Only accept model "done" when results noul >= 0.5; otherwise treat as wait
                        history.push("wait".to_string());
                        let elapsed = step_start.elapsed().as_secs_f64();
                        total_step_secs += elapsed;
                        steps_completed += 1;
                        self.log_step(
                            step,
                            current_url,
                            &decision,
                            false,
                            elapsed,
                            win_rect,
                            &prepared,
                            full_screen_size,
                        );
                        if no_available_targets {
                            consecutive_noop_steps += 1;
                            let _ = consecutive_noop_steps;
                            stop_reason =
                                StopReason::SafetyBlocked("no available targets".to_string());
                            break;
                        }
                        consecutive_noop_steps = 0;
                        if self.policy.on_wait().is_err() {
                            stop_reason = StopReason::MaxWaitsExceeded;
                            break;
                        }
                    }
                }
                ActionKind::Wait => {
                    history.push("wait".to_string());
                    let elapsed = step_start.elapsed().as_secs_f64();
                    total_step_secs += elapsed;
                    steps_completed += 1;
                    self.log_step(
                        step,
                        current_url,
                        &decision,
                        !no_available_targets,
                        elapsed,
                        win_rect,
                        &prepared,
                        full_screen_size,
                    );
                    if no_available_targets {
                        consecutive_noop_steps += 1;
                        let _ = consecutive_noop_steps;
                        stop_reason = StopReason::SafetyBlocked("no available targets".to_string());
                        break;
                    }
                    consecutive_noop_steps = 0;
                    if self.policy.on_wait().is_err() {
                        stop_reason = StopReason::MaxWaitsExceeded;
                        break;
                    }
                }
                ActionKind::DismissPopup => {
                    let default_popup_close = self.decider.popup_close_label();
                    let tgt = decision
                        .target
                        .clone()
                        .unwrap_or_else(|| default_popup_close.to_string());

                    let mut escaped_changed = false;
                    if !self.dry_run {
                        self.input.key(KeyCode::Escape)?;
                        self.input.release_modifiers()?;
                        if !self.timing.post_action_wait.is_zero() {
                            thread::sleep(self.timing.post_action_wait);
                        }
                        if let Ok(after_esc) = self.screen.capture() {
                            escaped_changed = is_changed(&full, &after_esc);
                        }
                    }
                    if let Some(ref mut logger) = self.logger {
                        let evt = PopupEscapeEventRecord {
                            event: "popup_escape".to_string(),
                            step,
                            screen_changed: escaped_changed,
                        };
                        let _ = logger.log_entry(&evt);
                    }

                    if escaped_changed {
                        self.decider.record_closed_popup("escape", None);
                        consecutive_noop_steps = 0;
                        self.policy.reset_wait();
                        executed = true;
                        history.push("dismiss popup via Escape".to_string());
                        let elapsed = step_start.elapsed().as_secs_f64();
                        total_step_secs += elapsed;
                        steps_completed += 1;
                        self.log_step(
                            step,
                            current_url,
                            &decision,
                            executed,
                            elapsed,
                            win_rect,
                            &prepared,
                            full_screen_size,
                        );
                    } else {
                        let state = format_state(goal, &history);
                        if let Err(e) = self.decider.resolve_dismiss_popup(
                            &state,
                            &prepared,
                            &mut decision,
                            Some((&mut self.audit_log, step)),
                        ) {
                            run_err = Some(AgentError::Decider(e));
                            break;
                        }
                        let box_conf = decision.box_confidence.unwrap_or(0.0);
                        let closeish = decision.closeish_noul.unwrap_or(0.0);
                        let popup_present = decision.popup_present_noul.unwrap_or(0.0);
                        if box_conf < LOW_CONFIDENCE_THRESHOLD {
                            if let Some(ref mut logger) = self.logger {
                                let evt = LowBoxConfEventRecord {
                                    event: "low_box_conf".to_string(),
                                    step,
                                    target: tgt.clone(),
                                    point: decision.click_point_screen_pt,
                                    confidence: box_conf,
                                    box_confidence: box_conf,
                                };
                                let _ = logger.log_entry(&evt);
                            }
                            let excl_steps = low_box_conf_exclude_steps(&tgt);
                            if let Some(entry) =
                                sensitive_excluded.iter_mut().find(|(l, _)| l == &tgt)
                            {
                                entry.1 = entry.1.max(excl_steps);
                            } else {
                                sensitive_excluded.push((tgt.clone(), excl_steps));
                            }
                            clicked_this_step = Some(format!("low_box_conf:{}", tgt));
                            history.push("wait".to_string());
                            let elapsed = step_start.elapsed().as_secs_f64();
                            total_step_secs += elapsed;
                            steps_completed += 1;
                            self.log_step(
                                step,
                                current_url,
                                &decision,
                                false,
                                elapsed,
                                win_rect,
                                &prepared,
                                full_screen_size,
                            );
                            consecutive_noop_steps += 1;
                            if consecutive_noop_steps >= MAX_CONSECUTIVE_NOOP_STEPS {
                                stop_reason = StopReason::SafetyBlocked(
                                    "3 consecutive no-op steps (low box confidence)".to_string(),
                                );
                                break;
                            }
                        } else if popup_present < POPUP_PRESENT_THRESHOLD
                            || closeish < DISMISS_CLOSEISH_THRESHOLD
                        {
                            if let Some(ref mut logger) = self.logger {
                                let evt = DismissRefusedEventRecord {
                                    event: "dismiss_refused".to_string(),
                                    step,
                                    target: tgt.clone(),
                                    point: decision.click_point_screen_pt,
                                    closeish_noul: closeish,
                                    popup_present_noul: decision.popup_present_noul,
                                };
                                let _ = logger.log_entry(&evt);
                            }
                            history.push("wait".to_string());
                            let elapsed = step_start.elapsed().as_secs_f64();
                            total_step_secs += elapsed;
                            steps_completed += 1;
                            self.log_step(
                                step,
                                current_url,
                                &decision,
                                false,
                                elapsed,
                                win_rect,
                                &prepared,
                                full_screen_size,
                            );
                            if self.policy.on_wait().is_err() {
                                stop_reason = StopReason::MaxWaitsExceeded;
                                break;
                            }
                        } else {
                            self.policy.reset_wait();
                            if let Some(pt) = decision.click_point_screen_pt {
                                if let Err(e) = self.policy.check_click(pt, win_rect) {
                                    let elapsed = step_start.elapsed().as_secs_f64();
                                    total_step_secs += elapsed;
                                    steps_completed += 1;
                                    self.log_step(
                                        step,
                                        current_url,
                                        &decision,
                                        false,
                                        elapsed,
                                        win_rect,
                                        &prepared,
                                        full_screen_size,
                                    );
                                    run_err = Some(AgentError::Safety(e));
                                    break;
                                }
                                if let Err(reason) = self.check_unoccluded_at(step, pt) {
                                    let elapsed = step_start.elapsed().as_secs_f64();
                                    total_step_secs += elapsed;
                                    steps_completed += 1;
                                    self.log_step(
                                        step,
                                        current_url,
                                        &decision,
                                        false,
                                        elapsed,
                                        win_rect,
                                        &prepared,
                                        full_screen_size,
                                    );
                                    stop_reason = StopReason::SafetyBlocked(reason);
                                    break;
                                }
                                let mut click_changed = true;
                                if !self.dry_run {
                                    self.input.click(pt.0, pt.1)?;
                                    executed = true;
                                    if !self.timing.post_action_wait.is_zero() {
                                        thread::sleep(self.timing.post_action_wait);
                                    }
                                    if let Ok(after_click) = self.screen.capture() {
                                        click_changed = is_changed(&full, &after_click);
                                    }
                                }
                                if click_changed {
                                    consecutive_noop_steps = 0;
                                } else {
                                    consecutive_noop_steps += 1;
                                }
                                clicks += 1;
                                clicked_this_step = Some(tgt.clone());
                                last_click_point = Some(pt);
                                history.push(format!(
                                    "click '{}' at ({:.1}, {:.1})",
                                    tgt, pt.0, pt.1
                                ));
                            }
                            if executed && !self.timing.post_action_wait.is_zero() {
                                thread::sleep(self.timing.post_action_wait);
                            }
                            let elapsed = step_start.elapsed().as_secs_f64();
                            total_step_secs += elapsed;
                            steps_completed += 1;
                            self.log_step(
                                step,
                                current_url,
                                &decision,
                                executed,
                                elapsed,
                                win_rect,
                                &prepared,
                                full_screen_size,
                            );
                            if consecutive_noop_steps >= MAX_CONSECUTIVE_NOOP_STEPS {
                                stop_reason = StopReason::SafetyBlocked(
                                    "3 consecutive no-op steps (low box confidence)".to_string(),
                                );
                                break;
                            }
                        }
                    }
                }
                ActionKind::Click => {
                    let box_conf = decision.box_confidence.unwrap_or(0.0);
                    let tgt = decision.target.as_deref().unwrap_or("unknown");
                    if box_conf < LOW_CONFIDENCE_THRESHOLD {
                        if let Some(ref mut logger) = self.logger {
                            let evt = LowBoxConfEventRecord {
                                event: "low_box_conf".to_string(),
                                step,
                                target: tgt.to_string(),
                                point: decision.click_point_screen_pt,
                                confidence: box_conf,
                                box_confidence: box_conf,
                            };
                            let _ = logger.log_entry(&evt);
                        }
                        let excl_steps = low_box_conf_exclude_steps(tgt);
                        if let Some(entry) =
                            sensitive_excluded.iter_mut().find(|(l, _)| l == tgt)
                        {
                            entry.1 = entry.1.max(excl_steps);
                        } else {
                            sensitive_excluded.push((tgt.to_string(), excl_steps));
                        }
                        history.push("wait".to_string());
                        let elapsed = step_start.elapsed().as_secs_f64();
                        total_step_secs += elapsed;
                        steps_completed += 1;
                        self.log_step(
                            step,
                            current_url,
                            &decision,
                            false,
                            elapsed,
                            win_rect,
                            &prepared,
                            full_screen_size,
                        );
                        consecutive_noop_steps += 1;
                        if consecutive_noop_steps >= MAX_CONSECUTIVE_NOOP_STEPS {
                            stop_reason = StopReason::SafetyBlocked(
                                "3 consecutive no-op steps (low box confidence)".to_string(),
                            );
                            break;
                        }
                        step_clicks.push(Some(format!("low_box_conf:{}", tgt)));
                        if !self.timing.step_wait.is_zero() {
                            thread::sleep(self.timing.step_wait);
                        }
                        continue;
                    }
                    consecutive_noop_steps = 0;
                    self.policy.reset_wait();
                    if let Some(pt) = decision.click_point_screen_pt {
                        if let Err(e) = self.policy.check_click(pt, win_rect) {
                            let elapsed = step_start.elapsed().as_secs_f64();
                            total_step_secs += elapsed;
                            steps_completed += 1;
                            self.log_step(
                                step,
                                current_url,
                                &decision,
                                false,
                                elapsed,
                                win_rect,
                                &prepared,
                                full_screen_size,
                            );
                            run_err = Some(AgentError::Safety(e));
                            break;
                        }
                        if let Err(reason) = self.check_unoccluded_at(step, pt) {
                            let elapsed = step_start.elapsed().as_secs_f64();
                            total_step_secs += elapsed;
                            steps_completed += 1;
                            self.log_step(
                                step,
                                current_url,
                                &decision,
                                false,
                                elapsed,
                                win_rect,
                                &prepared,
                                full_screen_size,
                            );
                            stop_reason = StopReason::SafetyBlocked(reason);
                            break;
                        }

                        if let Some(sens) = decision.sensitive_noul {
                            if sens >= SENSITIVE_BLOCK_THRESHOLD {
                                if let Some(ref mut logger) = self.logger {
                                    let evt = BlockedSensitiveEventRecord {
                                        event: "blocked_sensitive".to_string(),
                                        step,
                                        target: tgt.to_string(),
                                        point: pt,
                                        sensitive_noul: sens,
                                    };
                                    let _ = logger.log_entry(&evt);
                                }
                                if let Some(entry) =
                                    sensitive_excluded.iter_mut().find(|(l, _)| l == tgt)
                                {
                                    entry.1 = SENSITIVE_EXCLUDE_STEPS;
                                } else {
                                    sensitive_excluded
                                        .push((tgt.to_string(), SENSITIVE_EXCLUDE_STEPS));
                                }
                                history.push(format!("blocked_sensitive '{}'", tgt));
                                let elapsed = step_start.elapsed().as_secs_f64();
                                total_step_secs += elapsed;
                                steps_completed += 1;
                                self.log_step(
                                    step,
                                    current_url,
                                    &decision,
                                    false,
                                    elapsed,
                                    win_rect,
                                    &prepared,
                                    full_screen_size,
                                );
                                step_clicks.push(None);
                                if !self.timing.step_wait.is_zero() {
                                    thread::sleep(self.timing.step_wait);
                                }
                                continue;
                            }
                        }

                        if self.decider.is_checkout_once_target(tgt) {
                            if checkout_clicked {
                                if let Some(ref mut logger) = self.logger {
                                    let evt = CheckoutOnceEventRecord {
                                        event: "checkout_once".to_string(),
                                        step,
                                        target: tgt.to_string(),
                                        point: pt,
                                    };
                                    let _ = logger.log_entry(&evt);
                                }
                                history.push(format!("checkout_once '{}'", tgt));
                                let elapsed = step_start.elapsed().as_secs_f64();
                                total_step_secs += elapsed;
                                steps_completed += 1;
                                self.log_step(
                                    step,
                                    current_url,
                                    &decision,
                                    false,
                                    elapsed,
                                    win_rect,
                                    &prepared,
                                    full_screen_size,
                                );
                                step_clicks.push(None);
                                if !self.timing.step_wait.is_zero() {
                                    thread::sleep(self.timing.step_wait);
                                }
                                continue;
                            }
                            checkout_clicked = true;
                        }

                        let fill_val = self.decider.fill_for(tgt).map(|s| s.to_string());

                        if let Some(ref val) = fill_val {
                            if is_sensitive_fill_label(tgt) || is_sensitive_text(val) {
                                let elapsed = step_start.elapsed().as_secs_f64();
                                total_step_secs += elapsed;
                                steps_completed += 1;
                                self.log_step(
                                    step,
                                    current_url,
                                    &decision,
                                    false,
                                    elapsed,
                                    win_rect,
                                    &prepared,
                                    full_screen_size,
                                );
                                run_err = Some(AgentError::Safety(SafetyError::SensitiveTextBlocked));
                                break;
                            }
                        }

                        if tgt == "立即報名按鈕"
                            && self.decider.is_on_event_page()
                            && !price_box_asked
                        {
                            price_box_asked = true;
                            let state = format_state(goal, &history);
                            match self.decider.ask_price_box(&state, &prepared) {
                                Ok((price_bbox, price_conf)) => {
                                    if price_conf >= LOW_CONFIDENCE_THRESHOLD {
                                        let (cw, ch) = prepared.crop_size_px;
                                        let (ox, oy) = prepared.crop_origin_px;
                                        let top_px = (price_bbox.top * (ch as f64))
                                            .round()
                                            .clamp(0.0, ch as f64)
                                            as u32;
                                        let left_px = (price_bbox.left * (cw as f64))
                                            .round()
                                            .clamp(0.0, cw as f64)
                                            as u32;
                                        let bottom_px = (price_bbox.bottom * (ch as f64))
                                            .round()
                                            .clamp(0.0, ch as f64)
                                            as u32;
                                        let right_px = (price_bbox.right * (cw as f64))
                                            .round()
                                            .clamp(0.0, cw as f64)
                                            as u32;
                                        let crop_x =
                                            (ox + left_px).min(full.width().saturating_sub(1));
                                        let crop_y =
                                            (oy + top_px).min(full.height().saturating_sub(1));
                                        let crop_w = right_px
                                            .saturating_sub(left_px)
                                            .max(1)
                                            .min(full.width().saturating_sub(crop_x))
                                            .max(1);
                                        let crop_h = bottom_px
                                            .saturating_sub(top_px)
                                            .max(1)
                                            .min(full.height().saturating_sub(crop_y))
                                            .max(1);
                                        let price_sub =
                                            full.view(crop_x, crop_y, crop_w, crop_h).to_image();
                                        let crop_filename = format!("price_step_{}.jpg", step);
                                        let crop_path = if let Some(ref logger) = self.logger {
                                            let p = logger.run_dir().join(&crop_filename);
                                            if let Ok(jpeg_bytes) =
                                                encode_jpeg(&price_sub, JPEG_QUALITY)
                                            {
                                                let _ = std::fs::write(&p, jpeg_bytes);
                                            }
                                            p.to_string_lossy().to_string()
                                        } else {
                                            crop_filename
                                        };
                                        price_crops.push(crop_path.clone());
                                        if let Some(ref mut logger) = self.logger {
                                            let evt = PriceSeenEventRecord {
                                                event: "price_seen".to_string(),
                                                step,
                                                bbox: price_bbox,
                                                box_crop_px: CropBoxPx {
                                                    top: top_px,
                                                    left: left_px,
                                                    bottom: bottom_px,
                                                    right: right_px,
                                                },
                                                confidence: price_conf,
                                                crop_path,
                                            };
                                            let _ = logger.log_entry(&evt);
                                        }
                                    }
                                }
                                Err(e) => {
                                    run_err = Some(AgentError::Decider(e));
                                    break;
                                }
                            }
                        }

                        if !self.dry_run {
                            self.input.click(pt.0, pt.1)?;
                            executed = true;
                        }
                        clicks += 1;
                        clicked_this_step = Some(tgt.to_string());
                        last_click_point = Some(pt);

                        if !self.dry_run {
                            if let Some(ref sf) = decision.self_fix {
                                if let Some(ref chosen) = sf.chosen {
                                    let lesson = sf
                                        .chosen_lesson
                                        .clone()
                                        .filter(|s| !s.trim().is_empty())
                                        .or_else(|| {
                                            self.decider
                                                .targets()
                                                .iter()
                                                .find(|t| t.label == tgt)
                                                .and_then(|t| t.description.clone())
                                        })
                                        .unwrap_or_else(|| format!("Locate {}", tgt));
                                    let host = crate::experience::extract_host_from_url(
                                        current_url.as_deref().or(self.start_url.as_deref()),
                                    );
                                    pending_self_fix_click = Some(PendingSelfFixClick {
                                        step,
                                        preset: self.decider.preset_str().to_string(),
                                        host,
                                        target: tgt.to_string(),
                                        lesson,
                                        chosen: chosen.clone(),
                                        confidence: box_conf,
                                        prev_url: current_url.clone(),
                                        prev_img: full.clone(),
                                    });
                                }
                            }
                        }

                        if let Some(val) = fill_val {
                            if !self.dry_run {
                                if !self.timing.before_fill_wait.is_zero() {
                                    thread::sleep(self.timing.before_fill_wait);
                                }
                            }
                            if let Err(reason) = self.check_unoccluded_at(step, pt) {
                                stop_reason = StopReason::SafetyBlocked(reason);
                                break;
                            }
                            if !self.dry_run {
                                self.input.select_all()?;
                                self.input.type_text(&val)?;
                                self.input.release_modifiers()?;
                                if !self.timing.after_fill_wait.is_zero() {
                                    thread::sleep(self.timing.after_fill_wait);
                                }
                            }
                            history.push(format!("clicked and typed '{}' into {}", val, tgt));
                        } else {
                            if !self.dry_run && tgt == "trip type dropdown" {
                                if !self.timing.dropdown_reclick_wait.is_zero() {
                                    thread::sleep(self.timing.dropdown_reclick_wait);
                                }
                                let after = self.screen.capture()?;
                                let frac = changed_fraction(&full, &after);
                                if !is_changed(&full, &after) {
                                    if let Err(reason) = self.check_unoccluded_at(step, pt) {
                                        stop_reason = StopReason::SafetyBlocked(reason);
                                        break;
                                    }
                                    self.input.click(pt.0, pt.1)?;
                                    clicks += 1;
                                    if let Some(ref mut logger) = self.logger {
                                        let evt = ReclickEventRecord {
                                            event: "reclick_unchanged".to_string(),
                                            step,
                                            target: tgt.to_string(),
                                            point: pt,
                                            changed_fraction: frac,
                                        };
                                        let _ = logger.log_entry(&evt);
                                    }
                                }
                            }
                            if executed && !self.timing.post_action_wait.is_zero() {
                                thread::sleep(self.timing.post_action_wait);
                            }
                            if !self.dry_run && decision.click_source.as_deref() == Some("ocr") {
                                if let (Some(ref item_text), Some(center_frac)) =
                                    (&decision.ocr_item_text, decision.ocr_item_center_frac)
                                {
                                    pending_ocr_click = Some((
                                        item_text.clone(),
                                        center_frac,
                                        current_url.clone(),
                                        cur_tab_count,
                                        cur_window_id.clone(),
                                        full.clone(),
                                    ));
                                }
                            } else if !self.dry_run
                                && tgt == "活動卡片"
                                && self.decider.harness().is_none()
                            {
                                let post_url = self.browser.active_tab_url();
                                let post_tabs = self.browser.tab_count();
                                let post_win = self.browser.front_window_id();
                                let new_tab_or_win = match (initial_tab_count, post_tabs) {
                                    (Some(init), Some(cur)) => cur > init,
                                    _ => false,
                                } || match (&initial_window_id, &post_win) {
                                    (Some(init), Some(cur)) => cur != init,
                                    _ => false,
                                };
                                let url_changed = post_url != current_url;
                                if !url_changed && !new_tab_or_win {
                                    let state = format_state(goal, &history);
                                    let retry_res = self
                                        .decider
                                        .resolve_card_link_retry_box(&state, &prepared);
                                    let retry_opt = match retry_res {
                                        Ok(opt) => opt,
                                        Err(e) => {
                                            run_err = Some(AgentError::Decider(e));
                                            break;
                                        }
                                    };
                                    let mut navigated = false;
                                    if let Some((retry_bbox, _retry_conf, _retry_sens)) = retry_opt
                                    {
                                        let retry_pt =
                                            map_box_center_to_screen_pt(&retry_bbox, &prepared);
                                        if let Err(e) =
                                            self.policy.check_click(retry_pt, win_rect)
                                        {
                                            let elapsed = step_start.elapsed().as_secs_f64();
                                            total_step_secs += elapsed;
                                            steps_completed += 1;
                                            self.log_step(
                                                step,
                                                current_url,
                                                &decision,
                                                executed,
                                                elapsed,
                                                win_rect,
                                                &prepared,
                                                full_screen_size,
                                            );
                                            run_err = Some(AgentError::Safety(e));
                                            break;
                                        }
                                        if let Err(reason) =
                                            self.check_unoccluded_at(step, retry_pt)
                                        {
                                            let elapsed = step_start.elapsed().as_secs_f64();
                                            total_step_secs += elapsed;
                                            steps_completed += 1;
                                            self.log_step(
                                                step,
                                                current_url,
                                                &decision,
                                                executed,
                                                elapsed,
                                                win_rect,
                                                &prepared,
                                                full_screen_size,
                                            );
                                            stop_reason = StopReason::SafetyBlocked(reason);
                                            break;
                                        }
                                        self.input.click(retry_pt.0, retry_pt.1)?;
                                        clicks += 1;
                                        last_click_point = Some(retry_pt);
                                        if !self.timing.post_action_wait.is_zero() {
                                            thread::sleep(self.timing.post_action_wait);
                                        }
                                        let post_retry_url = self.browser.active_tab_url();
                                        let post_retry_tabs = self.browser.tab_count();
                                        let post_retry_win = self.browser.front_window_id();
                                        let retry_new_tab =
                                            match (initial_tab_count, post_retry_tabs) {
                                                (Some(init), Some(cur)) => cur > init,
                                                _ => false,
                                            } || match (&initial_window_id, &post_retry_win) {
                                                (Some(init), Some(cur)) => cur != init,
                                                _ => false,
                                            };
                                        navigated =
                                            post_retry_url != current_url || retry_new_tab;
                                    }
                                    if !navigated {
                                        history.push(format!(
                                            "click '{}' at ({:.1}, {:.1})",
                                            tgt, pt.0, pt.1
                                        ));
                                        let elapsed = step_start.elapsed().as_secs_f64();
                                        total_step_secs += elapsed;
                                        steps_completed += 1;
                                        self.log_step(
                                            step,
                                            current_url,
                                            &decision,
                                            executed,
                                            elapsed,
                                            win_rect,
                                            &prepared,
                                            full_screen_size,
                                        );
                                        stop_reason = StopReason::SafetyBlocked(
                                            "card click did not navigate".to_string(),
                                        );
                                        break;
                                    }
                                }
                            }
                            history.push(format!("click '{}' at ({:.1}, {:.1})", tgt, pt.0, pt.1));
                        }
                    }
                    if executed && !self.timing.post_action_wait.is_zero() {
                        thread::sleep(self.timing.post_action_wait);
                    }
                    let elapsed = step_start.elapsed().as_secs_f64();
                    total_step_secs += elapsed;
                    steps_completed += 1;
                    self.log_step(
                        step,
                        current_url,
                        &decision,
                        executed,
                        elapsed,
                        win_rect,
                        &prepared,
                        full_screen_size,
                    );
                }
                ActionKind::Type => {
                    consecutive_noop_steps = 0;
                    self.policy.reset_wait();
                    let text = decision.text_to_type.as_deref().unwrap_or("");
                    if let Some(ref tgt) = decision.target {
                        self.policy.check_target(tgt)?;
                    }
                    if is_sensitive_text(text) {
                        return Err(AgentError::Safety(SafetyError::SensitiveTextBlocked));
                    }
                    let check_pt = last_click_point.unwrap_or((
                        win_rect.x + win_rect.width / 2.0,
                        win_rect.y + win_rect.height / 2.0,
                    ));
                    if let Err(reason) = self.check_unoccluded_at(step, check_pt) {
                        stop_reason = StopReason::SafetyBlocked(reason);
                        break;
                    }
                    if !self.dry_run && !text.is_empty() {
                        self.input.type_text(text)?;
                        self.input.release_modifiers()?;
                        executed = true;
                    }
                    let tgt = decision.target.as_deref().unwrap_or("field");
                    history.push(format!("type into '{}'", tgt));
                    let elapsed = step_start.elapsed().as_secs_f64();
                    total_step_secs += elapsed;
                    steps_completed += 1;
                    self.log_step(
                        step,
                        current_url,
                        &decision,
                        executed,
                        elapsed,
                        win_rect,
                        &prepared,
                        full_screen_size,
                    );
                }
                ActionKind::PressEnter => {
                    consecutive_noop_steps = 0;
                    self.policy.reset_wait();
                    let check_pt = last_click_point.unwrap_or((
                        win_rect.x + win_rect.width / 2.0,
                        win_rect.y + win_rect.height / 2.0,
                    ));
                    if let Err(reason) = self.check_unoccluded_at(step, check_pt) {
                        stop_reason = StopReason::SafetyBlocked(reason);
                        break;
                    }
                    if !self.dry_run {
                        self.input.key(KeyCode::Return)?;
                        self.input.release_modifiers()?;
                        executed = true;
                        if !self.timing.post_action_wait.is_zero() {
                            thread::sleep(self.timing.post_action_wait);
                        }
                    }
                    history.push("press enter".to_string());
                    let elapsed = step_start.elapsed().as_secs_f64();
                    total_step_secs += elapsed;
                    steps_completed += 1;
                    self.log_step(
                        step,
                        current_url,
                        &decision,
                        executed,
                        elapsed,
                        win_rect,
                        &prepared,
                        full_screen_size,
                    );
                }
                ActionKind::Scroll => {
                    self.policy.reset_wait();
                    let content_top = win_rect.y + self.policy.tab_strip_pt;
                    let content_h = (win_rect.height - self.policy.tab_strip_pt).max(1.0);
                    let scroll_pt = (
                        win_rect.x + win_rect.width / 2.0,
                        content_top + content_h / 2.0,
                    );
                    if let Err(e) = self.policy.check_click(scroll_pt, win_rect) {
                        let elapsed = step_start.elapsed().as_secs_f64();
                        total_step_secs += elapsed;
                        steps_completed += 1;
                        self.log_step(
                            step,
                            current_url,
                            &decision,
                            false,
                            elapsed,
                            win_rect,
                            &prepared,
                            full_screen_size,
                        );
                        run_err = Some(AgentError::Safety(e));
                        break;
                    }
                    if let Err(reason) = self.check_unoccluded_at(step, scroll_pt) {
                        let elapsed = step_start.elapsed().as_secs_f64();
                        total_step_secs += elapsed;
                        steps_completed += 1;
                        self.log_step(
                            step,
                            current_url,
                            &decision,
                            false,
                            elapsed,
                            win_rect,
                            &prepared,
                            full_screen_size,
                        );
                        stop_reason = StopReason::SafetyBlocked(reason);
                        break;
                    }

                    let dir = self.decider.scroll_direction();
                    let delta_px = match dir {
                        ScrollDirection::Down => DEFAULT_SCROLL_DELTA_PX,
                        ScrollDirection::Up => DEFAULT_SCROLL_UP_DELTA_PX,
                    };
                    let (fallback_key, fallback_method) = match dir {
                        ScrollDirection::Down => (KeyCode::PageDown, "pagedown_fallback"),
                        ScrollDirection::Up => (KeyCode::PageUp, "pageup_fallback"),
                    };

                    let mut method = "wheel".to_string();
                    let mut focus_click_pt: Option<(f64, f64)> = None;
                    let mut screen_changed = false;

                    if !self.dry_run {
                        self.input
                            .scroll_vertical_at(scroll_pt.0, scroll_pt.1, delta_px)?;
                        executed = true;
                        if !self.timing.post_action_wait.is_zero() {
                            thread::sleep(self.timing.post_action_wait);
                        }
                        let after_wheel = self.screen.capture()?;
                        screen_changed = is_changed(&full, &after_wheel);
                        if !screen_changed {
                            // Fall back: safe click on blank space in the page content margin, then Page Down / Page Up
                            let blank_pt = (
                                win_rect.x + 12.0,
                                content_top + content_h / 2.0,
                            );
                            if let Err(e) = self.policy.check_click(blank_pt, win_rect) {
                                let elapsed = step_start.elapsed().as_secs_f64();
                                total_step_secs += elapsed;
                                steps_completed += 1;
                                self.log_step(
                                    step,
                                    current_url,
                                    &decision,
                                    false,
                                    elapsed,
                                    win_rect,
                                    &prepared,
                                    full_screen_size,
                                );
                                run_err = Some(AgentError::Safety(e));
                                break;
                            }
                            if let Err(reason) = self.check_unoccluded_at(step, blank_pt) {
                                let elapsed = step_start.elapsed().as_secs_f64();
                                total_step_secs += elapsed;
                                steps_completed += 1;
                                self.log_step(
                                    step,
                                    current_url,
                                    &decision,
                                    false,
                                    elapsed,
                                    win_rect,
                                    &prepared,
                                    full_screen_size,
                                );
                                stop_reason = StopReason::SafetyBlocked(reason);
                                break;
                            }
                            self.input.focus_click(blank_pt.0, blank_pt.1)?;
                            self.input.key(fallback_key)?;
                            self.input.release_modifiers()?;
                            if !self.timing.post_action_wait.is_zero() {
                                thread::sleep(self.timing.post_action_wait);
                            }
                            let after_pg = self.screen.capture()?;
                            screen_changed = is_changed(&full, &after_pg);
                            method = fallback_method.to_string();
                            focus_click_pt = Some(blank_pt);
                        }
                    }

                    if dir == ScrollDirection::Down
                        && (!screen_changed
                            || decision.footer_noul.unwrap_or(0.0) >= FOOTER_DETECTED_THRESHOLD)
                    {
                        self.decider.set_scroll_direction(ScrollDirection::Up);
                    }

                    if let Some(ref mut logger) = self.logger {
                        let evt = ScrollEventRecord {
                            event: "scroll".to_string(),
                            step,
                            direction: dir.as_str().to_string(),
                            method,
                            scroll_point: scroll_pt,
                            focus_click_point: focus_click_pt,
                            screen_changed,
                        };
                        let _ = logger.log_entry(&evt);
                    }

                    let hist_str = match dir {
                        ScrollDirection::Down => "scroll down",
                        ScrollDirection::Up => "scroll up",
                    };
                    history.push(hist_str.to_string());
                    let elapsed = step_start.elapsed().as_secs_f64();
                    total_step_secs += elapsed;
                    steps_completed += 1;
                    self.log_step(
                        step,
                        current_url,
                        &decision,
                        executed,
                        elapsed,
                        win_rect,
                        &prepared,
                        full_screen_size,
                    );

                    if screen_changed {
                        consecutive_noop_steps = 0;
                    } else {
                        consecutive_noop_steps += 1;
                        if consecutive_noop_steps >= MAX_CONSECUTIVE_NOOP_STEPS {
                            stop_reason = StopReason::SafetyBlocked(
                                "3 consecutive no-op steps (low box confidence)".to_string(),
                            );
                            break;
                        }
                    }
                }
            }
            step_clicks.push(clicked_this_step);
            if !self.timing.step_wait.is_zero() {
                thread::sleep(self.timing.step_wait);
            }
        }

        if let Some(pending) = pending_self_fix_click.take() {
            let post_url = self
                .browser
                .active_tab_url()
                .or_else(|| pending.prev_url.clone());
            let post_img = self.screen.capture().ok();
            self.record_pending_self_fix(pending, &post_url, post_img.as_ref());
        }

        let avg_seconds_per_step = if steps_completed > 0 {
            total_step_secs / (steps_completed as f64)
        } else {
            0.0
        };

        if let Some(err) = run_err {
            if let Some(ref logger) = self.logger {
                let (status, reason) = match &err {
                    AgentError::Safety(s) => ("blocked".to_string(), s.to_string()),
                    AgentError::Privacy(p) => ("blocked".to_string(), p.to_string()),
                    other => ("error".to_string(), other.to_string()),
                };
                let summary_json = RunSummaryJson {
                    status,
                    reason,
                    done_condition: self.decider.done_condition().as_str().to_string(),
                    steps: steps_completed,
                    clicks,
                    requests_sent: self.decider.requests_sent(),
                    images_sent: self.decider.images_sent(),
                    avg_seconds_per_step,
                    all_images_cropped,
                    tab_events,
                    price_crops: price_crops.clone(),
                };
                let _ = logger.write_summary(&summary_json);
            }
            return Err(err);
        }

        let summary = AgentRunSummary {
            steps_taken: steps_completed,
            stop_reason,
            done_condition: self.decider.done_condition().as_str().to_string(),
            history,
            clicks,
            requests_sent: self.decider.requests_sent(),
            images_sent: self.decider.images_sent(),
            avg_seconds_per_step,
            all_images_cropped,
            tab_events,
            price_crops,
        };

        if let Some(ref logger) = self.logger {
            let summary_json = summary.to_summary_json();
            let _ = logger.write_summary(&summary_json);
        }

        Ok(summary)
    }

    fn check_unoccluded_at(&mut self, step: usize, pt: (f64, f64)) -> Result<(), String> {
        let topmost = self.screen.topmost_app_at(pt.0, pt.1);
        let frontmost = self.browser.frontmost_app();
        let top_ok = topmost
            .as_deref()
            .map(|a| a.trim() == "Google Chrome")
            .unwrap_or(false);
        let front_ok = frontmost
            .as_deref()
            .map(|a| a.trim() == "Google Chrome")
            .unwrap_or(false);
        if top_ok && front_ok {
            return Ok(());
        }
        let app = if !top_ok {
            topmost.unwrap_or_else(|| frontmost.unwrap_or_else(|| "unknown".to_string()))
        } else {
            frontmost.unwrap_or_else(|| "unknown".to_string())
        };
        if let Some(ref mut logger) = self.logger {
            let evt = OccludedEventRecord {
                event: "occluded".to_string(),
                step,
                app: app.clone(),
                point: pt,
            };
            let _ = logger.log_entry(&evt);
        }
        Err(format!("Chrome occluded by {}", app))
    }

    #[allow(clippy::too_many_arguments)]
    fn log_step(
        &mut self,
        step: usize,
        url: Option<String>,
        decision: &Decision,
        executed: bool,
        seconds: f64,
        sent_image_crop_rect: Rect,
        prepared: &PreparedImage,
        full_screen_size: (u32, u32),
    ) {
        if let Some(ref mut logger) = self.logger {
            let box_crop_px = decision.bbox.map(|b| {
                let (cw, ch) = prepared.crop_size_px;
                CropBoxPx {
                    top: (b.top * (ch as f64)).round().max(0.0) as u32,
                    left: (b.left * (cw as f64)).round().max(0.0) as u32,
                    bottom: (b.bottom * (ch as f64)).round().max(0.0) as u32,
                    right: (b.right * (cw as f64)).round().max(0.0) as u32,
                }
            });
            let rec = StepRecord {
                step,
                url,
                kind: decision.kind.as_str().to_string(),
                kind_confidence: decision.kind_confidence,
                target: decision.target.clone(),
                box_crop_px,
                box_confidence: decision.box_confidence,
                reasked: decision.reasked,
                click_point: decision.click_point_screen_pt,
                seconds,
                results_noul: decision.results_noul,
                done_noul: decision.results_noul,
                sent_image_crop_rect,
                sent_size: prepared.sent_size_px,
                full_screen_size,
                decision: decision.clone(),
                executed,
                click_source: decision.click_source.clone(),
            };
            let _ = logger.log_entry(&rec);
        }
    }

    fn record_pending_self_fix(
        &self,
        pending: PendingSelfFixClick,
        current_url: &Option<String>,
        current_img: Option<&image::RgbaImage>,
    ) {
        let Some(store) = self.decider.experience_store() else {
            return;
        };
        let url_changed = current_url != &pending.prev_url;
        let screen_changed = current_img
            .map(|img| is_changed(&pending.prev_img, img))
            .unwrap_or(false);
        let won = url_changed || screen_changed;
        let evidence = if won {
            match (url_changed, screen_changed) {
                (true, true) => format!(
                    "step {} click '{}' changed URL ({} -> {}) and screen content",
                    pending.step,
                    pending.target,
                    pending.prev_url.as_deref().unwrap_or("none"),
                    current_url.as_deref().unwrap_or("none")
                ),
                (true, false) => format!(
                    "step {} click '{}' changed URL ({} -> {})",
                    pending.step,
                    pending.target,
                    pending.prev_url.as_deref().unwrap_or("none"),
                    current_url.as_deref().unwrap_or("none")
                ),
                _ => format!(
                    "step {} click '{}' changed screen content on {}",
                    pending.step,
                    pending.target,
                    pending.prev_url.as_deref().unwrap_or("page")
                ),
            }
        } else {
            String::new()
        };
        let _ = store.record_outcome(
            &pending.preset,
            &pending.host,
            &pending.target,
            &pending.lesson,
            &pending.chosen,
            pending.confidence,
            won,
            if won { Some(evidence.as_str()) } else { None },
        );
    }
}
