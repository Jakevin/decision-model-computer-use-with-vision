use crate::api::DecisionBackend;
use crate::browser::{Browser, DefaultAllowedBrowser};
use crate::capture::{changed_fraction, is_changed, CaptureError, Rect, Screen};
use crate::decider::{
    compute_excluded_targets, ActionKind, DeciderError, Decision, DoneCondition, VisionDecider,
    LOW_CONFIDENCE_THRESHOLD, POPUP_PRESENT_THRESHOLD,
};
use crate::input::{Input, InputError, KeyCode};
use crate::logging::RunLogger;
use crate::privacy::{prepare_image, AuditLog, PreparedImage, PrivacyError};
use crate::safety::{
    is_sensitive_fill_label, resolve_allowed_url_prefixes, SafetyError, SafetyPolicy,
};
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
pub const LOW_BOX_CONF_EXCLUDE_STEPS: usize = 2;
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
        }
    }
}

pub struct Agent<S: Screen, B: DecisionBackend, I: Input, Br: Browser = DefaultAllowedBrowser> {
    screen: S,
    decider: VisionDecider<B>,
    input: I,
    policy: SafetyPolicy,
    browser: Br,
    start_url: Option<String>,
    flights_preset: bool,
    timing: Timing,
    audit_log: AuditLog,
    logger: Option<RunLogger>,
    dry_run: bool,
}

impl<S: Screen, B: DecisionBackend, I: Input> Agent<S, B, I, DefaultAllowedBrowser> {
    pub fn new(screen: S, decider: VisionDecider<B>, input: I, policy: SafetyPolicy) -> Self {
        Self {
            screen,
            decider,
            input,
            policy,
            browser: DefaultAllowedBrowser::default(),
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
            start_url: self.start_url,
            flights_preset: self.flights_preset,
            timing: self.timing,
            audit_log: self.audit_log,
            logger: self.logger,
            dry_run: self.dry_run,
        }
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
        let mut checkout_clicked = false;

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

            let full = match self.screen.capture() {
                Ok(f) => f,
                Err(e) => {
                    run_err = Some(AgentError::Capture(e));
                    break;
                }
            };
            let full_screen_size = (full.width(), full.height());
            let win_rect = self.screen.chrome_window_rect();
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

            let decision = match self
                .decider
                .decide_with_excluded(goal, &history, &prepared, &excluded_targets)
            {
                Ok(d) => d,
                Err(e) => {
                    run_err = Some(AgentError::Decider(e));
                    break;
                }
            };

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
                        true,
                        elapsed,
                        win_rect,
                        &prepared,
                        full_screen_size,
                    );
                    if self.policy.on_wait().is_err() {
                        stop_reason = StopReason::MaxWaitsExceeded;
                        break;
                    }
                }
                ActionKind::DismissPopup => {
                    let box_conf = decision.box_confidence.unwrap_or(0.0);
                    let closeish = decision.closeish_noul.unwrap_or(0.0);
                    let popup_present = decision.popup_present_noul.unwrap_or(0.0);
                    let tgt = decision.target.as_deref().unwrap_or("popup close");
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
                        if let Some(entry) =
                            sensitive_excluded.iter_mut().find(|(l, _)| l == tgt)
                        {
                            entry.1 = entry.1.max(LOW_BOX_CONF_EXCLUDE_STEPS);
                        } else {
                            sensitive_excluded.push((tgt.to_string(), LOW_BOX_CONF_EXCLUDE_STEPS));
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
                    } else if popup_present < POPUP_PRESENT_THRESHOLD
                        || closeish < DISMISS_CLOSEISH_THRESHOLD
                    {
                        if let Some(ref mut logger) = self.logger {
                            let evt = DismissRefusedEventRecord {
                                event: "dismiss_refused".to_string(),
                                step,
                                target: tgt.to_string(),
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
                            if !self.dry_run {
                                self.input.click(pt.0, pt.1)?;
                                executed = true;
                            }
                            clicks += 1;
                            clicked_this_step = Some(tgt.to_string());
                            last_click_point = Some(pt);
                            if executed && !self.timing.post_action_wait.is_zero() {
                                thread::sleep(self.timing.post_action_wait);
                            }
                            history.push(format!("click '{}' at ({:.1}, {:.1})", tgt, pt.0, pt.1));
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
                        if let Some(entry) =
                            sensitive_excluded.iter_mut().find(|(l, _)| l == tgt)
                        {
                            entry.1 = entry.1.max(LOW_BOX_CONF_EXCLUDE_STEPS);
                        } else {
                            sensitive_excluded.push((tgt.to_string(), LOW_BOX_CONF_EXCLUDE_STEPS));
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
                        step_clicks.push(None);
                        if !self.timing.step_wait.is_zero() {
                            thread::sleep(self.timing.step_wait);
                        }
                        continue;
                    }
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

                        if tgt.eq_ignore_ascii_case("checkout button") {
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

                        if !self.dry_run {
                            self.input.click(pt.0, pt.1)?;
                            executed = true;
                        }
                        clicks += 1;
                        clicked_this_step = Some(tgt.to_string());
                        last_click_point = Some(pt);

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
                    if !self.dry_run {
                        self.input.scroll_vertical(-5)?;
                        executed = true;
                        if !self.timing.post_action_wait.is_zero() {
                            thread::sleep(self.timing.post_action_wait);
                        }
                    }
                    history.push("scroll down".to_string());
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
            }
            step_clicks.push(clicked_this_step);
            if !self.timing.step_wait.is_zero() {
                thread::sleep(self.timing.step_wait);
            }
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
            };
            let _ = logger.log_entry(&rec);
        }
    }
}

fn is_sensitive_text(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("password") || lower.contains("sk-vsr-")
}
