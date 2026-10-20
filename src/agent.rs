use crate::api::DecisionBackend;
use crate::browser::{Browser, DefaultAllowedBrowser};
use crate::capture::{CaptureError, Rect, Screen};
use crate::decider::{
    compute_excluded_targets, ActionKind, DeciderError, Decision, VisionDecider,
};
use crate::input::{Input, InputError, KeyCode};
use crate::logging::RunLogger;
use crate::privacy::{prepare_image, AuditLog, PreparedImage, PrivacyError};
use crate::safety::{SafetyError, SafetyPolicy};
use serde::{Deserialize, Serialize};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

pub const RESULTS_DONE_CONSECUTIVE_THRESHOLD: f64 = 0.8;
pub const RESULTS_DONE_CONSECUTIVE_COUNT: usize = 2;
pub const RESULTS_MODEL_DONE_THRESHOLD: f64 = 0.5;
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
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            start_url_wait: Duration::from_secs(3),
            activate_chrome_wait: Duration::from_millis(ACTIVATE_CHROME_SLEEP_MS),
            before_fill_wait: Duration::from_millis(800),
            after_fill_wait: Duration::from_millis(1500),
            post_action_wait: Duration::from_millis(1000),
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    Done,
    ResultsPageWithPrices,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummaryJson {
    pub status: String,
    pub reason: String,
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
            timing: self.timing,
            audit_log: self.audit_log,
            logger: self.logger,
            dry_run: self.dry_run,
        }
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

        if let Some(ref url) = self.start_url {
            let _ = self.browser.activate_chrome();
            let _ = self.browser.set_active_tab_url(url);
            if !self.timing.start_url_wait.is_zero() {
                thread::sleep(self.timing.start_url_wait);
            }
        }

        let mut stop_reason = StopReason::StepLimitReached(max_steps);

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

            // Read active tab URL before capture
            let current_url = self.browser.active_tab_url();
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

            let full = self.screen.capture()?;
            let full_screen_size = (full.width(), full.height());
            let win_rect = self.screen.chrome_window_rect();
            let scale = self.screen.scale_factor();

            let prepared = match prepare_image(&full, win_rect, scale) {
                Ok(p) => p,
                Err(PrivacyError::NoCrop) => {
                    stop_reason = StopReason::NoCrop;
                    break;
                }
                Err(e) => return Err(AgentError::Privacy(e)),
            };

            let win_rect = win_rect.expect("prepare_image succeeded so win_rect is Some");
            let is_this_cropped = prepared.crop_size_px.0 > 0 && prepared.crop_size_px.1 > 0;
            all_images_cropped = all_images_cropped && is_this_cropped;

            self.audit_log
                .record(&prepared, format!("step_{}_decision", step));

            let excluded_targets = compute_excluded_targets(&step_clicks);
            let decision =
                self.decider
                    .decide_with_excluded(goal, &history, &prepared, &excluded_targets)?;

            if let Some(ref target_label) = decision.target {
                if let Err(e) = self.policy.check_target(target_label) {
                    return Err(AgentError::Safety(e));
                }
            }

            if decision.results_noul >= RESULTS_DONE_CONSECUTIVE_THRESHOLD {
                consecutive_results_high += 1;
            } else {
                consecutive_results_high = 0;
            }

            if consecutive_results_high >= RESULTS_DONE_CONSECUTIVE_COUNT {
                let elapsed = step_start.elapsed().as_secs_f64();
                total_step_secs += elapsed;
                steps_completed += 1;
                history.push("results page with prices".to_string());
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
                stop_reason = StopReason::ResultsPageWithPrices;
                break;
            }

            let mut executed = false;
            let mut clicked_this_step: Option<String> = None;

            match decision.kind {
                ActionKind::Done => {
                    if decision.results_noul >= RESULTS_MODEL_DONE_THRESHOLD {
                        let elapsed = step_start.elapsed().as_secs_f64();
                        total_step_secs += elapsed;
                        steps_completed += 1;
                        history.push("done".to_string());
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
                ActionKind::Click => {
                    self.policy.reset_wait();
                    if let Some(pt) = decision.click_point_screen_pt {
                        self.policy.check_click(pt, win_rect)?;
                        if !self.browser.is_chrome_frontmost() {
                            return Err(AgentError::Safety(SafetyError::NotFrontmost));
                        }
                        let tgt = decision.target.as_deref().unwrap_or("unknown");
                        let fill_val = self.decider.fill_for(tgt).map(|s| s.to_string());

                        if let Some(ref val) = fill_val {
                            if is_sensitive_text(val) {
                                return Err(AgentError::Safety(SafetyError::SensitiveTextBlocked));
                            }
                        }

                        if !self.dry_run {
                            self.input.click(pt.0, pt.1)?;
                            executed = true;
                        }
                        clicks += 1;
                        clicked_this_step = Some(tgt.to_string());

                        if let Some(val) = fill_val {
                            if !self.dry_run {
                                if !self.timing.before_fill_wait.is_zero() {
                                    thread::sleep(self.timing.before_fill_wait);
                                }
                                self.input.select_all()?;
                                self.input.type_text(&val)?;
                                self.input.release_modifiers()?;
                                if !self.timing.after_fill_wait.is_zero() {
                                    thread::sleep(self.timing.after_fill_wait);
                                }
                            }
                            history.push(format!("clicked and typed '{}' into {}", val, tgt));
                        } else {
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
        }

        let avg_seconds_per_step = if steps_completed > 0 {
            total_step_secs / (steps_completed as f64)
        } else {
            0.0
        };

        let summary = AgentRunSummary {
            steps_taken: steps_completed,
            stop_reason,
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
