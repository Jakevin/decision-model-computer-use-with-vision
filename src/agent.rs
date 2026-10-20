use crate::api::DecisionBackend;
use crate::capture::{CaptureError, Screen};
use crate::decider::{ActionKind, DeciderError, Decision, VisionDecider};
use crate::input::{Input, InputError};
use crate::logging::RunLogger;
use crate::privacy::{prepare_image, AuditLog, PrivacyError};
use crate::safety::{SafetyError, SafetyPolicy};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    Done,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: usize,
    pub decision: Decision,
    pub executed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunSummary {
    pub steps_taken: usize,
    pub stop_reason: StopReason,
    pub history: Vec<String>,
}

pub struct Agent<S: Screen, B: DecisionBackend, I: Input> {
    screen: S,
    decider: VisionDecider<B>,
    input: I,
    policy: SafetyPolicy,
    audit_log: AuditLog,
    logger: Option<RunLogger>,
    dry_run: bool,
}

impl<S: Screen, B: DecisionBackend, I: Input> Agent<S, B, I> {
    pub fn new(screen: S, decider: VisionDecider<B>, input: I, policy: SafetyPolicy) -> Self {
        Self {
            screen,
            decider,
            input,
            policy,
            audit_log: AuditLog::new(),
            logger: None,
            dry_run: false,
        }
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

    pub fn run(&mut self, goal: &str) -> Result<AgentRunSummary, AgentError> {
        let mut history: Vec<String> = Vec::new();
        let max_steps = self.policy.max_steps;

        for step in 0..max_steps {
            let full = self.screen.capture()?;
            let win_rect = self.screen.chrome_window_rect();
            let scale = self.screen.scale_factor();

            let prepared = match prepare_image(&full, win_rect, scale) {
                Ok(p) => p,
                Err(PrivacyError::NoCrop) => {
                    return Ok(AgentRunSummary {
                        steps_taken: step,
                        stop_reason: StopReason::NoCrop,
                        history,
                    });
                }
                Err(e) => return Err(AgentError::Privacy(e)),
            };

            let win_rect = win_rect.expect("prepare_image succeeded so win_rect is Some");
            self.audit_log
                .record(&prepared, format!("step_{}_decision", step));

            let decision = self.decider.decide(goal, &history, &prepared)?;

            if let Some(ref target_label) = decision.target {
                if let Err(e) = self.policy.check_target(target_label) {
                    return Err(AgentError::Safety(e));
                }
            }

            let mut executed = false;

            match decision.kind {
                ActionKind::Done => {
                    history.push("done".to_string());
                    self.log_step(step, &decision, true);
                    return Ok(AgentRunSummary {
                        steps_taken: step + 1,
                        stop_reason: StopReason::Done,
                        history,
                    });
                }
                ActionKind::Wait => {
                    history.push("wait".to_string());
                    self.log_step(step, &decision, true);
                    if self.policy.on_wait().is_err() {
                        return Ok(AgentRunSummary {
                            steps_taken: step + 1,
                            stop_reason: StopReason::MaxWaitsExceeded,
                            history,
                        });
                    }
                }
                ActionKind::Click => {
                    self.policy.reset_wait();
                    if let Some(pt) = decision.click_point_screen_pt {
                        self.policy.check_click(pt, win_rect)?;
                        if !self.dry_run {
                            self.input.click(pt.0, pt.1)?;
                            executed = true;
                        }
                        let tgt = decision.target.as_deref().unwrap_or("unknown");
                        history.push(format!("click '{}' at ({:.1}, {:.1})", tgt, pt.0, pt.1));
                    }
                    self.log_step(step, &decision, executed);
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
                        executed = true;
                    }
                    let tgt = decision.target.as_deref().unwrap_or("field");
                    history.push(format!("type into '{}'", tgt));
                    self.log_step(step, &decision, executed);
                }
                ActionKind::Scroll => {
                    self.policy.reset_wait();
                    if !self.dry_run {
                        self.input.scroll_vertical(-5)?;
                        executed = true;
                    }
                    history.push("scroll down".to_string());
                    self.log_step(step, &decision, executed);
                }
            }
        }

        Ok(AgentRunSummary {
            steps_taken: max_steps,
            stop_reason: StopReason::StepLimitReached(max_steps),
            history,
        })
    }

    fn log_step(&mut self, step: usize, decision: &Decision, executed: bool) {
        if let Some(ref mut logger) = self.logger {
            let rec = StepRecord {
                step,
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
