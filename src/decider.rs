use crate::api::{
    Answer, ApiError, BoundingBox, DecisionBackend, Question, Request, Response, DEFAULT_MODEL,
};
use crate::privacy::PreparedImage;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

pub const NOTHING_TO_CLICK: &str = "nothing to click";
pub const LOW_CONFIDENCE_THRESHOLD: f64 = 0.2;
pub const RESULTS_QUESTION_INSTRUCTIONS: &str =
    "Does the image show a Google Flights results list with flight prices?";

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
pub enum ActionKind {
    #[serde(rename = "click")]
    Click,
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

pub fn compute_excluded_targets(step_clicks: &[Option<String>]) -> Vec<String> {
    let mut excluded = Vec::new();
    if step_clicks.len() >= 2 {
        if let (Some(prev1), Some(prev2)) = (
            &step_clicks[step_clicks.len() - 2],
            &step_clicks[step_clicks.len() - 1],
        ) {
            if prev1 == prev2 {
                excluded.push(prev1.clone());
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
    pub type_values: BTreeMap<String, String>,
    requests_sent: AtomicUsize,
    images_sent: AtomicUsize,
}

impl<B: DecisionBackend> VisionDecider<B> {
    pub fn new(backend: B, targets: Vec<TargetSpec>) -> Self {
        Self {
            backend,
            model: DEFAULT_MODEL.to_string(),
            targets,
            type_values: BTreeMap::new(),
            requests_sent: AtomicUsize::new(0),
            images_sent: AtomicUsize::new(0),
        }
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

    pub fn decide(
        &self,
        goal: &str,
        history: &[String],
        img: &PreparedImage,
    ) -> Result<Decision, DeciderError> {
        let step_clicks: Vec<Option<String>> =
            history.iter().map(|h| extract_clicked_target(h)).collect();
        let auto_excluded = compute_excluded_targets(&step_clicks);
        self.decide_with_excluded(goal, history, img, &auto_excluded)
    }

    pub fn decide_with_excluded(
        &self,
        goal: &str,
        history: &[String],
        img: &PreparedImage,
        excluded_targets: &[String],
    ) -> Result<Decision, DeciderError> {
        let state = format_state(goal, history);
        let active_fill_target = pending_fill_target(history, &self.type_values);

        let mut kind_criteria = BTreeMap::new();
        kind_criteria.insert(
            "click".to_string(),
            Some("Click an interactive element on the page".to_string()),
        );
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
        questions.insert(
            "results".to_string(),
            Question::Noul {
                instructions: RESULTS_QUESTION_INSTRUCTIONS.to_string(),
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

        let action_kind = ActionKind::from_choice(&kind_choice)
            .ok_or_else(|| DeciderError::UnknownActionKind(kind_choice.clone()))?;

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
                });
            }

            let (mut bbox, mut box_conf) = self.ask_box(&state, img, &tgt)?;
            let mut reasked = false;

            if box_conf < LOW_CONFIDENCE_THRESHOLD {
                let longer = self
                    .find_target_description(&tgt)
                    .map(|desc| format!("{} ({})", tgt, desc))
                    .unwrap_or_else(|| format!("Locate the interactive element: {}", tgt));
                let (bbox2, conf2) = self.ask_box(&state, img, &longer)?;
                bbox = bbox2;
                box_conf = conf2;
                reasked = true;
            }

            let pt = map_box_center_to_screen_pt(&bbox, img);
            let fill_text = self.type_values.get(&tgt).cloned();
            return Ok(Decision {
                kind: ActionKind::Click,
                kind_confidence: kind_conf,
                target: Some(tgt),
                click_point_screen_pt: Some(pt),
                bbox: Some(bbox),
                box_confidence: Some(box_conf),
                confidence: box_conf,
                reasked,
                text_to_type: fill_text,
                results_noul,
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

        Ok(Decision {
            kind: action_kind,
            kind_confidence: kind_conf,
            target: target_choice,
            click_point_screen_pt: None,
            bbox: None,
            box_confidence: None,
            confidence: kind_conf,
            reasked: false,
            text_to_type,
            results_noul,
        })
    }

    fn ask_box(
        &self,
        state: &str,
        img: &PreparedImage,
        instructions: &str,
    ) -> Result<(BoundingBox, f64), DeciderError> {
        let mut questions = BTreeMap::new();
        questions.insert(
            "target_box".to_string(),
            Question::Box {
                instructions: instructions.to_string(),
            },
        );
        let req = Request {
            model: self.model.clone(),
            state: state.to_string(),
            images: vec![img.data_url.clone()],
            questions,
        };
        let resp = self.ask_backend(&req)?;
        match resp.answers.get("target_box") {
            Some(Answer::Box { bbox, confidence }) => Ok((*bbox, *confidence)),
            Some(_) => Err(DeciderError::UnexpectedAnswerType("target_box".to_string())),
            None => Err(DeciderError::MissingAnswer("target_box".to_string())),
        }
    }
}
