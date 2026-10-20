use crate::api::{Answer, ApiError, BoundingBox, DecisionBackend, Question, Request, DEFAULT_MODEL};
use crate::privacy::PreparedImage;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const NOTHING_TO_CLICK: &str = "nothing to click";
pub const LOW_CONFIDENCE_THRESHOLD: f64 = 0.2;

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
#[serde(rename_all = "lowercase")]
pub enum ActionKind {
    Click,
    Type,
    Scroll,
    Wait,
    Done,
}

impl ActionKind {
    pub fn from_choice(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "click" => Some(Self::Click),
            "type" => Some(Self::Type),
            "scroll" => Some(Self::Scroll),
            "wait" => Some(Self::Wait),
            "done" => Some(Self::Done),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub kind: ActionKind,
    pub target: Option<String>,
    pub click_point_screen_pt: Option<(f64, f64)>,
    pub bbox: Option<BoundingBox>,
    pub confidence: f64,
    pub reasked: bool,
    pub text_to_type: Option<String>,
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

pub fn format_state(goal: &str, history: &[String]) -> String {
    let recent: Vec<&String> = history.iter().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect();
    if recent.is_empty() {
        format!("Goal: {}\nHistory: (none)", goal)
    } else {
        let joined = recent
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{}. {}", i + 1, h))
            .collect::<Vec<_>>()
            .join("\n");
        format!("Goal: {}\nHistory:\n{}", goal, joined)
    }
}

pub struct VisionDecider<B: DecisionBackend> {
    backend: B,
    model: String,
    targets: Vec<TargetSpec>,
    type_values: BTreeMap<String, String>,
}

impl<B: DecisionBackend> VisionDecider<B> {
    pub fn new(backend: B, targets: Vec<TargetSpec>) -> Self {
        Self {
            backend,
            model: DEFAULT_MODEL.to_string(),
            targets,
            type_values: BTreeMap::new(),
        }
    }

    pub fn with_type_values(mut self, type_values: BTreeMap<String, String>) -> Self {
        self.type_values = type_values;
        self
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn targets(&self) -> &[TargetSpec] {
        &self.targets
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
        let state = format_state(goal, history);

        let mut kind_criteria = BTreeMap::new();
        kind_criteria.insert(
            "click".to_string(),
            Some("Click an interactive element on the page".to_string()),
        );
        kind_criteria.insert(
            "type".to_string(),
            Some("Type text into the currently focused input field".to_string()),
        );
        kind_criteria.insert(
            "scroll".to_string(),
            Some("Scroll down the page to reveal more content".to_string()),
        );
        kind_criteria.insert(
            "wait".to_string(),
            Some("Wait for the page to finish loading".to_string()),
        );
        kind_criteria.insert(
            "done".to_string(),
            Some("The goal has been achieved or no further action is needed".to_string()),
        );

        let mut target_criteria = BTreeMap::new();
        for t in &self.targets {
            target_criteria.insert(t.label.clone(), t.description.clone());
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

        let req1 = Request {
            model: self.model.clone(),
            state: state.clone(),
            images: vec![img.data_url.clone()],
            questions,
        };

        let resp1 = self.backend.ask(&req1)?;

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

        let action_kind = ActionKind::from_choice(&kind_choice)
            .ok_or_else(|| DeciderError::UnknownActionKind(kind_choice.clone()))?;

        if action_kind == ActionKind::Click {
            let tgt = target_choice
                .clone()
                .unwrap_or_else(|| NOTHING_TO_CLICK.to_string());
            if tgt == NOTHING_TO_CLICK {
                return Ok(Decision {
                    kind: ActionKind::Wait,
                    target: Some(tgt),
                    click_point_screen_pt: None,
                    bbox: None,
                    confidence: kind_conf,
                    reasked: false,
                    text_to_type: None,
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
            return Ok(Decision {
                kind: ActionKind::Click,
                target: Some(tgt),
                click_point_screen_pt: Some(pt),
                bbox: Some(bbox),
                confidence: box_conf,
                reasked,
                text_to_type: None,
            });
        }

        let text_to_type = if action_kind == ActionKind::Type {
            target_choice
                .as_ref()
                .and_then(|t| self.type_values.get(t).cloned())
        } else {
            None
        };

        Ok(Decision {
            kind: action_kind,
            target: target_choice,
            click_point_screen_pt: None,
            bbox: None,
            confidence: kind_conf,
            reasked: false,
            text_to_type,
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
        let resp = self.backend.ask(&req)?;
        match resp.answers.get("target_box") {
            Some(Answer::Box { bbox, confidence }) => Ok((*bbox, *confidence)),
            Some(_) => Err(DeciderError::UnexpectedAnswerType("target_box".to_string())),
            None => Err(DeciderError::MissingAnswer("target_box".to_string())),
        }
    }
}
