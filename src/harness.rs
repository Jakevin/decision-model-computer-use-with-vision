use crate::decider::TargetSpec;
use crate::logging::redact;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::Path;

pub const DEFAULT_MAX_EXTRA_REQUESTS_PER_STEP: usize = 3;

pub const DEFAULT_CANDIDATE_ORDER: &[CandidateKind] = &[
    CandidateKind::Reask,
    CandidateKind::Experience,
    CandidateKind::Zoom,
    CandidateKind::Ocr,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CandidateKind {
    Reask,
    Experience,
    Zoom,
    Ocr,
}

impl CandidateKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Reask => "reask",
            Self::Experience => "experience",
            Self::Zoom => "zoom",
            Self::Ocr => "ocr",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "reask" => Some(Self::Reask),
            "experience" => Some(Self::Experience),
            "zoom" => Some(Self::Zoom),
            "ocr" => Some(Self::Ocr),
            _ => None,
        }
    }
}

fn deserialize_vec_or_default<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let opt = Option::<Vec<T>>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

fn deserialize_candidate_order<'de, D>(deserializer: D) -> Result<Vec<CandidateKind>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<Vec<CandidateKind>>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_else(default_candidate_order))
}

fn deserialize_max_extra_requests<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<usize>::deserialize(deserializer)?;
    Ok(opt.unwrap_or(DEFAULT_MAX_EXTRA_REQUESTS_PER_STEP))
}

pub fn default_candidate_order() -> Vec<CandidateKind> {
    DEFAULT_CANDIDATE_ORDER.to_vec()
}

pub fn default_max_extra_requests_per_step() -> usize {
    DEFAULT_MAX_EXTRA_REQUESTS_PER_STEP
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessTargetOverride {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, deserialize_with = "deserialize_vec_or_default")]
    pub reask: Vec<String>,
}

impl HarnessTargetOverride {
    pub fn new(
        label: impl Into<String>,
        description: Option<String>,
        reask: Vec<String>,
    ) -> Self {
        Self {
            label: label.into(),
            description,
            reask,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessOverrides {
    #[serde(default, deserialize_with = "deserialize_vec_or_default")]
    pub targets: Vec<HarnessTargetOverride>,
    #[serde(
        default = "default_candidate_order",
        deserialize_with = "deserialize_candidate_order"
    )]
    pub candidate_order: Vec<CandidateKind>,
    #[serde(
        default = "default_max_extra_requests_per_step",
        deserialize_with = "deserialize_max_extra_requests"
    )]
    pub max_extra_requests_per_step: usize,
}

impl Default for HarnessOverrides {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            candidate_order: default_candidate_order(),
            max_extra_requests_per_step: DEFAULT_MAX_EXTRA_REQUESTS_PER_STEP,
        }
    }
}

impl HarnessOverrides {
    pub fn parse_json(raw: &str) -> Result<Self, String> {
        let parsed: Self = serde_json::from_str(raw)
            .map_err(|e| redact(&format!("Harness JSON parse error: {}", e), None))?;
        for t in &parsed.targets {
            if t.label.trim().is_empty() {
                return Err("Harness target label cannot be empty".to_string());
            }
        }
        Ok(parsed)
    }

    pub fn load_from_path(path: &Path) -> Result<Option<Self>, String> {
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(path).map_err(|e| {
            redact(
                &format!("Failed to read harness file '{}': {}", path.display(), e),
                None,
            )
        })?;
        Self::parse_json(&raw).map(Some)
    }

    pub fn load_optional(path: Option<&Path>) -> Result<Option<Self>, String> {
        match path {
            Some(p) => Self::load_from_path(p),
            None => Ok(None),
        }
    }

    pub fn load_or_default(path: Option<&Path>) -> Result<Self, String> {
        Ok(Self::load_optional(path)?.unwrap_or_default())
    }

    pub fn find_target(&self, label: &str) -> Option<&HarnessTargetOverride> {
        let trimmed = label.trim();
        self.targets.iter().find(|t| t.label.trim() == trimmed)
    }

    pub fn reask_for(&self, label: &str) -> Option<&[String]> {
        self.find_target(label)
            .filter(|t| !t.reask.is_empty())
            .map(|t| t.reask.as_slice())
    }

    pub fn apply_to_targets(&self, base: &[TargetSpec]) -> Vec<TargetSpec> {
        let mut out = base.to_vec();
        for ov in &self.targets {
            if let Some(existing) = out.iter_mut().find(|t| t.label.trim() == ov.label.trim()) {
                if ov.description.is_some() {
                    existing.description = ov.description.clone();
                }
            } else {
                out.push(TargetSpec::new(ov.label.clone(), ov.description.clone()));
            }
        }
        out
    }
}

pub fn load_harness_file(path: &Path) -> Result<Option<HarnessOverrides>, String> {
    HarnessOverrides::load_from_path(path)
}
