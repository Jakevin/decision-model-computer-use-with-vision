use crate::api::ApiKey;
use crate::logging::redact;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_EXPERIENCE_DIR: &str = "./experience";
pub const MIN_USES_FOR_DISABLE: usize = 4;
pub const MIN_WIN_RATE: f64 = 0.5;

pub fn strip_image_bytes(input: &str) -> String {
    let mut out = input.to_string();
    if let Ok(re) = Regex::new(r"(?i)data:image/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+") {
        out = re.replace_all(&out, "[image_omitted]").into_owned();
    }
    if let Ok(re_raw) = Regex::new(r"(?i)data:image/[^\s\x22\x27)]*") {
        out = re_raw.replace_all(&out, "[image_omitted]").into_owned();
    }
    out
}

/// Drop query strings and fragments from URLs so tokens / personal data in
/// query parameters never land in the experience store.
pub fn strip_url_queries(input: &str) -> String {
    match Regex::new(r"(?i)(https?://[^\s?#\x22\x27()<>]+)[?#][^\s\x22\x27()<>]*") {
        Ok(re) => re.replace_all(input, "$1").into_owned(),
        Err(_) => input.to_string(),
    }
}

pub fn sanitize_text(input: &str, api_key: Option<&ApiKey>) -> String {
    let no_img = strip_url_queries(&strip_image_bytes(input));
    let redacted = redact(&no_img, api_key);
    redacted
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn sanitize_path_component(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "default".to_string();
    }
    let mapped: String = trimmed
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let clean = mapped.trim_matches('.').trim_matches('_');
    if clean.is_empty() {
        "default".to_string()
    } else {
        clean.to_string()
    }
}

pub fn extract_host_from_url(url: Option<&str>) -> String {
    let Some(raw) = url else {
        return "default".to_string();
    };
    let trimmed = raw.trim();
    let without_scheme = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("HTTPS://"))
        .or_else(|| trimmed.strip_prefix("http://"))
        .or_else(|| trimmed.strip_prefix("HTTP://"))
        .unwrap_or(trimmed);
    let authority_end = without_scheme
        .find(['/', '?', '#'])
        .unwrap_or(without_scheme.len());
    let authority = &without_scheme[..authority_end];
    let host = authority
        .split(':')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if host.is_empty() {
        "default".to_string()
    } else {
        sanitize_path_component(&host)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperienceEntry {
    pub cue: String,
    pub lesson: String,
    pub evidence: String,
    pub confidence: f64,
    pub adopted_candidate: String,
    pub uses: usize,
    pub wins: usize,
}

impl ExperienceEntry {
    pub fn win_rate(&self) -> f64 {
        if self.uses == 0 {
            0.0
        } else {
            (self.wins as f64) / (self.uses as f64)
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.uses >= MIN_USES_FOR_DISABLE && self.win_rate() < MIN_WIN_RATE
    }

    pub fn is_active(&self) -> bool {
        !self.is_disabled()
    }

    pub fn sanitized(&self, api_key: Option<&ApiKey>) -> Self {
        Self {
            cue: sanitize_text(&self.cue, api_key),
            lesson: sanitize_text(&self.lesson, api_key),
            evidence: sanitize_text(&self.evidence, api_key),
            confidence: self.confidence,
            adopted_candidate: sanitize_text(&self.adopted_candidate, api_key),
            uses: self.uses,
            wins: self.wins,
        }
    }

    pub fn to_markdown(&self, api_key: Option<&ApiKey>) -> String {
        let clean = self.sanitized(api_key);
        let raw = format!(
            "---\ncue: {}\nlesson: {}\nevidence: {}\nconfidence: {:.4}\nadopted_candidate: {}\nuses: {}\nwins: {}\n---\n",
            clean.cue,
            clean.lesson,
            clean.evidence,
            clean.confidence,
            clean.adopted_candidate,
            clean.uses,
            clean.wins,
        );
        redact(&strip_image_bytes(&raw), api_key)
    }

    pub fn from_markdown(raw: &str) -> Result<Self, String> {
        let mut cue: Option<String> = None;
        let mut lesson: Option<String> = None;
        let mut evidence: Option<String> = None;
        let mut confidence: Option<f64> = None;
        let mut adopted_candidate: Option<String> = None;
        let mut uses: Option<usize> = None;
        let mut wins: Option<usize> = None;

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed == "---" || trimmed.starts_with('#') {
                continue;
            }
            let stripped = trimmed.strip_prefix("- ").unwrap_or(trimmed);
            let Some((k, v)) = stripped.split_once(':') else {
                continue;
            };
            let key = k.trim().trim_matches('*').to_lowercase();
            let val = v.trim().to_string();
            match key.as_str() {
                "cue" => cue = Some(val),
                "lesson" => lesson = Some(val),
                "evidence" => evidence = Some(val),
                "confidence" => {
                    confidence = val.parse::<f64>().ok();
                }
                "adopted_candidate" => adopted_candidate = Some(val),
                "uses" => {
                    uses = val.parse::<usize>().ok();
                }
                "wins" => {
                    wins = val.parse::<usize>().ok();
                }
                _ => {}
            }
        }

        let cue = cue.ok_or_else(|| "Missing 'cue' in experience entry".to_string())?;
        let lesson = lesson.ok_or_else(|| "Missing 'lesson' in experience entry".to_string())?;
        let evidence =
            evidence.ok_or_else(|| "Missing 'evidence' in experience entry".to_string())?;
        if evidence.trim().is_empty() {
            return Err("Experience entry 'evidence' cannot be empty".to_string());
        }
        let confidence = confidence.unwrap_or(0.0);
        let adopted_candidate = adopted_candidate.unwrap_or_else(|| "reask".to_string());
        let uses = uses.unwrap_or(1);
        let wins = wins.unwrap_or(0);

        Ok(Self {
            cue,
            lesson,
            evidence,
            confidence,
            adopted_candidate,
            uses,
            wins,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExperienceStore {
    base_dir: PathBuf,
    api_key: Option<ApiKey>,
}

impl Default for ExperienceStore {
    fn default() -> Self {
        Self::new(DEFAULT_EXPERIENCE_DIR)
    }
}

impl ExperienceStore {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            api_key: None,
        }
    }

    pub fn with_api_key(mut self, api_key: Option<ApiKey>) -> Self {
        self.api_key = api_key;
        self
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn host_dir(&self, preset: &str, host: &str) -> PathBuf {
        let p = sanitize_path_component(preset);
        let h = sanitize_path_component(host);
        self.base_dir.join(p).join(h)
    }

    pub fn entries_dir(&self, preset: &str, host: &str) -> PathBuf {
        self.host_dir(preset, host).join("entries")
    }

    pub fn entry_path(&self, preset: &str, host: &str, target_label: &str) -> PathBuf {
        let slug = sanitize_path_component(target_label);
        self.entries_dir(preset, host).join(format!("{}.md", slug))
    }

    pub fn index_path(&self, preset: &str, host: &str) -> PathBuf {
        self.host_dir(preset, host).join("INDEX.md")
    }

    pub fn load_entry(&self, preset: &str, host: &str, target_label: &str) -> Option<ExperienceEntry> {
        let direct = self.entry_path(preset, host, target_label);
        if direct.exists() {
            if let Ok(raw) = fs::read_to_string(&direct) {
                if let Ok(entry) = ExperienceEntry::from_markdown(&raw) {
                    return Some(entry);
                }
            }
        }
        let entries_dir = self.entries_dir(preset, host);
        let read_dir = fs::read_dir(&entries_dir).ok()?;
        let trimmed_target = target_label.trim();
        for item in read_dir.flatten() {
            let path = item.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            if let Ok(raw) = fs::read_to_string(&path) {
                if let Ok(entry) = ExperienceEntry::from_markdown(&raw) {
                    if entry.cue.trim() == trimmed_target {
                        return Some(entry);
                    }
                }
            }
        }
        None
    }

    pub fn lookup(&self, preset: &str, host: &str, target_label: &str) -> Option<ExperienceEntry> {
        let entry = self.load_entry(preset, host, target_label)?;
        if entry.is_disabled() {
            None
        } else {
            Some(entry)
        }
    }

    pub fn write_entry(
        &self,
        preset: &str,
        host: &str,
        entry: &ExperienceEntry,
    ) -> Result<PathBuf, String> {
        let clean = entry.sanitized(self.api_key.as_ref());
        if clean.cue.trim().is_empty() {
            return Err("Experience entry cue cannot be empty".to_string());
        }
        if clean.lesson.trim().is_empty() {
            return Err("Experience entry lesson cannot be empty".to_string());
        }
        if clean.evidence.trim().is_empty() {
            return Err("Experience entry requires non-empty evidence".to_string());
        }

        let dir = self.entries_dir(preset, host);
        fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create entries dir '{}': {}", dir.display(), e))?;

        let path = self.entry_path(preset, host, &clean.cue);
        let md = clean.to_markdown(self.api_key.as_ref());
        fs::write(&path, md)
            .map_err(|e| format!("Failed to write experience entry '{}': {}", path.display(), e))?;

        let _ = self.rebuild_index(preset, host)?;
        Ok(path)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_outcome(
        &self,
        preset: &str,
        host: &str,
        target_label: &str,
        lesson: &str,
        adopted_candidate: &str,
        confidence: f64,
        won: bool,
        evidence: Option<&str>,
    ) -> Result<Option<ExperienceEntry>, String> {
        if let Some(mut existing) = self.load_entry(preset, host, target_label) {
            existing.uses += 1;
            if won {
                existing.wins += 1;
                if let Some(ev) = evidence.map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    existing.evidence = sanitize_text(ev, self.api_key.as_ref());
                }
                existing.confidence = confidence;
                existing.adopted_candidate =
                    sanitize_text(adopted_candidate, self.api_key.as_ref());
                if !lesson.trim().is_empty() {
                    existing.lesson = sanitize_text(lesson, self.api_key.as_ref());
                }
            }
            self.write_entry(preset, host, &existing)?;
            Ok(Some(existing.sanitized(self.api_key.as_ref())))
        } else if won {
            let ev = evidence
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| "Experience entry requires non-empty evidence".to_string())?;
            let entry = ExperienceEntry {
                cue: target_label.to_string(),
                lesson: lesson.to_string(),
                evidence: ev.to_string(),
                confidence,
                adopted_candidate: adopted_candidate.to_string(),
                uses: 1,
                wins: 1,
            };
            let clean = entry.sanitized(self.api_key.as_ref());
            self.write_entry(preset, host, &clean)?;
            Ok(Some(clean))
        } else {
            Ok(None)
        }
    }

    pub fn list_entries(&self, preset: &str, host: &str) -> Vec<ExperienceEntry> {
        let entries_dir = self.entries_dir(preset, host);
        let Ok(read_dir) = fs::read_dir(&entries_dir) else {
            return Vec::new();
        };
        let mut entries = Vec::new();
        for item in read_dir.flatten() {
            let path = item.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            if let Ok(raw) = fs::read_to_string(&path) {
                if let Ok(entry) = ExperienceEntry::from_markdown(&raw) {
                    entries.push(entry);
                }
            }
        }
        entries.sort_by(|a, b| a.cue.cmp(&b.cue));
        entries
    }

    pub fn rebuild_index(&self, preset: &str, host: &str) -> Result<PathBuf, String> {
        let host_dir = self.host_dir(preset, host);
        fs::create_dir_all(&host_dir)
            .map_err(|e| format!("Failed to create host dir '{}': {}", host_dir.display(), e))?;

        let entries = self.list_entries(preset, host);
        let mut lines = Vec::new();
        lines.push(format!(
            "# Experience Index ({}/{})",
            sanitize_path_component(preset),
            sanitize_path_component(host)
        ));
        lines.push(String::new());
        for e in &entries {
            let clean = e.sanitized(self.api_key.as_ref());
            let status = if clean.is_disabled() {
                "disabled"
            } else {
                "active"
            };
            lines.push(format!(
                "- **{}** [{}]: lesson=`{}` candidate=`{}` conf={:.4} uses={} wins={} evidence=`{}`",
                clean.cue,
                status,
                clean.lesson,
                clean.adopted_candidate,
                clean.confidence,
                clean.uses,
                clean.wins,
                clean.evidence
            ));
        }
        lines.push(String::new());
        let content = redact(&strip_image_bytes(&lines.join("\n")), self.api_key.as_ref());

        let index_path = self.index_path(preset, host);
        fs::write(&index_path, &content)
            .map_err(|e| format!("Failed to write INDEX.md '{}': {}", index_path.display(), e))?;

        let root_index_path = self.base_dir.join("INDEX.md");
        let _ = fs::write(&root_index_path, &content);

        Ok(index_path)
    }
}
