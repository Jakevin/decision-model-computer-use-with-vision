use crate::capture::Rect;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SafetyError {
    #[error("Click at ({x:.1}, {y:.1}) is inside the protected browser tab/address strip (y < {strip_limit:.1})")]
    ClickInTabStrip { x: f64, y: f64, strip_limit: f64 },
    #[error("Click at ({x:.1}, {y:.1}) is outside the Chrome window bounds")]
    ClickOutsideWindow { x: f64, y: f64 },
    #[error("URL '{0}' is not in the allowed hosts list")]
    DisallowedUrl(String),
    #[error("Exceeded maximum consecutive waits ({0})")]
    MaxConsecutiveWaitsExceeded(usize),
    #[error("Target label '{label}' contains forbidden word '{word}'")]
    ForbiddenTarget { label: String, word: String },
    #[error("Refusing to type sensitive or password-like content")]
    SensitiveTextBlocked,
    #[error("Google Chrome is not the frontmost application")]
    NotFrontmost,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPolicy {
    pub tab_strip_pt: f64,
    pub max_consecutive_waits: usize,
    pub max_steps: usize,
    pub allowed_hosts: Vec<String>,
    pub forbidden_target_words: Vec<String>,
    consecutive_waits: usize,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            tab_strip_pt: 80.0,
            max_consecutive_waits: 2,
            max_steps: 30,
            allowed_hosts: vec!["www.google.com/travel/flights".to_string()],
            forbidden_target_words: vec![
                "sign in".to_string(),
                "signin".to_string(),
                "log in".to_string(),
                "login".to_string(),
                "password".to_string(),
                "book".to_string(),
                "purchase".to_string(),
                "pay".to_string(),
                "checkout".to_string(),
                "buy".to_string(),
                "credit card".to_string(),
            ],
            consecutive_waits: 0,
        }
    }
}

impl SafetyPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check_click(
        &self,
        point_in_screen_pt: (f64, f64),
        window_rect: Rect,
    ) -> Result<(), SafetyError> {
        let (x, y) = point_in_screen_pt;
        let strip_limit = window_rect.y + self.tab_strip_pt;
        if y < strip_limit {
            return Err(SafetyError::ClickInTabStrip {
                x,
                y,
                strip_limit,
            });
        }
        if x < window_rect.x
            || x > window_rect.x + window_rect.width
            || y > window_rect.y + window_rect.height
        {
            return Err(SafetyError::ClickOutsideWindow { x, y });
        }
        Ok(())
    }

    pub fn check_url(&self, url: &str) -> Result<(), SafetyError> {
        let trimmed = url.trim();
        let without_scheme = trimmed
            .strip_prefix("https://")
            .or_else(|| trimmed.strip_prefix("http://"))
            .unwrap_or(trimmed);

        let allowed = self
            .allowed_hosts
            .iter()
            .any(|prefix| without_scheme.starts_with(prefix.as_str()));

        if allowed {
            Ok(())
        } else {
            Err(SafetyError::DisallowedUrl(url.to_string()))
        }
    }

    pub fn on_wait(&mut self) -> Result<usize, SafetyError> {
        self.consecutive_waits += 1;
        if self.consecutive_waits > self.max_consecutive_waits {
            Err(SafetyError::MaxConsecutiveWaitsExceeded(
                self.max_consecutive_waits,
            ))
        } else {
            Ok(self.consecutive_waits)
        }
    }

    pub fn reset_wait(&mut self) {
        self.consecutive_waits = 0;
    }

    pub fn consecutive_waits(&self) -> usize {
        self.consecutive_waits
    }

    pub fn check_target(&self, label: &str) -> Result<(), SafetyError> {
        let lower = label.to_lowercase();
        for word in &self.forbidden_target_words {
            let w = word.to_lowercase();
            if lower.contains(&w) {
                return Err(SafetyError::ForbiddenTarget {
                    label: label.to_string(),
                    word: word.clone(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_prefix_check() {
        let policy = SafetyPolicy::default();
        assert!(policy
            .check_url("https://www.google.com/travel/flights?q=SFO")
            .is_ok());
        assert!(policy
            .check_url("https://example.com/phishing")
            .is_err());
    }

    #[test]
    fn forbidden_target_words_blocked() {
        let policy = SafetyPolicy::default();
        assert!(policy.check_target("Sign In button").is_err());
        assert!(policy.check_target("Proceed to Checkout").is_err());
        assert!(policy.check_target("origin field").is_ok());
    }
}
