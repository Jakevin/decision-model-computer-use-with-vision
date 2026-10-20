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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedHttpsUrl {
    host: String,
    path: String,
}

fn parse_https_url(raw: &str) -> Option<ParsedHttpsUrl> {
    let trimmed = raw.trim();
    let rest = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("HTTPS://"))?;
    if rest.is_empty() {
        return None;
    }

    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    let host = authority
        .split(':')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if host.is_empty() {
        return None;
    }

    let after_authority = &rest[authority_end..];
    let path = if after_authority.starts_with('/') {
        let path_end = after_authority
            .find(['?', '#'])
            .unwrap_or(after_authority.len());
        after_authority[..path_end].to_string()
    } else {
        "/".to_string()
    };

    Some(ParsedHttpsUrl { host, path })
}

fn parse_allow_prefix(raw_prefix: &str) -> Option<(String, String)> {
    let trimmed = raw_prefix.trim();
    if trimmed.is_empty() {
        return None;
    }
    let without_scheme = if trimmed.contains("://") {
        trimmed
            .strip_prefix("https://")
            .or_else(|| trimmed.strip_prefix("HTTPS://"))?
    } else {
        trimmed
    };
    if without_scheme.is_empty() {
        return None;
    }

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
        return None;
    }

    let after_authority = &without_scheme[authority_end..];
    let path = if after_authority.starts_with('/') {
        let path_end = after_authority
            .find(['?', '#'])
            .unwrap_or(after_authority.len());
        let p = &after_authority[..path_end];
        if p.is_empty() {
            "/".to_string()
        } else {
            p.to_string()
        }
    } else {
        "/".to_string()
    };

    Some((host, path))
}

pub fn url_matches_prefix(url: &str, prefix: &str) -> bool {
    let Some(parsed_url) = parse_https_url(url) else {
        return false;
    };
    let Some((prefix_host, prefix_path)) = parse_allow_prefix(prefix) else {
        return false;
    };

    let host_ok = if let Some(base_domain) = prefix_host.strip_prefix("*.") {
        if base_domain.is_empty() {
            false
        } else {
            let suffix = format!(".{}", base_domain);
            parsed_url.host.ends_with(&suffix) && parsed_url.host.len() > suffix.len()
        }
    } else {
        parsed_url.host == prefix_host
    };

    if !host_ok {
        return false;
    }

    if prefix_path == "/" {
        parsed_url.path.starts_with('/')
    } else {
        let p = prefix_path.trim_end_matches('/');
        parsed_url.path == p
            || parsed_url
                .path
                .strip_prefix(p)
                .map(|rem| rem.starts_with('/'))
                .unwrap_or(false)
    }
}

pub fn derive_host_prefix_from_url(url: &str) -> Option<String> {
    let parsed = parse_https_url(url)?;
    Some(format!("https://{}", parsed.host))
}

pub fn resolve_allowed_url_prefixes(
    explicit_prefixes: &[String],
    preset_prefixes: &[String],
    start_url: Option<&str>,
) -> Vec<String> {
    if !explicit_prefixes.is_empty() {
        return explicit_prefixes.to_vec();
    }
    if !preset_prefixes.is_empty() {
        return preset_prefixes.to_vec();
    }
    if let Some(url) = start_url {
        if let Some(derived) = derive_host_prefix_from_url(url) {
            return vec![derived];
        }
    }
    Vec::new()
}

pub fn is_sensitive_fill_label(label: &str) -> bool {
    let lower = label.to_lowercase();
    ["name", "email", "phone", "card", "password"]
        .iter()
        .any(|w| lower.contains(w))
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
            allowed_hosts: Vec::new(),
            forbidden_target_words: vec![
                "sign in".to_string(),
                "signin".to_string(),
                "log in".to_string(),
                "login".to_string(),
                "password".to_string(),
                "book".to_string(),
                "purchase".to_string(),
                "pay".to_string(),
                "place order".to_string(),
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

    pub fn with_allowed_prefixes(mut self, prefixes: Vec<String>) -> Self {
        self.allowed_hosts = prefixes;
        self
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
        let allowed = self
            .allowed_hosts
            .iter()
            .any(|prefix| url_matches_prefix(url, prefix));

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
    fn url_prefix_check_host_subdomain_path_scheme_and_lookalike() {
        let policy = SafetyPolicy::default().with_allowed_prefixes(vec![
            "https://www.google.com/travel/flights".to_string(),
            "https://techcrunch.com".to_string(),
            "https://*.bizzabo.com".to_string(),
        ]);

        assert!(policy
            .check_url("https://www.google.com/travel/flights?q=SFO")
            .is_ok());
        assert!(policy
            .check_url("https://www.google.com/travel/flights/search")
            .is_ok());
        assert!(policy.check_url("https://www.google.com/maps").is_err());

        assert!(policy
            .check_url("https://techcrunch.com/events/disrupt/")
            .is_ok());

        assert!(policy
            .check_url("https://events.bizzabo.com/tc-disrupt/tickets")
            .is_ok());
        assert!(policy
            .check_url("https://register.events.bizzabo.com/checkout")
            .is_ok());
        assert!(policy
            .check_url("https://evil-bizzabo.com/tickets")
            .is_err());
        assert!(policy
            .check_url("https://evil-techcrunch.com/events")
            .is_err());
        assert!(policy
            .check_url("https://techcrunch.com.evil.com/events")
            .is_err());

        assert!(policy
            .check_url("http://techcrunch.com/events")
            .is_err());
        assert!(policy
            .check_url("http://www.google.com/travel/flights")
            .is_err());
    }

    #[test]
    fn default_safety_policy_has_no_hardcoded_allowlist_and_derives_from_start_url() {
        let default_policy = SafetyPolicy::default();
        assert!(default_policy.allowed_hosts.is_empty());
        assert!(default_policy
            .check_url("https://www.google.com/travel/flights")
            .is_err());

        // When neither --allow-url-prefix nor preset gives an allowlist, derive from --start-url's host
        let derived = resolve_allowed_url_prefixes(
            &[],
            &[],
            Some("https://techcrunch.com/events/tc-disrupt-2026/?utm=1"),
        );
        assert_eq!(derived, vec!["https://techcrunch.com".to_string()]);

        // When preset gives an allowlist and --allow-url-prefix is empty, use preset
        let from_preset = resolve_allowed_url_prefixes(
            &[],
            &["https://www.google.com/travel/flights".to_string()],
            Some("https://www.google.com/travel/flights?hl=en"),
        );
        assert_eq!(
            from_preset,
            vec!["https://www.google.com/travel/flights".to_string()]
        );

        // When --allow-url-prefix is supplied, use it
        let explicit = resolve_allowed_url_prefixes(
            &[
                "https://techcrunch.com".to_string(),
                "https://*.bizzabo.com".to_string(),
            ],
            &["https://www.google.com/travel/flights".to_string()],
            Some("https://techcrunch.com/events"),
        );
        assert_eq!(
            explicit,
            vec![
                "https://techcrunch.com".to_string(),
                "https://*.bizzabo.com".to_string(),
            ]
        );
    }

    #[test]
    fn forbidden_target_words_blocked_and_checkout_allowed_by_word_filter() {
        let policy = SafetyPolicy::default();
        assert!(policy.check_target("Sign In button").is_err());
        assert!(policy.check_target("Place Order").is_err());
        assert!(policy.check_target("Pay Now").is_err());
        assert!(policy.check_target("Purchase ticket").is_err());
        assert!(policy.check_target("Book flight").is_err());
        assert!(policy.check_target("checkout button").is_ok());
        assert!(policy.check_target("origin field").is_ok());
    }
}
