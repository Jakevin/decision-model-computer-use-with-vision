use clap::{Parser, Subcommand};
use decision_cu::agent::{Agent, AgentError, Timing, FLIGHTS_DEFAULT_URL_PREFIX};
use decision_cu::api::{ApiKey, DecisionClient};
use decision_cu::decider::{load_targets_file, DoneCondition, Preset, VisionDecider};
use decision_cu::experience::DEFAULT_EXPERIENCE_DIR;
use decision_cu::harness::HarnessOverrides;
use decision_cu::logging::{redact, RunLogger};
use decision_cu::safety::{is_sensitive_fill_label, resolve_allowed_url_prefixes, SafetyPolicy};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "decision-cu",
    version,
    about = "Screenshot-only macOS computer-use agent driven by the vllm-sr Decision 3.0 API"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run the computer-use agent loop toward a specified task/goal
    Run {
        /// Natural language task/goal to accomplish (alias: --goal)
        #[arg(long = "task", visible_alias = "goal")]
        task: String,

        /// Maximum number of steps before stopping
        #[arg(long, default_value_t = 30)]
        max_steps: usize,

        /// Activate Chrome and navigate active tab to this URL before step 0
        #[arg(
            long = "start-url",
            value_name = "URL",
            conflicts_with = "start_url_preset"
        )]
        start_url: Option<String>,

        /// Best-effort start URL preset (e.g. one-way), mutually exclusive with --start-url
        #[arg(
            long = "start-url-preset",
            value_name = "PRESET",
            conflicts_with = "start_url"
        )]
        start_url_preset: Option<String>,

        /// Repeatable allowed URL prefix (https scheme, exact host or *.domain, path prefix)
        #[arg(long = "allow-url-prefix", value_name = "PREFIX")]
        allow_url_prefix: Vec<String>,

        /// Target preset: flights, events, events-zh, or generic
        #[arg(long = "preset", default_value = "flights", value_name = "PRESET")]
        preset: String,

        /// Path to a JSON file containing custom targets ([{"label":..,"description":..}])
        #[arg(long = "targets-file", value_name = "PATH")]
        targets_file: Option<PathBuf>,

        /// Done condition: flights-results, checkout-page, or model (defaults from --preset)
        #[arg(long = "done-when", value_name = "CONDITION")]
        done_when: Option<String>,

        /// Repeatable target-to-text fill mapping (e.g. --fill "origin field=Osaka")
        #[arg(long = "fill", value_name = "KEY=VALUE")]
        fill: Vec<String>,

        /// Optional harness JSON path for target overrides and self-fix probe order/budget
        #[arg(long = "harness", value_name = "PATH")]
        harness: Option<PathBuf>,

        /// Disable minimal in-run self-correction probe (baseline runs)
        #[arg(long = "no-self-fix", default_value_t = false)]
        no_self_fix: bool,

        /// Directory for experience store entries and INDEX.md
        #[arg(
            long = "experience-dir",
            default_value = DEFAULT_EXPERIENCE_DIR,
            value_name = "PATH"
        )]
        experience_dir: PathBuf,

        /// Perform capture and decision making without executing mouse/keyboard actions
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Print a single final JSON result object to stdout ({status, steps, reason, ...})
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CliResultOutput {
    pub status: String,
    pub steps: usize,
    pub reason: String,
    pub done_condition: String,
    pub clicks: usize,
    pub requests_sent: usize,
    pub images_sent: usize,
    pub avg_seconds_per_step: f64,
    pub all_images_cropped: bool,
    pub tab_events: usize,
    #[serde(default)]
    pub price_crops: Vec<String>,
}

impl CliResultOutput {
    pub fn error_or_blocked(status: &str, reason: impl Into<String>) -> Self {
        Self {
            status: status.to_string(),
            steps: 0,
            reason: reason.into(),
            done_condition: "flights-results".to_string(),
            clicks: 0,
            requests_sent: 0,
            images_sent: 0,
            avg_seconds_per_step: 0.0,
            all_images_cropped: true,
            tab_events: 0,
            price_crops: Vec::new(),
        }
    }

    pub fn with_done_condition(mut self, done_condition: DoneCondition) -> Self {
        self.done_condition = done_condition.as_str().to_string();
        self
    }

    pub fn with_counters(mut self, requests_sent: usize, images_sent: usize) -> Self {
        self.requests_sent = requests_sent;
        self.images_sent = images_sent;
        self
    }

    pub fn exit_code(&self) -> u8 {
        match self.status.as_str() {
            "done" => 0,
            "step_limit" => 2,
            "wait_limit" => 3,
            "blocked" => 4,
            _ => 1,
        }
    }
}

pub fn parse_fills(fills: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    for entry in fills {
        let (k, v) = entry
            .split_once('=')
            .ok_or_else(|| format!("Invalid --fill '{}': expected KEY=VALUE", entry))?;
        let key = k.trim();
        if key.is_empty() {
            return Err(format!("Invalid --fill '{}': key cannot be empty", entry));
        }
        if is_sensitive_fill_label(key) {
            return Err(format!(
                "Refusing --fill for sensitive label '{}'",
                key
            ));
        }
        map.insert(key.to_string(), v.to_string());
    }
    Ok(map)
}

pub fn resolve_start_url_preset(preset: &str) -> Result<String, String> {
    match preset.trim() {
        "one-way" => {
            Ok("https://www.google.com/travel/flights?hl=en&q=one%20way%20flights".to_string())
        }
        other => Err(format!(
            "Invalid --start-url-preset '{}': expected 'one-way'",
            other
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_run(
    task: &str,
    max_steps: usize,
    start_url: Option<String>,
    start_url_preset: Option<String>,
    allow_url_prefix: &[String],
    preset_str: &str,
    targets_file: Option<&std::path::Path>,
    done_when: Option<&str>,
    fills: &[String],
    harness_file: Option<&std::path::Path>,
    no_self_fix: bool,
    experience_dir: &std::path::Path,
    dry_run: bool,
) -> (CliResultOutput, Option<ApiKey>) {
    let preset = match Preset::from_name(preset_str) {
        Ok(p) => p,
        Err(e) => return (CliResultOutput::error_or_blocked("error", e), None),
    };

    let done_condition = match done_when {
        Some(dw) => match DoneCondition::from_name(dw) {
            Ok(dc) => dc,
            Err(e) => return (CliResultOutput::error_or_blocked("error", e), None),
        },
        None => preset.default_done_condition(),
    };

    let targets = match targets_file {
        Some(path) => match load_targets_file(path) {
            Ok(t) => t,
            Err(e) => {
                return (
                    CliResultOutput::error_or_blocked("error", e)
                        .with_done_condition(done_condition),
                    None,
                )
            }
        },
        None => preset.default_targets(),
    };

    let harness_opt = match HarnessOverrides::load_optional(harness_file) {
        Ok(h) => h,
        Err(e) => {
            return (
                CliResultOutput::error_or_blocked("error", redact(&e, None))
                    .with_done_condition(done_condition),
                None,
            )
        }
    };

    let resolved_start_url = match (start_url, start_url_preset) {
        (Some(u), _) => Some(u),
        (None, Some(sp)) => match resolve_start_url_preset(&sp) {
            Ok(u) => Some(u),
            Err(e) => {
                return (
                    CliResultOutput::error_or_blocked("error", e)
                        .with_done_condition(done_condition),
                    None,
                )
            }
        },
        (None, None) => None,
    };

    let type_values = match parse_fills(fills) {
        Ok(m) => m,
        Err(e) => {
            return (
                CliResultOutput::error_or_blocked("error", e).with_done_condition(done_condition),
                None,
            )
        }
    };

    let api_key = match ApiKey::from_env() {
        Ok(k) => k,
        Err(e) => {
            return (
                CliResultOutput::error_or_blocked("error", e.to_string())
                    .with_done_condition(done_condition),
                None,
            );
        }
    };

    let client = match DecisionClient::new(api_key.clone()) {
        Ok(c) => c,
        Err(e) => {
            return (
                CliResultOutput::error_or_blocked(
                    "error",
                    redact(&e.to_string(), Some(&api_key)),
                )
                .with_done_condition(done_condition),
                Some(api_key),
            );
        }
    };

    let decider = VisionDecider::new(client, targets)
        .with_preset(preset)
        .with_done_condition(done_condition)
        .with_type_values(type_values)
        .with_harness_opt(harness_opt)
        .with_experience_store(
            decision_cu::experience::ExperienceStore::new(experience_dir)
                .with_api_key(Some(api_key.clone())),
        )
        .with_self_fix(!no_self_fix);

    let preset_prefixes = if preset == Preset::Flights {
        vec![FLIGHTS_DEFAULT_URL_PREFIX.to_string()]
    } else {
        Vec::new()
    };
    let mut policy = SafetyPolicy::default().with_allowed_prefixes(resolve_allowed_url_prefixes(
        allow_url_prefix,
        &preset_prefixes,
        resolved_start_url.as_deref(),
    ));
    policy.max_steps = max_steps;

    let logger = match RunLogger::new_in_runs_dir(Some(api_key.clone())) {
        Ok(l) => l,
        Err(e) => {
            return (
                CliResultOutput::error_or_blocked(
                    "error",
                    redact(&e.to_string(), Some(&api_key)),
                )
                .with_done_condition(done_condition),
                Some(api_key),
            );
        }
    };

    #[cfg(target_os = "macos")]
    {
        use decision_cu::browser::MacBrowser;
        use decision_cu::capture::MacScreen;
        use decision_cu::input::EnigoInput;
        use decision_cu::ocr::VisionOcr;

        let screen = MacScreen::new();
        let input = match EnigoInput::new() {
            Ok(i) => i,
            Err(e) => {
                return (
                    CliResultOutput::error_or_blocked(
                        "error",
                        redact(&e.to_string(), Some(&api_key)),
                    )
                    .with_done_condition(done_condition),
                    Some(api_key),
                );
            }
        };

        let mut agent = Agent::new(screen, decider, input, policy)
            .with_browser(MacBrowser::new())
            .with_ocr(VisionOcr::new())
            .with_flights_preset(preset == Preset::Flights)
            .with_timing(Timing::default())
            .with_start_url(resolved_start_url)
            .with_logger(logger)
            .with_dry_run(dry_run);

        match agent.run(task) {
            Ok(summary) => {
                let s = summary.to_summary_json();
                (
                    CliResultOutput {
                        status: s.status,
                        steps: s.steps,
                        reason: redact(&s.reason, Some(&api_key)),
                        done_condition: s.done_condition,
                        clicks: s.clicks,
                        requests_sent: s.requests_sent,
                        images_sent: s.images_sent,
                        avg_seconds_per_step: s.avg_seconds_per_step,
                        all_images_cropped: s.all_images_cropped,
                        tab_events: s.tab_events,
                        price_crops: s.price_crops,
                    },
                    Some(api_key),
                )
            }
            Err(AgentError::Safety(s)) => (
                CliResultOutput::error_or_blocked(
                    "blocked",
                    redact(&s.to_string(), Some(&api_key)),
                )
                .with_done_condition(done_condition)
                .with_counters(
                    agent.decider().requests_sent(),
                    agent.decider().images_sent(),
                ),
                Some(api_key),
            ),
            Err(AgentError::Privacy(p)) => (
                CliResultOutput::error_or_blocked(
                    "blocked",
                    redact(&p.to_string(), Some(&api_key)),
                )
                .with_done_condition(done_condition)
                .with_counters(
                    agent.decider().requests_sent(),
                    agent.decider().images_sent(),
                ),
                Some(api_key),
            ),
            Err(other) => (
                CliResultOutput::error_or_blocked(
                    "error",
                    redact(&other.to_string(), Some(&api_key)),
                )
                .with_done_condition(done_condition)
                .with_counters(
                    agent.decider().requests_sent(),
                    agent.decider().images_sent(),
                ),
                Some(api_key),
            ),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (decider, policy, logger, task, resolved_start_url, dry_run);
        (
            CliResultOutput::error_or_blocked(
                "error",
                "decision-cu live execution requires macOS",
            )
            .with_done_condition(done_condition),
            Some(api_key),
        )
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            task,
            max_steps,
            start_url,
            start_url_preset,
            allow_url_prefix,
            preset,
            targets_file,
            done_when,
            fill,
            harness,
            no_self_fix,
            experience_dir,
            dry_run,
            json,
        } => {
            let (out, key_opt) = execute_run(
                &task,
                max_steps,
                start_url,
                start_url_preset,
                &allow_url_prefix,
                &preset,
                targets_file.as_deref(),
                done_when.as_deref(),
                &fill,
                harness.as_deref(),
                no_self_fix,
                &experience_dir,
                dry_run,
            );
            let code = out.exit_code();

            if json {
                if let Ok(serialized) = serde_json::to_string(&out) {
                    println!("{}", redact(&serialized, key_opt.as_ref()));
                }
            } else if code == 0 {
                println!(
                    "status={} steps={} reason=\"{}\"",
                    out.status, out.steps, out.reason
                );
            } else {
                eprintln!(
                    "status={} steps={} reason=\"{}\"",
                    out.status, out.steps, out.reason
                );
            }

            ExitCode::from(code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use decision_cu::decider::{
        events_preset_targets, events_zh_preset_targets, flights_preset_targets,
        generic_preset_targets,
    };

    #[test]
    fn cli_parses_task_and_goal_alias_and_new_flags() {
        let c1 = Cli::try_parse_from([
            "decision-cu",
            "run",
            "--task",
            "find flights",
            "--max-steps",
            "12",
            "--start-url",
            "https://www.google.com/travel/flights",
            "--allow-url-prefix",
            "https://techcrunch.com",
            "--allow-url-prefix",
            "https://*.bizzabo.com",
            "--preset",
            "events",
            "--targets-file",
            "/tmp/custom_targets.json",
            "--done-when",
            "checkout-page",
            "--fill",
            "origin field=Osaka",
            "--fill",
            "destination field=Taipei",
            "--harness",
            "/tmp/harness.json",
            "--no-self-fix",
            "--experience-dir",
            "/tmp/exp_dir",
            "--dry-run",
            "--json",
        ])
        .unwrap();
        match c1.command {
            Commands::Run {
                task,
                max_steps,
                start_url,
                start_url_preset,
                allow_url_prefix,
                preset,
                targets_file,
                done_when,
                fill,
                harness,
                no_self_fix,
                experience_dir,
                dry_run,
                json,
            } => {
                assert_eq!(task, "find flights");
                assert_eq!(max_steps, 12);
                assert_eq!(
                    start_url.as_deref(),
                    Some("https://www.google.com/travel/flights")
                );
                assert!(start_url_preset.is_none());
                assert_eq!(
                    allow_url_prefix,
                    vec![
                        "https://techcrunch.com".to_string(),
                        "https://*.bizzabo.com".to_string()
                    ]
                );
                assert_eq!(preset, "events");
                assert_eq!(
                    targets_file.as_deref(),
                    Some(std::path::Path::new("/tmp/custom_targets.json"))
                );
                assert_eq!(done_when.as_deref(), Some("checkout-page"));
                assert_eq!(fill.len(), 2);
                let parsed = parse_fills(&fill).unwrap();
                assert_eq!(parsed.get("origin field").unwrap(), "Osaka");
                assert_eq!(parsed.get("destination field").unwrap(), "Taipei");
                assert_eq!(
                    harness.as_deref(),
                    Some(std::path::Path::new("/tmp/harness.json"))
                );
                assert!(no_self_fix);
                assert_eq!(experience_dir, std::path::Path::new("/tmp/exp_dir"));
                assert!(dry_run);
                assert!(json);
            }
        }

        let c2 = Cli::try_parse_from([
            "decision-cu",
            "run",
            "--goal",
            "legacy alias",
            "--start-url-preset",
            "one-way",
        ])
        .unwrap();
        match c2.command {
            Commands::Run {
                task,
                max_steps,
                start_url,
                start_url_preset,
                allow_url_prefix,
                preset,
                targets_file,
                done_when,
                fill,
                harness,
                no_self_fix,
                experience_dir,
                dry_run,
                json,
            } => {
                assert_eq!(task, "legacy alias");
                assert_eq!(max_steps, 30);
                assert!(start_url.is_none());
                assert_eq!(start_url_preset.as_deref(), Some("one-way"));
                assert_eq!(
                    resolve_start_url_preset("one-way").unwrap(),
                    "https://www.google.com/travel/flights?hl=en&q=one%20way%20flights"
                );
                assert!(allow_url_prefix.is_empty());
                assert_eq!(preset, "flights");
                assert!(targets_file.is_none());
                assert!(done_when.is_none());
                assert!(fill.is_empty());
                assert!(harness.is_none());
                assert!(!no_self_fix);
                assert_eq!(experience_dir, std::path::Path::new(DEFAULT_EXPERIENCE_DIR));
                assert!(!dry_run);
                assert!(!json);
            }
        }

        // Mutually exclusive: --start-url and --start-url-preset cannot be combined
        let conflict = Cli::try_parse_from([
            "decision-cu",
            "run",
            "--task",
            "test",
            "--start-url",
            "https://www.google.com/travel/flights",
            "--start-url-preset",
            "one-way",
        ]);
        assert!(conflict.is_err());
    }

    #[test]
    fn preset_targets_and_targets_file_loading() {
        assert_eq!(Preset::from_name("flights").unwrap(), Preset::Flights);
        assert_eq!(Preset::from_name("events").unwrap(), Preset::Events);
        assert_eq!(Preset::from_name("events-zh").unwrap(), Preset::EventsZh);
        assert_eq!(Preset::from_name("generic").unwrap(), Preset::Generic);
        assert!(Preset::from_name("unknown").is_err());

        let flights = flights_preset_targets();
        assert_eq!(flights.len(), 10);

        let events = events_preset_targets();
        let event_labels: Vec<&str> = events.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(
            event_labels,
            vec![
                "events menu",
                "upcoming event card",
                "tickets link",
                "cheapest ticket option",
                "quantity increase",
                "checkout button",
                "popup close",
            ]
        );

        let events_zh = events_zh_preset_targets();
        let events_zh_labels: Vec<&str> = events_zh.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(
            events_zh_labels,
            vec![
                "搜尋或活動分類",
                "活動卡片",
                "立即報名按鈕",
                "票種選項",
                "數量加號",
                "下一步按鈕",
                "關閉彈窗",
            ]
        );
        assert_eq!(
            Preset::EventsZh.default_done_condition(),
            DoneCondition::CheckoutPage
        );
        assert_eq!(Preset::EventsZh.checkout_once_label(), "下一步按鈕");
        assert_eq!(Preset::EventsZh.popup_close_label(), "關閉彈窗");

        let generic = generic_preset_targets();
        let generic_labels: Vec<&str> = generic.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(
            generic_labels,
            vec![
                "navigation link toward the goal",
                "primary call-to-action button toward the goal",
                "popup close",
            ]
        );

        let temp_path = std::env::temp_dir().join("decision_cu_test_targets.json");
        std::fs::write(
            &temp_path,
            r#"[{"label":"custom btn","description":"a custom button"},{"label":"other link","description":null}]"#,
        )
        .unwrap();
        let loaded = load_targets_file(&temp_path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].label, "custom btn");
        assert_eq!(loaded[0].description.as_deref(), Some("a custom button"));
        assert_eq!(loaded[1].label, "other link");
        assert_eq!(loaded[1].description, None);
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn exit_codes_per_status() {
        let cases = [
            ("done", 0u8),
            ("error", 1u8),
            ("step_limit", 2u8),
            ("wait_limit", 3u8),
            ("blocked", 4u8),
        ];
        for (status, expected) in cases {
            let out = CliResultOutput::error_or_blocked(status, "test");
            assert_eq!(out.exit_code(), expected);
        }
    }
}
