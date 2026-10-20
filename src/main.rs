use clap::{Parser, Subcommand};
use decision_cu::agent::{Agent, AgentError, Timing};
use decision_cu::api::{ApiKey, DecisionClient};
use decision_cu::decider::{TargetSpec, VisionDecider};
use decision_cu::logging::{redact, RunLogger};
use decision_cu::safety::SafetyPolicy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
        #[arg(long = "start-url", value_name = "URL")]
        start_url: Option<String>,

        /// Repeatable target-to-text fill mapping (e.g. --fill "origin field=Osaka")
        #[arg(long = "fill", value_name = "KEY=VALUE")]
        fill: Vec<String>,

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
    pub clicks: usize,
    pub requests_sent: usize,
    pub images_sent: usize,
    pub avg_seconds_per_step: f64,
    pub all_images_cropped: bool,
    pub tab_events: usize,
}

impl CliResultOutput {
    pub fn error_or_blocked(status: &str, reason: impl Into<String>) -> Self {
        Self {
            status: status.to_string(),
            steps: 0,
            reason: reason.into(),
            clicks: 0,
            requests_sent: 0,
            images_sent: 0,
            avg_seconds_per_step: 0.0,
            all_images_cropped: true,
            tab_events: 0,
        }
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
        map.insert(key.to_string(), v.to_string());
    }
    Ok(map)
}

pub fn flights_preset_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec::new(
            "trip type dropdown",
            Some("'Round trip' dropdown at top-left of search form".to_string()),
        ),
        TargetSpec::new(
            "one way option",
            Some("'One way' item in that open dropdown".to_string()),
        ),
        TargetSpec::new(
            "origin field",
            Some(
                "The departure city or airport text input field ('Where from?') on Google Flights"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "origin suggestion",
            Some(
                "First airport/city suggestion row in the list under the from field".to_string(),
            ),
        ),
        TargetSpec::new(
            "destination field",
            Some(
                "The arrival city or airport text input field ('Where to?') on Google Flights"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "destination suggestion",
            Some("First airport/city suggestion row in the list under the to field".to_string()),
        ),
        TargetSpec::new(
            "departure date",
            Some(
                "The departure date selector input box on the Google Flights search form"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "departure date option",
            Some("The day cell for the requested date in the open calendar".to_string()),
        ),
        TargetSpec::new(
            "return date",
            Some(
                "The return date selector input box on the Google Flights search form".to_string(),
            ),
        ),
        TargetSpec::new(
            "done button",
            Some(
                "The blue 'Done' confirmation button inside the date picker calendar popover"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "search button",
            Some(
                "The pill-shaped blue 'Search' or 'Explore' button with a magnifying glass icon"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "nonstop filter",
            Some(
                "The 'Stops' filter button or 'Nonstop only' filter option in the flight results bar"
                    .to_string(),
            ),
        ),
        TargetSpec::new(
            "sort by cheapest",
            Some(
                "The 'Cheapest' tab or sort option above the list of available flights".to_string(),
            ),
        ),
    ]
}

fn execute_run(
    task: &str,
    max_steps: usize,
    start_url: Option<String>,
    fills: &[String],
    dry_run: bool,
) -> (CliResultOutput, Option<ApiKey>) {
    let type_values = match parse_fills(fills) {
        Ok(m) => m,
        Err(e) => return (CliResultOutput::error_or_blocked("error", e), None),
    };

    let api_key = match ApiKey::from_env() {
        Ok(k) => k,
        Err(e) => {
            return (
                CliResultOutput::error_or_blocked("error", e.to_string()),
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
                ),
                Some(api_key),
            );
        }
    };

    let targets = flights_preset_targets();
    let decider = VisionDecider::new(client, targets).with_type_values(type_values);

    let mut policy = SafetyPolicy::default();
    policy.max_steps = max_steps;

    let logger = match RunLogger::new_in_runs_dir(Some(api_key.clone())) {
        Ok(l) => l,
        Err(e) => {
            return (
                CliResultOutput::error_or_blocked(
                    "error",
                    redact(&e.to_string(), Some(&api_key)),
                ),
                Some(api_key),
            );
        }
    };

    #[cfg(target_os = "macos")]
    {
        use decision_cu::browser::MacBrowser;
        use decision_cu::capture::MacScreen;
        use decision_cu::input::EnigoInput;

        let screen = MacScreen::new();
        let input = match EnigoInput::new() {
            Ok(i) => i,
            Err(e) => {
                return (
                    CliResultOutput::error_or_blocked(
                        "error",
                        redact(&e.to_string(), Some(&api_key)),
                    ),
                    Some(api_key),
                );
            }
        };

        let mut agent = Agent::new(screen, decider, input, policy)
            .with_browser(MacBrowser::new())
            .with_timing(Timing::default())
            .with_start_url(start_url)
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
                        clicks: s.clicks,
                        requests_sent: s.requests_sent,
                        images_sent: s.images_sent,
                        avg_seconds_per_step: s.avg_seconds_per_step,
                        all_images_cropped: s.all_images_cropped,
                        tab_events: s.tab_events,
                    },
                    Some(api_key),
                )
            }
            Err(AgentError::Safety(s)) => (
                CliResultOutput::error_or_blocked(
                    "blocked",
                    redact(&s.to_string(), Some(&api_key)),
                ),
                Some(api_key),
            ),
            Err(AgentError::Privacy(p)) => (
                CliResultOutput::error_or_blocked(
                    "blocked",
                    redact(&p.to_string(), Some(&api_key)),
                ),
                Some(api_key),
            ),
            Err(other) => (
                CliResultOutput::error_or_blocked(
                    "error",
                    redact(&other.to_string(), Some(&api_key)),
                ),
                Some(api_key),
            ),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (decider, policy, logger, task, start_url, dry_run);
        (
            CliResultOutput::error_or_blocked(
                "error",
                "decision-cu live execution requires macOS",
            ),
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
            fill,
            dry_run,
            json,
        } => {
            let (out, key_opt) = execute_run(&task, max_steps, start_url, &fill, dry_run);
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
            "--fill",
            "origin field=Osaka",
            "--fill",
            "destination field=Taipei",
            "--dry-run",
            "--json",
        ])
        .unwrap();
        match c1.command {
            Commands::Run {
                task,
                max_steps,
                start_url,
                fill,
                dry_run,
                json,
            } => {
                assert_eq!(task, "find flights");
                assert_eq!(max_steps, 12);
                assert_eq!(
                    start_url.as_deref(),
                    Some("https://www.google.com/travel/flights")
                );
                assert_eq!(fill.len(), 2);
                let parsed = parse_fills(&fill).unwrap();
                assert_eq!(parsed.get("origin field").unwrap(), "Osaka");
                assert_eq!(parsed.get("destination field").unwrap(), "Taipei");
                assert!(dry_run);
                assert!(json);
            }
        }

        let c2 = Cli::try_parse_from(["decision-cu", "run", "--goal", "legacy alias"]).unwrap();
        match c2.command {
            Commands::Run {
                task,
                max_steps,
                start_url,
                fill,
                dry_run,
                json,
            } => {
                assert_eq!(task, "legacy alias");
                assert_eq!(max_steps, 30);
                assert!(start_url.is_none());
                assert!(fill.is_empty());
                assert!(!dry_run);
                assert!(!json);
            }
        }
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
