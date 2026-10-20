use clap::{Parser, Subcommand};
use decision_cu::agent::{Agent, AgentError, StopReason};
use decision_cu::api::{ApiKey, DecisionClient};
use decision_cu::decider::{TargetSpec, VisionDecider};
use decision_cu::logging::{redact, RunLogger};
use decision_cu::safety::SafetyPolicy;
use serde::{Deserialize, Serialize};
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

        /// Perform capture and decision making without executing mouse/keyboard actions
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Print a single final JSON result object to stdout ({status, steps, reason})
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliResultOutput {
    pub status: String,
    pub steps: usize,
    pub reason: String,
}

impl CliResultOutput {
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

pub fn flights_preset_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec::new(
            "origin field",
            Some(
                "The departure city or airport text input field ('Where from?') on Google Flights"
                    .to_string(),
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
            "departure date",
            Some(
                "The departure date selector input box on the Google Flights search form"
                    .to_string(),
            ),
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
    dry_run: bool,
) -> (CliResultOutput, Option<ApiKey>) {
    let api_key = match ApiKey::from_env() {
        Ok(k) => k,
        Err(e) => {
            return (
                CliResultOutput {
                    status: "error".to_string(),
                    steps: 0,
                    reason: e.to_string(),
                },
                None,
            );
        }
    };

    let client = match DecisionClient::new(api_key.clone()) {
        Ok(c) => c,
        Err(e) => {
            return (
                CliResultOutput {
                    status: "error".to_string(),
                    steps: 0,
                    reason: redact(&e.to_string(), Some(&api_key)),
                },
                Some(api_key),
            );
        }
    };

    let targets = flights_preset_targets();
    let decider = VisionDecider::new(client, targets);

    let mut policy = SafetyPolicy::default();
    policy.max_steps = max_steps;

    let logger = match RunLogger::new_in_runs_dir(Some(api_key.clone())) {
        Ok(l) => l,
        Err(e) => {
            return (
                CliResultOutput {
                    status: "error".to_string(),
                    steps: 0,
                    reason: redact(&e.to_string(), Some(&api_key)),
                },
                Some(api_key),
            );
        }
    };

    #[cfg(target_os = "macos")]
    {
        use decision_cu::capture::MacScreen;
        use decision_cu::input::EnigoInput;

        let screen = MacScreen::new();
        let input = match EnigoInput::new() {
            Ok(i) => i,
            Err(e) => {
                return (
                    CliResultOutput {
                        status: "error".to_string(),
                        steps: 0,
                        reason: redact(&e.to_string(), Some(&api_key)),
                    },
                    Some(api_key),
                );
            }
        };

        let mut agent = Agent::new(screen, decider, input, policy)
            .with_logger(logger)
            .with_dry_run(dry_run);

        match agent.run(task) {
            Ok(summary) => {
                let (status, reason) = match summary.stop_reason {
                    StopReason::Done => (
                        "done".to_string(),
                        "Task completed ('done' returned by model)".to_string(),
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
                    StopReason::SafetyBlocked(msg) => ("blocked".to_string(), msg),
                };
                (
                    CliResultOutput {
                        status,
                        steps: summary.steps_taken,
                        reason: redact(&reason, Some(&api_key)),
                    },
                    Some(api_key),
                )
            }
            Err(AgentError::Safety(s)) => (
                CliResultOutput {
                    status: "blocked".to_string(),
                    steps: 0,
                    reason: redact(&s.to_string(), Some(&api_key)),
                },
                Some(api_key),
            ),
            Err(AgentError::Privacy(p)) => (
                CliResultOutput {
                    status: "blocked".to_string(),
                    steps: 0,
                    reason: redact(&p.to_string(), Some(&api_key)),
                },
                Some(api_key),
            ),
            Err(other) => (
                CliResultOutput {
                    status: "error".to_string(),
                    steps: 0,
                    reason: redact(&other.to_string(), Some(&api_key)),
                },
                Some(api_key),
            ),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (decider, policy, logger, task, dry_run);
        (
            CliResultOutput {
                status: "error".to_string(),
                steps: 0,
                reason: "decision-cu live execution requires macOS".to_string(),
            },
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
            dry_run,
            json,
        } => {
            let (out, key_opt) = execute_run(&task, max_steps, dry_run);
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
    fn cli_parses_task_and_goal_alias_and_json_flag() {
        let c1 = Cli::try_parse_from([
            "decision-cu",
            "run",
            "--task",
            "find flights",
            "--max-steps",
            "12",
            "--dry-run",
            "--json",
        ])
        .unwrap();
        match c1.command {
            Commands::Run {
                task,
                max_steps,
                dry_run,
                json,
            } => {
                assert_eq!(task, "find flights");
                assert_eq!(max_steps, 12);
                assert!(dry_run);
                assert!(json);
            }
        }

        let c2 = Cli::try_parse_from(["decision-cu", "run", "--goal", "legacy alias"]).unwrap();
        match c2.command {
            Commands::Run {
                task,
                max_steps,
                dry_run,
                json,
            } => {
                assert_eq!(task, "legacy alias");
                assert_eq!(max_steps, 30);
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
            let out = CliResultOutput {
                status: status.to_string(),
                steps: 3,
                reason: "test".to_string(),
            };
            assert_eq!(out.exit_code(), expected);
        }
    }
}
