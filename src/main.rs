use clap::{Parser, Subcommand};
use decision_cu::agent::Agent;
use decision_cu::api::{ApiKey, DecisionClient};
use decision_cu::decider::{TargetSpec, VisionDecider};
use decision_cu::logging::RunLogger;
use decision_cu::safety::SafetyPolicy;

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
    /// Run the computer-use agent loop toward a specified goal
    Run {
        /// Natural language goal to accomplish
        #[arg(long)]
        goal: String,

        /// Maximum number of steps before stopping
        #[arg(long, default_value_t = 30)]
        max_steps: usize,

        /// Perform capture and decision making without executing mouse/keyboard actions
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}

pub fn flights_preset_targets() -> Vec<TargetSpec> {
    vec![
        TargetSpec::new(
            "origin field",
            Some("The departure city or airport text input field ('Where from?') on Google Flights".to_string()),
        ),
        TargetSpec::new(
            "destination field",
            Some("The arrival city or airport text input field ('Where to?') on Google Flights".to_string()),
        ),
        TargetSpec::new(
            "departure date",
            Some("The departure date selector input box on the Google Flights search form".to_string()),
        ),
        TargetSpec::new(
            "return date",
            Some("The return date selector input box on the Google Flights search form".to_string()),
        ),
        TargetSpec::new(
            "done button",
            Some("The blue 'Done' confirmation button inside the date picker calendar popover".to_string()),
        ),
        TargetSpec::new(
            "search button",
            Some("The pill-shaped blue 'Search' or 'Explore' button with a magnifying glass icon".to_string()),
        ),
        TargetSpec::new(
            "nonstop filter",
            Some("The 'Stops' filter button or 'Nonstop only' filter option in the flight results bar".to_string()),
        ),
        TargetSpec::new(
            "sort by cheapest",
            Some("The 'Cheapest' tab or sort option above the list of available flights".to_string()),
        ),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            goal,
            max_steps,
            dry_run,
        } => {
            let api_key = ApiKey::from_env()?;
            let client = DecisionClient::new(api_key.clone())?;
            let targets = flights_preset_targets();
            let decider = VisionDecider::new(client, targets);

            let mut policy = SafetyPolicy::default();
            policy.max_steps = max_steps;

            let logger = RunLogger::new_in_runs_dir(Some(api_key))?;

            #[cfg(target_os = "macos")]
            {
                use decision_cu::capture::MacScreen;
                use decision_cu::input::EnigoInput;

                let screen = MacScreen::new();
                let input = EnigoInput::new()?;
                let mut agent = Agent::new(screen, decider, input, policy)
                    .with_logger(logger)
                    .with_dry_run(dry_run);

                let summary = agent.run(&goal)?;
                println!(
                    "Run finished after {} steps: {:?}",
                    summary.steps_taken, summary.stop_reason
                );
            }

            #[cfg(not(target_os = "macos"))]
            {
                let _ = (decider, policy, logger, goal, dry_run);
                return Err("decision-cu live execution requires macOS".into());
            }
        }
    }

    Ok(())
}
