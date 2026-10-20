use decision_cu::browser::{activate_chrome, frontmost_app};
use decision_cu::capture::{changed_fraction, is_changed, topmost_app_at, Screen};
use decision_cu::input::Input;
use serde::Serialize;
use std::env;
use std::process::ExitCode;
use std::thread;
use std::time::Duration;

#[derive(Serialize)]
struct ProbeOutput {
    changed_fraction: f64,
    is_changed: bool,
    frontmost: Option<String>,
    topmost: Option<String>,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <x> <y> <out.png>", args[0]);
        return ExitCode::from(1);
    }

    let x: f64 = match args[1].parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Invalid x coordinate '{}': {}", args[1], e);
            return ExitCode::from(1);
        }
    };
    let y: f64 = match args[2].parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Invalid y coordinate '{}': {}", args[2], e);
            return ExitCode::from(1);
        }
    };
    let out_path = &args[3];

    let _ = activate_chrome();
    let topmost = topmost_app_at(x, y);
    if topmost.as_deref() != Some("Google Chrome") {
        let out = ProbeOutput {
            changed_fraction: 0.0,
            is_changed: false,
            frontmost: frontmost_app(),
            topmost,
        };
        if let Ok(json) = serde_json::to_string(&out) {
            println!("{}", json);
        }
        return ExitCode::from(2);
    }

    #[cfg(target_os = "macos")]
    {
        use decision_cu::capture::MacScreen;
        use decision_cu::input::QuartzInput;

        let screen = MacScreen::new();
        let before = match screen.capture() {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Capture before failed: {}", e);
                return ExitCode::from(1);
            }
        };

        let mut input = match QuartzInput::new() {
            Ok(i) => i,
            Err(e) => {
                eprintln!("Input init failed: {}", e);
                return ExitCode::from(1);
            }
        };

        if let Err(e) = input.click(x, y) {
            eprintln!("Click failed: {}", e);
            return ExitCode::from(1);
        }

        thread::sleep(Duration::from_millis(1500));

        let after = match screen.capture() {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Capture after failed: {}", e);
                return ExitCode::from(1);
            }
        };

        if let Err(e) = after.save(out_path) {
            eprintln!("Failed to save '{}': {}", out_path, e);
            return ExitCode::from(1);
        }

        let frac = changed_fraction(&before, &after);
        let changed = is_changed(&before, &after);
        let out = ProbeOutput {
            changed_fraction: frac,
            is_changed: changed,
            frontmost: frontmost_app(),
            topmost: topmost_app_at(x, y),
        };
        if let Ok(json) = serde_json::to_string(&out) {
            println!("{}", json);
        }
        ExitCode::from(0)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (
            out_path,
            changed_fraction,
            is_changed,
            Duration::from_millis(1500),
            thread::sleep,
            Input::click,
        );
        ExitCode::from(1)
    }
}
