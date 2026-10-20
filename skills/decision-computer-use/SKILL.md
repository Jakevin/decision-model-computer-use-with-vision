---
name: decision-computer-use
description: Use this skill when an AI agent or user wants to drive a macOS Google Chrome browser window via screenshot-only visual decisions (vllm-sr Decision 3.0 API) using the `decision-cu` Rust CLI, such as searching or inspecting flights on Google Flights with strict read-only and window-crop safety guarantees.
---

# decision-computer-use

## What It Does

`decision-cu` is a screenshot-only macOS computer-use agent written in Rust and powered by the **vllm-sr Decision 3.0 API** (`https://decision.vllm-sr.ai/v1/systemone`, model `vllm-sr/d3-turbo`).

At each step, `decision-cu`:
1. Finds the active, non-minimized `Google Chrome` window on the primary macOS display (`xcap`) and crops the screenshot strictly to that window (downscaled to `<= 1.6 MP`, JPEG quality `85`).
2. Queries the Decision 3.0 API with two `choice` questions (`"kind"`: `click | type | scroll | wait | done`, and `"target"`: preset UI targets + `"nothing to click"`).
3. When `"kind"` is `click`, sends a second `box` question for the selected target label (automatically re-asking once with a longer description if `confidence < 0.2`) and clicks the centre of the bounding box mapped to macOS screen points (`enigo`).

## Prerequisites

1. **macOS Permissions** (in **System Settings → Privacy & Security**, granted to the terminal or parent runner process):
   - **Screen & System Audio Recording**: required to locate the Chrome window bounds and capture its pixels.
   - **Accessibility**: required to move/click the mouse and synthesize keyboard events.
2. **Google Chrome**:
   - Must be open, visible (not minimized), and positioned on the primary monitor at an allowed URL (by default `https://www.google.com/travel/flights`).
3. **`VSR_API_KEY` Environment Variable**:
   - Must be set in the environment before invoking `decision-cu`. Never pass the key on the command line or write it to tracked files:
     ```bash
     export VSR_API_KEY="sk-vsr-..."
     ```

## Verified CLI Commands & Flags

Verified against `cargo run -- run --help`:

```text
Run the computer-use agent loop toward a specified task/goal

Usage: decision-cu run [OPTIONS] --task <TASK>

Options:
      --task <TASK>            Natural language task/goal to accomplish (alias: --goal) [alias: --goal]
      --max-steps <MAX_STEPS>  Maximum number of steps before stopping [default: 30]
      --dry-run                Perform capture and decision making without executing mouse/keyboard actions
      --json                   Print a single final JSON result object to stdout ({status, steps, reason})
  -h, --help                   Print help
```

### Example Invocations

```bash
# Recommended for AI agents: structured JSON output on stdout
cargo run --quiet -- run --task "Search nonstop flights from SFO to JFK" --max-steps 20 --json

# Dry-run inspection (captures, crops, and queries Decision 3.0 without clicking or typing)
cargo run --quiet -- run --task "Check departure date selector" --max-steps 5 --dry-run --json

# Using --goal alias
cargo run --quiet -- run --goal "Sort flights by cheapest" --max-steps 10 --json
```

## How to Give a Task

- Provide a clear, read-only visual inspection or search instruction via `--task "<text>"` (or `--goal "<text>"`).
- Keep the active Google Chrome window focused on the target page (e.g., `https://www.google.com/travel/flights`).
- When calling from an automated agent, always pass `--json` so stdout contains a single machine-readable JSON line.

## Outputs & Exit Codes

When `--json` is passed, `decision-cu` prints exactly one JSON object to `stdout`:

```json
{"status":"done","steps":4,"reason":"Task completed ('done' returned by model)"}
```

| `status` | Exit Code | Meaning |
| :--- | :---: | :--- |
| `done` | `0` | The model returned `"done"` and the task finished normally. |
| ` error` | `1` | Missing `VSR_API_KEY`, HTTP/API transport error, or platform input/capture failure. |
| `step_limit` | `2` | Reached `--max-steps` before the model returned `"done"`. |
| `wait_limit` | `3` | Exceeded `max_consecutive_waits` (`> 2` consecutive `"wait"` decisions). |
| `blocked` | `4` | Blocked by privacy or safety policy (`NoCrop`, `CropTooSmall`, `ClickInTabStrip`, `ClickOutsideWindow`, `ForbiddenTarget`, or `SensitiveTextBlocked`). |

Detailed per-step JSONL logs are written to `./runs/run-<timestamp>.jsonl` with automatic secret redaction (`redact()`).

## Safety Limits

- **Chrome-Window Crop Only**: If no `Google Chrome` window is found (`crop == None`) or the cropped area is `< 50x50` px, `decision-cu` stops immediately (`status: "blocked"`, exit code `4`) and sends **zero** bytes over the network.
- **Tab & Address Strip Guard**: Any click with `y < window.y + 80.0` points is rejected (`ClickInTabStrip`) so the agent cannot switch tabs, edit the URL bar, or click window close/minimize buttons.
- **Read-Only & No Sign-In / Purchase**: Targets matching forbidden words (`"sign in"`, `"signin"`, `"log in"`, `"login"`, `"password"`, `"book"`, `"purchase"`, `"pay"`, `"checkout"`, `"buy"`, `"credit card"`) are refused (`ForbiddenTarget`), and password/secret typing is blocked.
- **Modifier Hygiene**: `Shift`, `Control`, `Alt`, and `Meta` keys are explicitly released before and after every mouse or keyboard action.

## Privacy Note

- Only the cropped `Google Chrome` window image (downscaled to `<= 1.6 MP` JPEG q85) is transmitted to `https://decision.vllm-sr.ai/v1/systemone`, where requests may be logged per the provider's documentation (`https://decision.vllm-sr.ai/docs`).
- Local audit records (`AuditLog`) store only crop coordinates, dimensions, and purpose strings—never image bytes.
- `VSR_API_KEY` and any `sk-vsr-[0-9A-Za-z]{43}` token are scrubbed from `Debug`/`Display` formatting, error strings, stdout, and `./runs/*.jsonl` logs.

## Troubleshooting

- **`status: "blocked"` with `"No active Google Chrome window crop available"`**:
  - Ensure `Google Chrome` is open, not minimized, on the primary screen, and that your terminal has **Screen Recording** permission in macOS System Settings.
- **`status: "blocked"` with `"ClickInTabStrip"`**:
  - The model attempted to click within the top 80 points of the Chrome window (tab bar / address bar). Navigate to the target page manually before starting the run.
- **`status: "error"` with `"VSR_API_KEY environment variable is missing or empty"`**:
  - Export `VSR_API_KEY` in the shell environment before running `decision-cu`.
- **Clicks or keystrokes do not take effect**:
  - Grant **Accessibility** permission to the terminal in **System Settings → Privacy & Security → Accessibility** and restart the terminal.
