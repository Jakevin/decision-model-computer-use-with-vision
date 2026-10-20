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
      --task <TASK>                Natural language task/goal to accomplish (alias: --goal) [alias: --goal]
      --max-steps <MAX_STEPS>      Maximum number of steps before stopping [default: 30]
      --start-url <URL>            Activate Chrome and navigate active tab to this URL before step 0
      --start-url-preset <PRESET>  Best-effort start URL preset (e.g. one-way), mutually exclusive with --start-url
      --allow-url-prefix <PREFIX>  Repeatable allowed URL prefix (https scheme, exact host or *.domain, path prefix)
      --preset <PRESET>            Target preset: flights, events, or generic [default: flights]
      --targets-file <PATH>        Path to a JSON file containing custom targets ([{"label":..,"description":..}])
      --done-when <CONDITION>      Done condition: flights-results, checkout-page, or model (defaults from --preset)
      --fill <KEY=VALUE>           Repeatable target-to-text fill mapping (e.g. --fill "origin field=Osaka")
      --dry-run                    Perform capture and decision making without executing mouse/keyboard actions
      --json                       Print a single final JSON result object to stdout ({status, steps, reason, ...})
  -h, --help                       Print help
```

### Example Invocations

```bash
# Live one-way flight search with start URL and field fills
cargo run --quiet -- run \
  --task "Search one-way flights from Osaka to Taipei on Nov 3, 2026" \
  --start-url "https://www.google.com/travel/flights" \
  --fill "origin field=Osaka" \
  --fill "destination field=Taipei" \
  --fill "departure date=Nov 3, 2026" \
  --max-steps 25 \
  --json

# Dry-run inspection (captures, crops, and queries Decision 3.0 without clicking or typing)
cargo run --quiet -- run --task "Check departure date selector" --max-steps 5 --dry-run --json

# Using --goal alias
cargo run --quiet -- run --goal "Sort flights by cheapest" --max-steps 10 --json
```

## How to Give a Task

- Provide a clear, read-only visual inspection or search instruction via `--task "<text>"` (or `--goal "<text>"`).
- Pass `--start-url "https://www.google.com/travel/flights"` to automatically activate Chrome and set the active tab URL before step 0 (never opens or closes tabs/windows).
- Pass repeatable `--fill "target=value"` flags for fields that should be automatically selected (`Meta+A`) and typed into right after clicking them.
- When calling from an automated agent, always pass `--json` so stdout contains a single machine-readable JSON line.

## Outputs & Exit Codes

When `--json` is passed, `decision-cu` prints exactly one JSON object to `stdout` including all run summary fields:

```json
{"status":"done","steps":8,"reason":"results page with prices","clicks":6,"requests_sent":14,"images_sent":14,"avg_seconds_per_step":1.42,"all_images_cropped":true,"tab_events":0}
```

| `status` | Exit Code | Meaning |
| :--- | :---: | :--- |
| `done` | `0` | Results page with flight prices reached (`results` noul >= 0.8 for 2 consecutive steps) or model returned `"done"` with `results` noul >= 0.5. |
| `error` | `1` | Missing `VSR_API_KEY`, HTTP/API transport error, or platform input/capture failure. |
| `step_limit` | `2` | Reached `--max-steps` before completion. |
| `wait_limit` | `3` | Exceeded `max_consecutive_waits` (`> 2` consecutive `"wait"` decisions). |
| `blocked` | `4` | Blocked by privacy or safety policy (`NoCrop`, `CropTooSmall`, `ClickInTabStrip`, `ClickOutsideWindow`, `ForbiddenTarget`, `SensitiveTextBlocked`, or `> 2` off-prefix `tab_events`). |

Detailed per-step logs are written to `./runs/<timestamp>/steps.jsonl` and the run summary is written to `./runs/<timestamp>/summary.json` with automatic secret redaction (`redact()`).

## Measured result (single run)

> **Note:** This is **ONE single live run**, not a benchmark.

- **Environment**: `2026-10-20` (`Asia/Tokyo`), commit `5515ddf`, model `vllm-sr/d3-turbo`, macOS on Mac mini, Chrome window `1200x916` pt.
- **Task**: Cheapest one-way Osaka (`KIX`) to Taipei (`TPE`) on Nov 3, 2026 on Google Flights (read-only).
- **Command**:
  ```bash
  decision-cu run --task "Find the cheapest one-way flight from Osaka (KIX) to Taipei (TPE) on Nov 3, 2026 on Google Flights" --start-url "https://www.google.com/travel/flights?hl=en" --fill "origin field=Osaka" --fill "destination field=Taipei" --fill "departure date=Nov 3, 2026" --max-steps 25 --json
  ```
- **Result**: `status: "done"` — reached the Google Flights results page (`KIX` to `TPE`, one way, Tue Nov 3) showing prices; cheapest shown at the time of the run was **¥20,470** (Jetstar, nonstop). Completed in `11` steps with `9` clicks on the intended targets (trip type, One way, origin, KIX suggestion, destination, TPE suggestion, departure date, Done, Search) plus `1` Return key press; `22` API requests / `22` images (every image a Chrome-window-only capture); average `4.77 s` per step; no tab opened or closed; no text typed outside Chrome. Earlier live attempts failed (other app windows covering Chrome, missing settle delays) and were fixed prior to this run.

## Safety Limits

- **Chrome-Window Crop Only**: If no `Google Chrome` window is found (`crop == None`) or the cropped area is `< 50x50` px, `decision-cu` stops immediately (`status: "blocked"`, exit code `4`) and sends **zero** bytes over the network.
- **Tab & Address Strip Guard**: Any click with `y < window.y + 80.0` points is rejected (`ClickInTabStrip`) so the agent cannot switch tabs, edit the URL bar, or click window close/minimize buttons.
- **URL Allowlist (`--allow-url-prefix`)**: Only `https://` URLs matching configured `--allow-url-prefix` entries (exact host or `*.domain` subdomain wildcard + path prefix) are permitted (`https://www.google.com/travel/flights` for `flights` preset, or derived from `--start-url`'s host when unspecified).
- **Read-Only, Sensitive-Target Gate & Checkout-Once**: Targets matching forbidden words (`"sign in"`, `"signin"`, `"log in"`, `"login"`, `"password"`, `"book"`, `"purchase"`, `"pay"`, `"place order"`, `"buy"`, `"credit card"`) are refused (`ForbiddenTarget`); `--fill` is refused for personal-info labels (`name`, `email`, `phone`, `card`, `password`); every click `box` request also asks a `"sensitive"` `noul` question and blocks the click (`blocked_sensitive`, excluding the target for 3 steps) if `sensitive >= 0.5`; and `"checkout button"` is clicked at most once per run (`checkout_once`).
- **Dismiss-Popup Gate**: In `events` and `generic` presets, action kind `"dismiss popup"` locates `"popup close"` and checks `noul` `"closeish"` (`"Is this button a Close, X, Reject, Decline or No-thanks control (not Accept, Subscribe or Sign up)?"`), clicking only when `closeish >= 0.6` and otherwise logging `dismiss_refused` and treating the step as `wait`.
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
