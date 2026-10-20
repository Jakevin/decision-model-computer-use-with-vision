# decision-model-computer-use-with-vision

A screenshot-only macOS computer-use agent written in Rust (`decision-cu`, edition 2021), driven by the **vllm-sr Decision 3.0 API** (`https://decision.vllm-sr.ai/v1/systemone`, model `vllm-sr/d3-turbo`).

---

## English

### What It Is

`decision-cu` is a typesafe, privacy-conscious, read-only macOS computer-use agent in Rust. Instead of relying on DOM scraping, browser extensions, or accessibility trees, it operates strictly on cropped screenshots of the active Google Chrome window and uses structured visual decision queries (`choice` and `box` questions) via the vllm-sr Decision 3.0 API to navigate and inspect web interfaces (such as Google Flights).

An agent skill definition is also included at [`skills/decision-computer-use/SKILL.md`](skills/decision-computer-use/SKILL.md) so AI coding and automation agents can invoke `decision-cu` with structured `--json` output and well-defined exit codes.

### How It Works

1. **Screenshot-Only Capture (`src/capture.rs` & `src/privacy.rs`)**:
   - Locates the active `Google Chrome` window on macOS via `xcap` and captures the primary monitor.
   - Crops strictly to the Chrome window bounds (accounting for the Retina display scale factor). If no Chrome window is found (`None` crop) or the crop is smaller than `50x50` pixels, the agent immediately stops and **sends nothing** over the network.
   - Downscales the cropped image if needed so that total resolution is `<= 1.6 MP`, encodes it as JPEG (`quality = 85`), and formats it as a `data:image/jpeg;base64,...` data URL.
2. **Two-Stage Structured Visual Decision (`src/decider.rs`)**:
   - **Stage 1 (`choice` questions)**: Sends a single request containing the current state (`goal` + last 8 history entries), the cropped screenshot, a `"kind"` choice question (`click`, `type`, `scroll`, `wait`, `done`), and a `"target"` choice question populated with configured target labels plus `"nothing to click"`.
   - **Stage 2 (`box` question)**: When the chosen action kind is `click` (and the target is not `"nothing to click"`), sends a second request with a `"box"` question to locate the bounding box of the chosen target label.
   - **Low-Confidence Re-Ask**: If the returned bounding box confidence is `< 0.2`, the decider automatically re-asks once using the target's longer descriptive prompt.
3. **Coordinate Mapping & Input Execution (`src/input.rs`)**:
   - Computes the centre of the fractional bounding box `((left + right) / 2, (top + bottom) / 2)`, maps it from cropped pixel coordinates back into macOS screen points using the window origin and display scale factor, validates all safety rules, and performs the action via `enigo` (explicitly releasing `Shift`, `Control`, `Alt`, and `Meta` modifiers before and after every action).

### macOS Installation & Permissions

1. **Install Rust Toolchain**:
   Install Rust (edition 2021 compatible) via [rustup](https://rustup.rs/):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"
   ```
2. **Grant macOS System Permissions**:
   Because `decision-cu` captures window screenshots and synthesizes mouse/keyboard events, you must grant two permissions to your terminal application (e.g., Terminal.app, iTerm2, Warp, or your IDE terminal) in **System Settings → Privacy & Security**:
   - **Screen & System Audio Recording** (required by `xcap` to read window bounds and capture the screen).
   - **Accessibility** (required by `enigo` to move/click the mouse and send keystrokes).

### `VSR_API_KEY` Environment Setup

The API key is read **exclusively** from the `VSR_API_KEY` environment variable:

```bash
export VSR_API_KEY="sk-vsr-your-secret-key-here"
```

- **Never commit your API key** to version control (`.env` is gitignored).
- Internally, the key is wrapped in an `ApiKey` newtype whose `Debug` and `Display` implementations always output `ApiKey(***)`.
- All transport errors, API errors, CLI outputs, and JSONL run logs in `./runs/` pass through secret sanitization (`redact()`) so neither `VSR_API_KEY` nor any `sk-vsr-[0-9A-Za-z]{43}` token is ever printed or written to disk.

### Running & Testing

Build and run the test suite:

```bash
cargo build
cargo test
```

Run the CLI agent (with Google Chrome open to Google Flights):

```bash
# Using --task (or --goal alias) in dry-run mode with structured JSON output
cargo run -- run --task "Find nonstop flights from SFO to JFK next Friday" --max-steps 15 --dry-run --json

# Live execution
cargo run -- run --task "Find nonstop flights from SFO to JFK next Friday" --max-steps 30 --json
```

#### CLI Flags & Exit Codes (`decision-cu run`)

- `--task <TASK>` (alias `--goal <TASK>`): Natural language task/goal to accomplish.
- `--max-steps <MAX_STEPS>`: Maximum number of steps before stopping (default `30`).
- `--dry-run`: Perform capture and decision making without executing mouse/keyboard actions.
- `--json`: Print a single final JSON result object to stdout: `{"status":"done|step_limit|wait_limit|blocked|error","steps":N,"reason":"..."}`.

Exit codes:
- `0` (`done`): Task completed.
- `1` (`error`): Missing `VSR_API_KEY`, transport/API failure, or platform error.
- `2` (`step_limit`): Reached `--max-steps`.
- `3` (`wait_limit`): Exceeded maximum consecutive waits (`> 2`).
- `4` (`blocked`): Blocked by privacy (`NoCrop` / `CropTooSmall`) or safety policy (`ClickInTabStrip`, `ForbiddenTarget`, etc.).

See [`skills/decision-computer-use/SKILL.md`](skills/decision-computer-use/SKILL.md) for full agent skill instructions.

### Safety Rules

Enforced by `SafetyPolicy` (`src/safety.rs`) and `Agent` (`src/agent.rs`):
- **Protected Tab & Address Bar Strip**: Any click with `y < window.y + 80.0` points is blocked with `SafetyError::ClickInTabStrip`, preventing the agent from switching tabs, typing arbitrary URLs in the address bar, or closing the browser window.
- **Window Bounds Check**: Clicks outside the Chrome window rectangle are rejected.
- **Allowed Hosts Prefix Check**: URL verification restricts navigation to allowed host prefixes (`www.google.com/travel/flights` by default).
- **Read-Only & Forbidden Target Filter**: Refuses to interact with targets containing sensitive or transactional words (`"sign in"`, `"login"`, `"password"`, `"book"`, `"purchase"`, `"pay"`, `"checkout"`, `"buy"`, `"credit card"`), and never types passwords or secrets.
- **Loop Termination Limits**: Automatically halts when `"done"` is returned, when `max_steps` (default `30`) is reached, or after `> 2` consecutive `"wait"` actions.

### Privacy Note

- Full-screen captures are never transmitted: if the target Chrome window cannot be found (`crop == None`), `prepare_image` returns `Err(PrivacyError::NoCrop)` and **zero network requests are made**.
- Local audit records (`AuditLog`) store only crop origin, crop pixel dimensions, downscaled dimensions, and step purpose—never image bytes.
- **External API Notice**: Cropped Chrome-window screenshots are sent to `https://decision.vllm-sr.ai` (`vllm-sr/d3-turbo`) for visual decision inference and may be logged or processed by the service per their documentation (`https://decision.vllm-sr.ai/docs`). Keep unrelated sensitive information out of the visible Chrome window while running the agent.

### Limitations

- **macOS Only**: Window discovery (`xcap`) and input synthesis (`enigo`) are configured specifically for macOS desktop environments (`cfg(target_os = "macos")`).
- **Single-Window Crop**: Expects a visible, non-minimized `Google Chrome` window on the primary monitor.
- **Preset Vocabulary**: The `choice` target question selects from the configured `TargetSpec` list (defaulting to the Google Flights preset in `src/main.rs`). Tasks on other allowed web pages require supplying matching `TargetSpec` labels.

### Credits

Design inspired by `awlevin/typesafe-computer-use` (no code copied).

---

## 繁體中文 (Traditional Chinese)

### 專案簡介 (What It Is)

`decision-cu` 是一個以 Rust（2021 Edition）從零編寫、僅依賴螢幕截圖（screenshot-only）的 macOS 電腦操作代理程式（Computer-Use Agent），由 **vllm-sr Decision 3.0 API**（`https://decision.vllm-sr.ai/v1/systemone`，模型 `vllm-sr/d3-turbo`）驅動。本專案不依賴 DOM 解析、瀏覽器外掛或無障礙樹（Accessibility Tree），而是直接擷取並裁切 Google Chrome 視窗畫面，透過型別安全的結構化視覺問答（`choice` 選擇題與 `box` 邊界框定位題）完成網頁操作。

本倉庫同時提供 [`skills/decision-computer-use/SKILL.md`](skills/decision-computer-use/SKILL.md) 技能定義檔，方便 AI 代理程式透過 `--task` 與 `--json` 旗標自動化呼叫並解析結果與狀態碼。

### 運作原理 (How It Works)

1. **僅限截圖與強制視窗裁切 (`src/capture.rs` & `src/privacy.rs`)**：
   - 使用 `xcap` 尋找 macOS 上名稱包含 `Google Chrome` 的視窗並擷取主螢幕畫面（自動處理 Retina 縮放比例 `scale_factor`）。
   - 若找不到 Chrome 視窗（`crop` 為 `None`）或裁切區域小於 `50x50` 像素，立即回傳錯誤並中止執行，**絕不傳送任何全螢幕截圖或網路請求**。
   - 影像若超過 `1.6 MP` 會自動等比例縮小至 `<= 1.6 MP`，並壓縮為品質 `85` 的 JPEG `data:image/jpeg;base64,...` 字串。
2. **兩階段 Decision 3.0 視覺決策 (`src/decider.rs`)**：
   - **第一階段（`choice` 問題）**：發送包含目標狀態（`goal` 與最近 8 筆歷史紀錄）與裁切後截圖的請求，同時詢問 `"kind"`（`click`、`type`、`scroll`、`wait`、`done`）與 `"target"`（預設目標清單加上 `"nothing to click"`）。
   - **第二階段（`box` 問題）**：當動作為 `click` 時，針對所選目標發送第二次請求（`box` 問題）以取得該元件在畫面中的相對比例邊界框（`top`、`left`、`bottom`、`right`）。
   - **低信心度重問機制**：若第一次 `box` 回傳的 `confidence < 0.2`，會自動改用該目標的詳細描述（longer description）重新詢問一次。
3. **座標轉換與點擊執行 (`src/input.rs`)**：
   - 計算邊界框中心點並由裁切像素座標還原為 macOS 螢幕點座標（screen points），通過安全檢查後透過 `enigo` 執行點擊或輸入（每次操作前後皆會強制釋放 `Shift`、`Control`、`Alt`、`Meta` 等修飾鍵）。

### macOS 安裝與系統權限設定

1. **安裝 Rust 工具鏈**：
   透過 [rustup](https://rustup.rs/) 安裝 Rust：
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"
   ```
2. **授予 macOS 終端機權限**：
   請至 **系統設定 (System Settings) → 隱私權與安全性 (Privacy & Security)**，為您使用的終端機應用程式開啟以下兩項權限：
   - **螢幕與系統音訊錄製 (Screen & System Audio Recording)**：供 `xcap` 讀取視窗位置與擷取畫面。
   - **輔助使用 (Accessibility)**：供 `enigo` 控制滑鼠點擊與鍵盤輸入。

### `VSR_API_KEY` 環境變數設定

API 金鑰**僅**從環境變數 `VSR_API_KEY` 讀取：

```bash
export VSR_API_KEY="sk-vsr-your-secret-key-here"
```

- **切勿將金鑰提交至 Git**（`.env` 與 `runs/` 已列入 `.gitignore`）。
- 金鑰封裝於 `ApiKey` 新型別中，其 `Debug` 與 `Display` 輸出永遠為 `ApiKey(***)`。
- 所有錯誤訊息、CLI 輸出與 `./runs/` 下的 JSONL 執行紀錄皆經過 `redact()` 過濾，自動將目前金鑰與任何符合 `sk-vsr-[0-9A-Za-z]{43}` 格式的字串替換為 `"***"`。

### 執行與測試

編譯與執行完整單元/整合測試：

```bash
cargo build
cargo test
```

執行 CLI（請先開啟 Google Chrome 並停留在 Google Flights 頁面）：

```bash
# 乾跑模式（僅截圖與詢問模型決策，不實際移動滑鼠或按鍵），並輸出單行 JSON 結果
cargo run -- run --task "Find nonstop flights from SFO to JFK next Friday" --max-steps 15 --dry-run --json

# 實際執行模式（亦可使用 --goal 別名）
cargo run -- run --task "Find nonstop flights from SFO to JFK next Friday" --max-steps 30 --json
```

#### CLI 參數與結束狀態碼 (Exit Codes)

- `--task <TASK>`（別名 `--goal <TASK>`）：指定自然語言任務目標。
- `--max-steps <MAX_STEPS>`：最大步數上限（預設 `30`）。
- `--dry-run`：乾跑模式，不執行實際滑鼠或鍵盤動作。
- `--json`：於標準輸出印出單一 JSON 結果物件：`{"status":"done|step_limit|wait_limit|blocked|error","steps":N,"reason":"..."}`。
- 結束狀態碼：`0` (`done`)、`1` (`error`)、`2` (`step_limit`)、`3` (`wait_limit`)、`4` (`blocked`)。完整說明請參考 [`skills/decision-computer-use/SKILL.md`](skills/decision-computer-use/SKILL.md)。

### 安全規則 (Safety Rules)

- **分頁列與網址列保護（Tab Strip Protection）**：任何位於 `y < window.y + 80.0` 點座標的點擊都會被 `ClickInTabStrip` 攔截，防止代理程式切換分頁、修改網址列或關閉視窗。
- **允許網域前綴檢查**：預設僅允許 `www.google.com/travel/flights`。
- **唯讀與敏感目標攔截**：禁止點擊包含 `"sign in"`、`"book"`、`"purchase"`、`"pay"`、`"checkout"` 等字詞的目標，且絕不輸入密碼或金鑰。
- **自動停止條件**：當回傳 `"done"`、達到步數上限（預設 30 步）或連續等待（`wait`）超過 2 次時自動停止。

### 隱私聲明 (Privacy Note)

- 本工具僅擷取並傳送裁切後的 Chrome 視窗畫面；若未偵測到 Chrome 視窗則不會發送任何請求。本地端的 `AuditLog` 僅記錄裁切座標與尺寸，不保存任何影像位元組。
- **外部服務提示**：裁切後的 Chrome 視窗截圖會傳送至 `https://decision.vllm-sr.ai` 進行視覺推論，並可能依據其官方文件政策被伺服器端記錄。執行期間請勿在 Chrome 視窗內顯示無關的個人敏感資訊。

### 限制 (Limitations)

- 僅支援 macOS 桌面環境（`cfg(target_os = "macos")`）。
- 需確保 Google Chrome 視窗顯示於主螢幕且未最小化。
- 預設目標清單針對 Google Flights 設計，若用於其他頁面需調整對應的 `TargetSpec` 定義。

### 致謝 (Credits)

Design inspired by `awlevin/typesafe-computer-use` (no code copied).
