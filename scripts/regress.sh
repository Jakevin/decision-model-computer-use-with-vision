#!/usr/bin/env bash
# scripts/regress.sh — Offline-defined regression harness for decision-cu (Task f10).
#
# Runs Task A (Google Flights KIX -> TPE, one-way, 2026-11-03, cheapest) and
# Task B (Accupass events-zh FREE event registration flow up to checkout-page)
# 3x each WITH --no-self-fix (baseline: legacy low-confidence re-ask and page-turn
# scroll retry still run; only new self-correction paths — rewritten description
# re-ask, experience library read/write, zoom, and OCR candidates — are disabled)
# and 3x each WITHOUT --no-self-fix (self-fix + experience),
# then summarizes the summary.json fields across all runs.
#
# ============================================================================
# SAFETY RULES FOR TASK B (ENFORCED VIA EXISTING SAFETY + FLAGS — DO NOT LOOSEN)
# - Preset: --preset events-zh
# - Start URL: --start-url "https://www.accupass.com/search"
# - Allowed URL prefix: --allow-url-prefix "https://www.accupass.com/"
# - Stop condition: --done-when checkout-page (registration form / confirm page
#   visible => stop immediately).
# - Paid events (any price > 0) MUST be skipped; only select a FREE event (免費).
# - NEVER click the final 「送出報名」/「確認報名」/「確認送出」/「完成報名」/Submit/Confirm
#   button; NEVER pay; NEVER log in.
# - Existing safety invariants (80pt tab-strip guard, forbidden target words,
#   checkout-once on 「下一步按鈕」, sensitive_noul >= 0.5 gate, login form/URL
#   detection, privacy window crop) remain strictly enforced.
# ============================================================================

set -euo pipefail

# Build once and call the binary directly. (`--quiet` used to be passed as
# `cargo run --quiet --`; it is a cargo flag, decision-cu itself has no --quiet.)
if [[ -z "${DECISION_CU_BIN:-}" ]]; then
  cargo build --release >&2
  DECISION_CU_BIN="./target/release/decision-cu"
fi
BIN="${DECISION_CU_BIN}"
REGRESS_OUT_DIR="${REGRESS_OUT_DIR:-./runs/regress_$(date +%Y%m%d_%H%M%S)}"
EXPERIENCE_DIR="${EXPERIENCE_DIR:-./experience}"
HARNESS_ARGS=()
if [[ -n "${HARNESS_FILE:-}" ]]; then
  HARNESS_ARGS=(--harness "${HARNESS_FILE}")
fi

mkdir -p "${REGRESS_OUT_DIR}"
SUMMARY_TSV="${REGRESS_OUT_DIR}/regress_records.tsv"
printf "task\tmode\trun_idx\trun_dir\tstatus\treason\tdone_condition\tsteps\tclicks\trequests_sent\timages_sent\tavg_seconds_per_step\tall_images_cropped\ttab_events\n" > "${SUMMARY_TSV}"

latest_run_dir() {
  ls -1dt ./runs/run_* 2>/dev/null | head -n 1 || true
}

record_summary() {
  local task_name="$1"
  local mode_name="$2"
  local idx="$3"
  local rdir="$4"
  local sjson="${rdir}/summary.json"
  if [[ -f "${sjson}" ]] && command -v jq >/dev/null 2>&1; then
    local row
    row="$(jq -r '[
      .status // "unknown",
      (.reason // "" | gsub("[\t\r\n]"; " ")),
      .done_condition // "",
      (.steps // 0 | tostring),
      (.clicks // 0 | tostring),
      (.requests_sent // 0 | tostring),
      (.images_sent // 0 | tostring),
      (.avg_seconds_per_step // 0 | tostring),
      (.all_images_cropped // false | tostring),
      (.tab_events // 0 | tostring)
    ] | @tsv' "${sjson}")"
    printf "%s\t%s\t%s\t%s\t%s\n" "${task_name}" "${mode_name}" "${idx}" "${rdir}" "${row}" >> "${SUMMARY_TSV}"
  else
    printf "%s\t%s\t%s\t%s\tmissing\tno summary.json\t\t0\t0\t0\t0\t0\tfalse\t0\n" \
      "${task_name}" "${mode_name}" "${idx}" "${rdir:-none}" >> "${SUMMARY_TSV}"
  fi
}

run_task_a() {
  local mode_flag="$1"
  local mode_label="$2"
  local idx="$3"
  local extra_flags=()
  if [[ -n "${mode_flag}" ]]; then
    extra_flags+=("${mode_flag}")
  fi

  ${BIN} run \
    --task "Find the cheapest one-way flight from Osaka (KIX) to Taipei (TPE) on Nov 3, 2026 on Google Flights" \
    --start-url "https://www.google.com/travel/flights?hl=en" \
    --fill "origin field=Osaka" \
    --fill "destination field=Taipei" \
    --fill "departure date=Nov 3, 2026" \
    --max-steps 25 \
    --experience-dir "${EXPERIENCE_DIR}" \
    ${HARNESS_ARGS[@]+"${HARNESS_ARGS[@]}"} \
    ${extra_flags[@]+"${extra_flags[@]}"} \
    --json || true

  local rdir
  rdir="$(latest_run_dir)"
  record_summary "TaskA_KIX_TPE" "${mode_label}" "${idx}" "${rdir}"
}

run_task_b() {
  local mode_flag="$1"
  local mode_label="$2"
  local idx="$3"
  local extra_flags=()
  if [[ -n "${mode_flag}" ]]; then
    extra_flags+=("${mode_flag}")
  fi

  # SAFETY RULES FOR TASK B:
  # 1. --preset events-zh
  # 2. --start-url https://www.accupass.com/search
  # 3. --allow-url-prefix https://www.accupass.com/
  # 4. --done-when checkout-page (stop immediately when registration form / confirm page is visible)
  # 5. Find ONE FREE event (免費), skip any paid event (price > 0), and NEVER click
  #    「送出報名」/「確認報名」/「確認送出」/「完成報名」/Submit/Confirm; never pay; never log in.
  ${BIN} run \
    --preset events-zh \
    --start-url "https://www.accupass.com/search" \
    --allow-url-prefix "https://www.accupass.com/" \
    --done-when checkout-page \
    --task "在 Accupass 搜尋並找到一個免費（免費 / NT\$0，跳過任何付費 price > 0 活動）的活動，進入活動頁並點選立即報名與免費票種，走到報名表單或確認頁面即立即停止（嚴禁點擊送出報名、確認報名、確認送出、完成報名、Submit、Confirm，嚴禁付款或登入）" \
    --max-steps 25 \
    --experience-dir "${EXPERIENCE_DIR}" \
    ${HARNESS_ARGS[@]+"${HARNESS_ARGS[@]}"} \
    ${extra_flags[@]+"${extra_flags[@]}"} \
    --json || true

  local rdir
  rdir="$(latest_run_dir)"
  record_summary "TaskB_Accupass_Free" "${mode_label}" "${idx}" "${rdir}"
}

echo "=== Running Task A (KIX -> TPE) 3x WITH --no-self-fix (baseline) ==="
for i in 1 2 3; do
  run_task_a "--no-self-fix" "no_self_fix" "${i}"
done

echo "=== Running Task A (KIX -> TPE) 3x WITHOUT --no-self-fix (self_fix) ==="
for i in 1 2 3; do
  run_task_a "" "self_fix" "${i}"
done

echo "=== Running Task B (Accupass events-zh FREE event) 3x WITH --no-self-fix (baseline) ==="
for i in 1 2 3; do
  run_task_b "--no-self-fix" "no_self_fix" "${i}"
done

echo "=== Running Task B (Accupass events-zh FREE event) 3x WITHOUT --no-self-fix (self_fix) ==="
for i in 1 2 3; do
  run_task_b "" "self_fix" "${i}"
done

echo ""
echo "=== Summary of summary.json fields (${SUMMARY_TSV}) ==="
column -t -s $'\t' "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"
