use decision_cu::agent::{Agent, StopReason};
use decision_cu::api::{
    Answer, ApiError, ApiKey, BoundingBox, DecisionBackend, Request, Response, Usage, DEFAULT_MODEL,
};
use decision_cu::browser::{Browser, BrowserError};
use decision_cu::capture::{CaptureError, Rect, Screen};
use decision_cu::decider::{DoneCondition, Preset, TargetSpec, VisionDecider};
use decision_cu::experience::{ExperienceEntry, ExperienceStore};
use decision_cu::harness::{CandidateKind, HarnessOverrides, HarnessTargetOverride};
use decision_cu::input::{Input, InputError, KeyCode};
use decision_cu::logging::RunLogger;
use decision_cu::ocr::{FakeOcr, OcrLine};
use decision_cu::privacy::prepare_image;
use decision_cu::safety::SafetyPolicy;
use image::{Rgba, RgbaImage};
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};

struct MockScreen {
    images: RefCell<VecDeque<RgbaImage>>,
    default_img: RgbaImage,
    rect: Option<Rect>,
    scale: f64,
}

impl MockScreen {
    fn new(img: RgbaImage, rect: Option<Rect>, scale: f64) -> Self {
        Self {
            images: RefCell::new(VecDeque::new()),
            default_img: img,
            rect,
            scale,
        }
    }

    fn sequence(images: Vec<RgbaImage>, rect: Option<Rect>, scale: f64) -> Self {
        let default_img = images
            .last()
            .cloned()
            .unwrap_or_else(|| RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255])));
        Self {
            images: RefCell::new(VecDeque::from(images)),
            default_img,
            rect,
            scale,
        }
    }
}

impl Screen for MockScreen {
    fn capture(&self) -> Result<RgbaImage, CaptureError> {
        if let Some(next) = self.images.borrow_mut().pop_front() {
            Ok(next)
        } else {
            Ok(self.default_img.clone())
        }
    }
    fn chrome_window_rect(&self) -> Option<Rect> {
        self.rect
    }
    fn scale_factor(&self) -> f64 {
        self.scale
    }
}

struct MockBackend {
    requests: RefCell<Vec<Request>>,
    responses: RefCell<VecDeque<Response>>,
}

impl MockBackend {
    fn new(responses: Vec<Response>) -> Self {
        Self {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::from(responses)),
        }
    }
}

impl DecisionBackend for MockBackend {
    fn ask(&self, req: &Request) -> Result<Response, ApiError> {
        req.validate()?;
        self.requests.borrow_mut().push(req.clone());
        self.responses
            .borrow_mut()
            .pop_front()
            .ok_or_else(|| ApiError::Transport("No more mock responses".to_string()))
    }
    fn requests_sent(&self) -> usize {
        self.requests.borrow().len()
    }
}

#[derive(Default)]
struct MockInput {
    clicks: Vec<(f64, f64)>,
    focus_clicks: Vec<(f64, f64)>,
    keys: Vec<KeyCode>,
    scrolls: Vec<i32>,
}

impl Input for MockInput {
    fn click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        self.clicks.push((x, y));
        Ok(())
    }
    fn focus_click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        self.focus_clicks.push((x, y));
        Ok(())
    }
    fn move_mouse(&mut self, _x: f64, _y: f64) -> Result<(), InputError> {
        Ok(())
    }
    fn type_text(&mut self, _text: &str) -> Result<(), InputError> {
        Ok(())
    }
    fn key(&mut self, key: KeyCode) -> Result<(), InputError> {
        self.keys.push(key);
        Ok(())
    }
    fn scroll_vertical(&mut self, amount: i32) -> Result<(), InputError> {
        self.scrolls.push(amount);
        Ok(())
    }
    fn release_modifiers(&mut self) -> Result<(), InputError> {
        Ok(())
    }
    fn select_all(&mut self) -> Result<(), InputError> {
        Ok(())
    }
}

struct SequenceBrowser {
    urls: RefCell<VecDeque<Option<String>>>,
    default_url: Option<String>,
}

impl SequenceBrowser {
    fn new(urls: Vec<Option<String>>) -> Self {
        let default_url = urls.last().cloned().flatten();
        Self {
            urls: RefCell::new(VecDeque::from(urls)),
            default_url,
        }
    }
}

impl Browser for SequenceBrowser {
    fn active_tab_url(&self) -> Option<String> {
        if let Some(next) = self.urls.borrow_mut().pop_front() {
            next
        } else {
            self.default_url.clone()
        }
    }
    fn set_active_tab_url(&mut self, _url: &str) -> Result<(), BrowserError> {
        Ok(())
    }
    fn activate_chrome(&mut self) -> bool {
        true
    }
    fn go_back(&mut self) -> Result<(), BrowserError> {
        Ok(())
    }
}

fn make_choice_response(kind: &str, target: &str, results_noul: f64) -> Response {
    let mut answers = BTreeMap::new();
    answers.insert(
        "kind".to_string(),
        Answer::Choice {
            choice: kind.to_string(),
            probabilities: BTreeMap::from([(kind.to_string(), 0.99)]),
            confidence: 0.99,
        },
    );
    answers.insert(
        "target".to_string(),
        Answer::Choice {
            choice: target.to_string(),
            probabilities: BTreeMap::from([(target.to_string(), 0.95)]),
            confidence: 0.95,
        },
    );
    answers.insert("results".to_string(), Answer::Noul { noul: results_noul });
    answers.insert("popup_present".to_string(), Answer::Noul { noul: 0.05 });
    answers.insert("footer".to_string(), Answer::Noul { noul: 0.05 });
    answers.insert("login_form".to_string(), Answer::Noul { noul: 0.02 });
    Response {
        answers,
        model: DEFAULT_MODEL.to_string(),
        usage: Usage {
            input_tokens: 100,
            output_tokens: 0,
        },
    }
}

fn make_box_response(
    top: f64,
    left: f64,
    bottom: f64,
    right: f64,
    confidence: f64,
    sensitive_noul: f64,
) -> Response {
    let mut answers = BTreeMap::new();
    answers.insert(
        "target_box".to_string(),
        Answer::Box {
            bbox: BoundingBox {
                top,
                left,
                bottom,
                right,
            },
            confidence,
        },
    );
    answers.insert(
        "sensitive".to_string(),
        Answer::Noul {
            noul: sensitive_noul,
        },
    );
    Response {
        answers,
        model: DEFAULT_MODEL.to_string(),
        usage: Usage {
            input_tokens: 100,
            output_tokens: 0,
        },
    }
}

fn make_item_choice_response(choice: &str, confidence: f64) -> Response {
    let mut answers = BTreeMap::new();
    answers.insert(
        "item".to_string(),
        Answer::Choice {
            choice: choice.to_string(),
            probabilities: BTreeMap::from([(choice.to_string(), confidence)]),
            confidence,
        },
    );
    Response {
        answers,
        model: DEFAULT_MODEL.to_string(),
        usage: Usage {
            input_tokens: 100,
            output_tokens: 0,
        },
    }
}

#[test]
fn a_no_harness_file_produces_identical_decisions_to_before() {
    let missing_path = std::env::temp_dir().join("decision_cu_nonexistent_harness_f10.json");
    let _ = std::fs::remove_file(&missing_path);

    let loaded = HarnessOverrides::load_from_path(&missing_path).unwrap();
    assert!(loaded.is_none());
    assert!(HarnessOverrides::load_optional(None).unwrap().is_none());

    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let prepared = prepare_image(&img, Some(window), 2.0).unwrap();

    let responses = || {
        vec![
            make_choice_response("click", "search button", 0.1),
            make_box_response(0.30, 0.30, 0.40, 0.50, 0.12, 0.02),
            make_box_response(0.45, 0.40, 0.55, 0.60, 0.89, 0.02),
        ]
    };

    let targets = vec![TargetSpec::new(
        "search button",
        Some("Blue search button".to_string()),
    )];

    let decider_before = VisionDecider::new(MockBackend::new(responses()), targets.clone());
    let decider_missing_harness =
        VisionDecider::new(MockBackend::new(responses()), targets.clone()).with_harness_opt(loaded);

    let d1 = decider_before.decide("Find flights", &[], &prepared).unwrap();
    let d2 = decider_missing_harness
        .decide("Find flights", &[], &prepared)
        .unwrap();

    assert_eq!(d1.kind, d2.kind);
    assert_eq!(d1.target, d2.target);
    assert_eq!(d1.click_point_screen_pt, d2.click_point_screen_pt);
    assert_eq!(d1.bbox, d2.bbox);
    assert_eq!(d1.box_confidence, d2.box_confidence);
    assert_eq!(d1.reasked, d2.reasked);
    assert_eq!(
        decider_before.requests_sent(),
        decider_missing_harness.requests_sent()
    );

    // Baseline flag --no-self-fix disables new self-fix probe paths while preserving legacy low-confidence re-ask
    let decider_no_self_fix = VisionDecider::new(
        MockBackend::new(responses()),
        targets,
    )
    .with_self_fix(false);

    let d_baseline = decider_no_self_fix
        .decide("Find flights", &[], &prepared)
        .unwrap();
    assert!(d_baseline.reasked);
    assert!(d_baseline.self_fix.is_none());
    assert!(d_baseline.click_point_screen_pt.is_some());
    assert_eq!(decider_no_self_fix.requests_sent(), 3);
}

#[test]
fn b_low_conf_picks_best_candidate_and_respects_budget_and_ocr_bonus() {
    let temp_dir = std::env::temp_dir().join("decision_cu_test_f10_b");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let store = ExperienceStore::new(temp_dir.join("experience"));
    store
        .write_entry(
            "events-zh",
            "www.accupass.com",
            &ExperienceEntry {
                cue: "搜尋或活動分類".to_string(),
                lesson: "頁面頂部中間的搜尋輸入框或活動分類按鈕".to_string(),
                evidence: "step 0 click navigated to search".to_string(),
                confidence: 0.85,
                adopted_candidate: "reask".to_string(),
                uses: 2,
                wins: 2,
            },
        )
        .unwrap();

    // Part 1: Budget = 2 with 4 candidates in candidate_order -> only first 2 ("reask", "experience") are probed
    let harness_budget_2 = HarnessOverrides {
        targets: vec![HarnessTargetOverride {
            label: "搜尋或活動分類".to_string(),
            description: Some("頂部搜尋列".to_string()),
            reask: vec!["點擊頂部搜尋欄".to_string()],
        }],
        candidate_order: vec![
            CandidateKind::Reask,
            CandidateKind::Experience,
            CandidateKind::Zoom,
            CandidateKind::Ocr,
        ],
        max_extra_requests_per_step: 2,
    };

    let img1 = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let mut img2 = img1.clone();
    for y in 300..900 {
        for x in 300..900 {
            img2.put_pixel(x, y, Rgba([20, 40, 200, 255]));
        }
    }
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen::sequence(vec![img1.clone(), img2], Some(window), 2.0);

    let backend = MockBackend::new(vec![
        // Step 0: choice + base box (0.11) + reask (0.52) + experience (0.84)
        make_choice_response("click", "搜尋或活動分類", 0.1),
        make_box_response(0.30, 0.30, 0.40, 0.40, 0.11, 0.02),
        make_box_response(0.35, 0.35, 0.45, 0.45, 0.52, 0.02),
        make_box_response(0.50, 0.40, 0.60, 0.60, 0.84, 0.02),
        // Step 1: done
        make_choice_response("done", "nothing to click", 0.92),
    ]);

    let logger = RunLogger::new_in_dir(&temp_dir.join("logs"), None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage)
        .with_harness(harness_budget_2)
        .with_experience_store(store.clone());

    let browser = SequenceBrowser::new(vec![
        Some("https://www.accupass.com/search".to_string()),
        Some("https://www.accupass.com/search?q=free".to_string()),
    ]);

    let ocr = FakeOcr::new(vec![OcrLine::new(
        "免費活動",
        0.95,
        0.50,
        0.40,
        0.60,
        0.60,
    )]);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser)
        .with_ocr(ocr)
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com/search".to_string()))
        .with_logger(logger);

    let summary = agent.run("Find free event").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.clicks, 1);

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    let self_fix_evt: serde_json::Value = log_str
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .find(|v| v.get("event").and_then(|e| e.as_str()) == Some("self_fix"))
        .expect("Expected self_fix event in steps.jsonl");

    assert_eq!(self_fix_evt["step"], 0);
    assert_eq!(self_fix_evt["target"], "搜尋或活動分類");
    assert_eq!(self_fix_evt["chosen"], "experience");
    assert_eq!(self_fix_evt["budget_used"], 2);
    assert!((self_fix_evt["base_conf"].as_f64().unwrap() - 0.11).abs() < 1e-6);
    let tried = self_fix_evt["tried"].as_array().unwrap();
    assert_eq!(tried.len(), 2);
    assert_eq!(tried[0]["candidate"], "reask");
    assert_eq!(tried[1]["candidate"], "experience");

    // Part 2: OCR center & box center agreement bonus within TRIED_POSITION_TOLERANCE_FRAC (0.08)
    // "reask" has raw conf 0.76 at center (0.50, 0.50); "experience" has raw conf 0.80 at center (0.25, 0.80);
    // "ocr" has raw conf 0.72 at center (0.52, 0.51) (within 0.08 of reask!).
    // With the +0.10 agreement bonus, "reask" becomes 0.86 (> 0.80) and is chosen!
    let harness_bonus = HarnessOverrides {
        targets: vec![],
        candidate_order: vec![
            CandidateKind::Reask,
            CandidateKind::Experience,
            CandidateKind::Ocr,
        ],
        max_extra_requests_per_step: 3,
    };

    let prepared = prepare_image(&img1, Some(window), 2.0).unwrap();
    let backend_bonus = MockBackend::new(vec![
        // base box (0.10)
        make_box_response(0.20, 0.20, 0.30, 0.30, 0.10, 0.01),
        // reask box: center (0.50, 0.50), raw conf 0.76 -> +0.10 OCR agreement bonus = 0.86
        make_box_response(0.45, 0.45, 0.55, 0.55, 0.76, 0.01),
        // experience box: center (0.25, 0.80), raw conf 0.80 (no OCR agreement)
        make_box_response(0.75, 0.20, 0.85, 0.30, 0.80, 0.01),
        // ocr item choice: index "0" at center (0.52, 0.51), raw conf 0.72
        make_item_choice_response("0", 0.72),
    ]);

    let decider_bonus = VisionDecider::new(backend_bonus, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_harness(harness_bonus)
        .with_experience_store(store);
    decider_bonus.set_current_url(Some("https://www.accupass.com/search".to_string()));
    decider_bonus.set_ocr_candidates(vec![OcrLine::new(
        "免費活動",
        0.95,
        0.46,
        0.47,
        0.56,
        0.57,
    )]);

    let (bbox, final_conf, _sens, reasked) = decider_bonus
        .resolve_click_box("Goal: find event", &prepared, "搜尋或活動分類", None)
        .unwrap();
    assert!(reasked);
    assert!((final_conf - 0.86).abs() < 1e-6);
    assert!((bbox.center().0 - 0.50).abs() < 1e-6);
    let sf = decider_bonus.last_self_fix().unwrap();
    assert_eq!(sf.chosen.as_deref(), Some("reask"));
    assert_eq!(sf.budget_used, 3);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn c_safety_blocked_candidate_never_adopted() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let prepared = prepare_image(&img, Some(window), 2.0).unwrap();

    // Case 1: Candidate box falls in the top 80pt tab strip (unsafe) AND second candidate has high sensitive_noul >= 0.5 (unsafe)
    // Note: prepare_image crops top 80pt, so a box with negative top/bottom or mapped y < 80pt is rejected,
    // and a box with sensitive_noul = 0.85 (>= 0.5) or forbidden target description is rejected.
    let harness = HarnessOverrides {
        targets: vec![HarnessTargetOverride {
            label: "立即報名按鈕".to_string(),
            description: Some("報名按鈕".to_string()),
            reask: vec!["点击付款 purchase button".to_string()], // contains forbidden words "付款" and "purchase"
        }],
        candidate_order: vec![
            CandidateKind::Reask,
            CandidateKind::Zoom,
            CandidateKind::Ocr,
        ],
        max_extra_requests_per_step: 3,
    };

    let backend = MockBackend::new(vec![
        // Base box: low confidence 0.10
        make_box_response(0.40, 0.40, 0.50, 0.50, 0.10, 0.02),
        // Reask candidate: high confidence 0.95, but prompt contains forbidden word ("付款" / "purchase")
        make_box_response(0.40, 0.40, 0.50, 0.50, 0.95, 0.02),
        // Zoom candidate: high confidence 0.93, but sensitive_noul = 0.88 (>= 0.5)
        make_box_response(0.40, 0.40, 0.50, 0.50, 0.93, 0.88),
        // OCR candidate: high confidence 0.96, but OCR text is "會員登入" (contains forbidden word "登入")
        make_item_choice_response("0", 0.96),
    ]);

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_harness(harness);
    decider.set_ocr_candidates(vec![OcrLine::new(
        "會員登入",
        0.98,
        0.40,
        0.40,
        0.50,
        0.55,
    )]);

    let (_bbox, box_conf, _sens, reasked) = decider
        .resolve_click_box("Goal: register", &prepared, "立即報名按鈕", None)
        .unwrap();
    assert!(reasked);
    assert!(
        box_conf < 0.20,
        "Safety-blocked candidates must not be adopted (got conf {})",
        box_conf
    );
    let sf = decider.last_self_fix().unwrap();
    assert!(sf.chosen.is_none());
    assert_eq!(sf.tried.len(), 3);
}

#[test]
fn d_experience_entry_written_only_with_evidence_and_disabled_after_poor_win_rate() {
    let temp_dir = std::env::temp_dir().join("decision_cu_test_f10_d");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let exp_dir = temp_dir.join("experience");
    let store = ExperienceStore::new(&exp_dir);

    let img_same = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let mut img_changed = img_same.clone();
    for y in 200..1000 {
        for x in 200..1200 {
            img_changed.put_pixel(x, y, Rgba([10, 180, 90, 255]));
        }
    }
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };

    // 1. Run where self_fix adopts a candidate, but neither URL nor screen changes (no evidence -> NOT written)
    {
        let screen = MockScreen::sequence(
            vec![img_same.clone(), img_same.clone()],
            Some(window),
            2.0,
        );
        let backend = MockBackend::new(vec![
            make_choice_response("click", "搜尋或活動分類", 0.1),
            make_box_response(0.30, 0.30, 0.40, 0.40, 0.10, 0.01),
            make_box_response(0.45, 0.40, 0.55, 0.60, 0.88, 0.01),
            make_choice_response("done", "nothing to click", 0.95),
        ]);
        let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
            .with_preset(Preset::EventsZh)
            .with_done_condition(DoneCondition::CheckoutPage)
            .with_experience_store(store.clone());
        let browser = SequenceBrowser::new(vec![
            Some("https://www.accupass.com/search".to_string()),
            Some("https://www.accupass.com/search".to_string()),
        ]);
        let mut agent =
            Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
                .with_browser(browser)
                .with_flights_preset(false)
                .with_start_url(Some("https://www.accupass.com/search".to_string()));
        let _ = agent.run("Find free event").unwrap();

        assert!(
            store
                .load_entry("events-zh", "www.accupass.com", "搜尋或活動分類")
                .is_none(),
            "Experience entry must NOT be created without win evidence"
        );
    }

    // 2. Run where self_fix adopts a candidate AND URL/screen changes -> entry created with evidence & INDEX.md rebuilt
    {
        let screen = MockScreen::sequence(
            vec![img_same.clone(), img_changed.clone()],
            Some(window),
            2.0,
        );
        let backend = MockBackend::new(vec![
            make_choice_response("click", "搜尋或活動分類", 0.1),
            make_box_response(0.30, 0.30, 0.40, 0.40, 0.10, 0.01),
            make_box_response(0.45, 0.40, 0.55, 0.60, 0.88, 0.01),
            make_choice_response("done", "nothing to click", 0.95),
        ]);
        let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
            .with_preset(Preset::EventsZh)
            .with_done_condition(DoneCondition::CheckoutPage)
            .with_experience_store(store.clone());
        let browser = SequenceBrowser::new(vec![
            Some("https://www.accupass.com/search".to_string()),
            Some("https://www.accupass.com/search?q=free".to_string()),
        ]);
        let mut agent =
            Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
                .with_browser(browser)
                .with_flights_preset(false)
                .with_start_url(Some("https://www.accupass.com/search".to_string()));
        let _ = agent.run("Find free event").unwrap();

        let entry = store
            .lookup("events-zh", "www.accupass.com", "搜尋或活動分類")
            .expect("Expected experience entry after first win with evidence");
        assert_eq!(entry.uses, 1);
        assert_eq!(entry.wins, 1);
        assert_eq!(entry.adopted_candidate, "reask");
        assert!(!entry.evidence.is_empty());
        assert!(!entry.is_disabled());

        let index_md =
            std::fs::read_to_string(store.index_path("events-zh", "www.accupass.com")).unwrap();
        assert!(index_md.contains("搜尋或活動分類"));
        assert!(index_md.contains("active"));
    }

    // 3. Record 3 losses -> uses = 4, wins = 1 (win rate 0.25 < 0.5) -> entry disabled
    for _ in 0..3 {
        store
            .record_outcome(
                "events-zh",
                "www.accupass.com",
                "搜尋或活動分類",
                "lesson",
                "reask",
                0.88,
                false,
                None,
            )
            .unwrap();
    }

    let raw_entry = store
        .load_entry("events-zh", "www.accupass.com", "搜尋或活動分類")
        .unwrap();
    assert_eq!(raw_entry.uses, 4);
    assert_eq!(raw_entry.wins, 1);
    assert!(raw_entry.is_disabled());
    assert!(
        store
            .lookup("events-zh", "www.accupass.com", "搜尋或活動分類")
            .is_none(),
        "Disabled entry must not be returned by lookup()"
    );
    let index_md =
        std::fs::read_to_string(store.index_path("events-zh", "www.accupass.com")).unwrap();
    assert!(index_md.contains("disabled"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn e_redact_applied_to_experience_and_self_fix_logs_and_never_stores_image_bytes() {
    let temp_dir = std::env::temp_dir().join("decision_cu_test_f10_e");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let secret_str = format!("sk-{}-{}", "vsr", "secret-test-key-987654");
    let secret = ApiKey::new(&secret_str);
    let store =
        ExperienceStore::new(temp_dir.join("experience")).with_api_key(Some(secret.clone()));

    let token_43 = format!("sk-{}-{}", "vsr", "1234567890123456789012345678901234567890123");
    let token_43 = token_43.as_str();
    let entry = ExperienceEntry {
        cue: "search button".to_string(),
        lesson: format!(
            "Locate search button (auth {} and data:image/jpeg;base64,/9j/4AAQSkZJRg==)",
            token_43
        ),
        evidence: format!("VSR_API_KEY={} navigated", secret_str),
        confidence: 0.91,
        adopted_candidate: "reask".to_string(),
        uses: 1,
        wins: 1,
    };
    let path = store
        .write_entry("flights", "www.google.com", &entry)
        .unwrap();

    let raw_md = std::fs::read_to_string(&path).unwrap();
    assert!(!raw_md.contains(secret_str.as_str()));
    assert!(!raw_md.contains(token_43));
    assert!(!raw_md.contains("data:image/jpeg;base64"));
    assert!(!raw_md.contains("/9j/4AAQSkZJRg=="));
    assert!(raw_md.contains("***"));
    assert!(raw_md.contains("[image_omitted]"));

    let index_md = std::fs::read_to_string(store.index_path("flights", "www.google.com")).unwrap();
    assert!(!index_md.contains(secret_str.as_str()));
    assert!(!index_md.contains(token_43));

    // Also verify self_fix event log in steps.jsonl is redacted
    let logger = RunLogger::new_in_dir(&temp_dir.join("logs"), Some(secret)).unwrap();
    let steps_path = logger.path().to_path_buf();

    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen::new(img, Some(window), 2.0);
    let backend = MockBackend::new(vec![
        make_choice_response("click", "search button", 0.1),
        make_box_response(0.30, 0.30, 0.40, 0.40, 0.12, 0.01),
        make_box_response(0.45, 0.45, 0.55, 0.55, 0.89, 0.01),
        make_box_response(0.45, 0.45, 0.55, 0.55, 0.91, 0.01),
        make_box_response(0.45, 0.45, 0.55, 0.55, 0.85, 0.01),
        make_choice_response("done", "nothing to click", 0.95),
    ]);

    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new(
            "search button",
            Some(format!("Search button {}", secret_str)),
        )],
    )
    .with_preset(Preset::Flights)
    .with_experience_store(store);

    let browser = SequenceBrowser::new(vec![
        Some("https://www.google.com/travel/flights".to_string()),
        Some("https://www.google.com/travel/flights?q=results".to_string()),
    ]);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser)
        .with_start_url(Some("https://www.google.com/travel/flights".to_string()))
        .with_logger(logger);

    let _ = agent.run("Find flights").unwrap();
    let log_contents = std::fs::read_to_string(&steps_path).unwrap();
    assert!(log_contents.contains("\"event\":\"self_fix\""));
    assert!(!log_contents.contains(secret_str.as_str()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn f_no_harness_non_card_low_conf_does_not_add_zoom_request() {
    // Pre-self-fix behaviour: a non-card target with low confidence gets exactly one
    // longer re-ask and no zoom. Without harness/experience/OCR this must be unchanged.
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let prepared = prepare_image(&img, Some(window), 2.0).unwrap();
    let targets = vec![TargetSpec::new(
        "search button",
        Some("Blue search button".to_string()),
    )];
    let decider = VisionDecider::new(
        MockBackend::new(vec![
            make_choice_response("click", "search button", 0.1),
            make_box_response(0.30, 0.30, 0.40, 0.50, 0.10, 0.02),
            make_box_response(0.30, 0.30, 0.40, 0.50, 0.11, 0.02),
        ]),
        targets,
    );
    let d = decider.decide("Find flights", &[], &prepared).unwrap();
    assert!(d.reasked);
    assert!(d.click_point_screen_pt.is_none());
    assert_eq!(decider.requests_sent(), 3);
}

#[test]
fn g_experience_text_strips_url_query_and_fragment() {
    use decision_cu::experience::sanitize_text;
    let out = sanitize_text(
        "click changed URL (https://www.accupass.com/event/abc?token=SECRET&email=a@b.c -> https://x.com/p#frag)",
        None,
    );
    assert!(!out.contains("SECRET"));
    assert!(!out.contains("a@b.c"));
    assert!(!out.contains("frag"));
    assert!(out.contains("https://www.accupass.com/event/abc"));
    assert!(out.contains("https://x.com/p"));
}

#[test]
fn h_no_self_fix_runs_legacy_reask_and_page_turn_retry_while_disabling_new_paths() {
    use decision_cu::api::Question;

    let temp_dir = std::env::temp_dir().join("decision_cu_test_f10_h");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let exp_dir = temp_dir.join("experience");
    let store = ExperienceStore::new(&exp_dir);

    // Pre-populate an experience entry to prove it is NOT read or updated when self_fix_enabled=false
    store
        .write_entry(
            "events-zh",
            "www.accupass.com",
            &ExperienceEntry {
                cue: "活動卡片".to_string(),
                lesson: "經驗庫中的活動卡片提示".to_string(),
                evidence: "prior win".to_string(),
                confidence: 0.91,
                adopted_candidate: "reask".to_string(),
                uses: 1,
                wins: 1,
            },
        )
        .unwrap();

    // Configure harness with rewritten description and custom re-ask prompts, plus all 4 candidate kinds
    let harness = HarnessOverrides {
        targets: vec![HarnessTargetOverride {
            label: "活動卡片".to_string(),
            description: Some("HARNESS_REWRITTEN_DESC".to_string()),
            reask: vec!["HARNESS_CUSTOM_REASK_PROMPT".to_string()],
        }],
        candidate_order: vec![
            CandidateKind::Reask,
            CandidateKind::Experience,
            CandidateKind::Zoom,
            CandidateKind::Ocr,
        ],
        max_extra_requests_per_step: 4,
    };

    let img_same = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let mut img_changed = img_same.clone();
    for y in 250..950 {
        for x in 250..1150 {
            img_changed.put_pixel(x, y, Rgba([30, 120, 210, 255]));
        }
    }
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };

    // Screen sequence across steps:
    // Step 0 (click with low initial conf -> legacy reask succeeds):
    //   capture 1: step 0 initial window capture (img_same)
    // Step 1 (scroll where wheel does not change screen, PageDown fallback changes screen):
    //   capture 2: step 1 initial window capture (img_same)
    //   capture 3: after wheel scroll (img_same -> unchanged, triggers PageDown fallback!)
    //   capture 4: after PageDown fallback (img_changed -> changed!)
    // Step 2 (done):
    //   capture 5: step 2 initial window capture (img_changed)
    let screen = MockScreen::sequence(
        vec![
            img_same.clone(),
            img_same.clone(),
            img_same.clone(),
            img_changed.clone(),
            img_changed.clone(),
        ],
        Some(window),
        2.0,
    );

    let backend = MockBackend::new(vec![
        // Step 0: choice ("click", "活動卡片") + initial low-conf box (0.11)
        // + legacy card re-ask 1 (0.12) + legacy card re-ask 2 (0.14) + legacy content 2x zoom (0.87)
        make_choice_response("click", "活動卡片", 0.1),
        make_box_response(0.30, 0.30, 0.40, 0.40, 0.11, 0.01),
        make_box_response(0.32, 0.32, 0.42, 0.42, 0.12, 0.01),
        make_box_response(0.34, 0.34, 0.44, 0.44, 0.14, 0.01),
        make_box_response(0.45, 0.40, 0.55, 0.60, 0.87, 0.01),
        // Step 1: choice ("scroll", "nothing to click") -> runs wheel scroll then PageDown retry
        make_choice_response("scroll", "nothing to click", 0.2),
        // Step 2: choice ("done", "nothing to click")
        make_choice_response("done", "nothing to click", 0.95),
    ]);

    let logger = RunLogger::new_in_dir(&temp_dir.join("logs"), None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage)
        .with_harness(harness)
        .with_experience_store(store.clone())
        .with_self_fix(false);

    // Experience lookup must return None when self_fix_enabled is false
    assert!(decider.lookup_experience("活動卡片").is_none());

    // OCR candidates are supplied via FakeOcr, but must NOT be queried when self_fix_enabled=false
    let ocr = FakeOcr::new(vec![OcrLine::new(
        "免費講座活動",
        0.96,
        0.45,
        0.40,
        0.55,
        0.60,
    )]);

    let browser = SequenceBrowser::new(vec![
        Some("https://www.accupass.com/search".to_string()),
        Some("https://www.accupass.com/search?q=free".to_string()),
        Some("https://www.accupass.com/search?q=free".to_string()),
        Some("https://www.accupass.com/search?q=free".to_string()),
    ]);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser)
        .with_ocr(ocr)
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com/search".to_string()))
        .with_logger(logger);

    let summary = agent.run("Find free event").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.clicks, 1);
    // 7 requests: 5 in step 0 (choice + initial box + 2 card_primary_reask_descriptions + 1 legacy content 2x zoom),
    // 1 in step 1 (scroll), 1 in step 2 (done)
    assert_eq!(summary.requests_sent, 7);
    assert_eq!(summary.images_sent, 7);
    assert!(agent.decider().last_self_fix().is_none());

    // Inspect the sent requests to verify:
    // 1) Legacy card re-ask used card_primary_reask_descriptions("活動卡片") followed by legacy content 2x zoom
    // 2) Harness rewritten description / reask ("HARNESS_REWRITTEN_DESC", "HARNESS_CUSTOM_REASK_PROMPT") did NOT fire
    // 3) Experience prompt ("經驗庫中的活動卡片提示") did NOT fire
    // 4) OCR item choice question ("item") did NOT fire
    // 5) Only the single legacy step_0_content_2x zoom record was recorded (no extra self-fix zoom requests)
    let sent_reqs = agent.decider().backend().requests.borrow();
    assert_eq!(sent_reqs.len(), 7);

    // Request 0: target criteria must use original base description, not HARNESS_REWRITTEN_DESC
    if let Some(Question::Choice { criteria, .. }) = sent_reqs[0].questions.get("target") {
        let card_desc = criteria.get("活動卡片").and_then(|d| d.as_deref());
        assert_eq!(card_desc, Some("列表中顯示日期與標題的即將舉辦活動卡片"));
    } else {
        panic!("Expected target choice question in req 0");
    }

    // Request 1: initial box query for "活動卡片"
    if let Some(Question::Box { instructions }) = sent_reqs[1].questions.get("target_box") {
        assert_eq!(instructions, "活動卡片");
    } else {
        panic!("Expected initial target_box question in req 1");
    }

    // Requests 2 & 3: legacy card-specific re-ask descriptions from card_primary_reask_descriptions("活動卡片")
    if let Some(Question::Box { instructions }) = sent_reqs[2].questions.get("target_box") {
        assert_eq!(
            instructions,
            decision_cu::decider::CARD_REASK_FIRST_CARD_IMAGE_DESC
        );
    } else {
        panic!("Expected card re-ask 1 target_box question in req 2");
    }
    if let Some(Question::Box { instructions }) = sent_reqs[3].questions.get("target_box") {
        assert_eq!(
            instructions,
            decision_cu::decider::CARD_REASK_TITLE_TEXT_DESC
        );
    } else {
        panic!("Expected card re-ask 2 target_box question in req 3");
    }

    // Request 4: legacy content 2x zoom using first card re-ask description
    if let Some(Question::Box { instructions }) = sent_reqs[4].questions.get("target_box") {
        assert_eq!(
            instructions,
            decision_cu::decider::CARD_REASK_FIRST_CARD_IMAGE_DESC
        );
    } else {
        panic!("Expected legacy content 2x zoom target_box question in req 4");
    }
    assert_ne!(
        sent_reqs[4].images[0], sent_reqs[1].images[0],
        "Legacy content 2x zoom must send the 2x content crop image"
    );

    for req in sent_reqs.iter() {
        assert!(
            !req.questions.contains_key("item"),
            "OCR item question must not be sent when self_fix_enabled=false"
        );
        for q in req.questions.values() {
            if let Question::Box { instructions } = q {
                assert!(!instructions.contains("HARNESS_"));
                assert!(!instructions.contains("經驗庫中的活動卡片提示"));
            }
        }
    }

    // Verify the legacy step_0_content_2x audit record was recorded and no extra zoom records exist
    let zoom_records: Vec<_> = agent
        .audit_log()
        .records()
        .iter()
        .filter(|r| r.purpose.contains("2x"))
        .collect();
    assert_eq!(
        zoom_records.len(),
        1,
        "Expected exactly 1 legacy content 2x zoom record, got {:?}",
        zoom_records
    );
    assert_eq!(zoom_records[0].purpose, "step_0_content_2x");

    // Verify legacy popup corner 2x zoom in resolve_dismiss_popup still works when self_fix_enabled=false
    {
        use decision_cu::decider::{ActionKind, Decision};
        use decision_cu::privacy::AuditLog;

        let popup_answers = |conf: f64, closeish: f64| -> Response {
            let mut answers = BTreeMap::new();
            answers.insert(
                "target_box".to_string(),
                Answer::Box {
                    bbox: BoundingBox {
                        top: 0.80,
                        left: 0.80,
                        bottom: 0.85,
                        right: 0.88,
                    },
                    confidence: conf,
                },
            );
            answers.insert("closeish".to_string(), Answer::Noul { noul: closeish });
            Response {
                answers,
                model: DEFAULT_MODEL.to_string(),
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 0,
                },
            }
        };
        let popup_backend = MockBackend::new(vec![
            popup_answers(0.05, 0.2), // initial ask
            popup_answers(0.05, 0.2), // reask 1
            popup_answers(0.05, 0.2), // reask 2
            popup_answers(0.05, 0.2), // reask 3
            popup_answers(0.05, 0.2), // reask 4
            popup_answers(0.05, 0.2), // reask 5
            popup_answers(0.05, 0.2), // reask 6
            popup_answers(0.92, 0.90), // bottom_right_2x corner zoom succeeds!
        ]);
        let popup_decider = VisionDecider::new(popup_backend, Preset::EventsZh.default_targets())
            .with_preset(Preset::EventsZh)
            .with_self_fix(false);
        let prepared = prepare_image(&img_same, Some(window), 2.0).unwrap();
        let mut popup_decision = Decision {
            kind: ActionKind::DismissPopup,
            kind_confidence: 0.9,
            target: Some("關閉彈窗".to_string()),
            click_point_screen_pt: None,
            bbox: None,
            box_confidence: None,
            confidence: 0.9,
            reasked: false,
            text_to_type: None,
            results_noul: 0.1,
            sensitive_noul: None,
            closeish_noul: None,
            popup_present_noul: Some(0.9),
            footer_noul: None,
            login_form_noul: None,
            click_source: None,
            ocr_item_text: None,
            ocr_item_center_frac: None,
            self_fix: None,
        };
        let mut popup_audit = AuditLog::default();
        popup_decider
            .resolve_dismiss_popup("Goal", &prepared, &mut popup_decision, Some((&mut popup_audit, 0)))
            .unwrap();
        assert!(popup_decision.click_point_screen_pt.is_some());
        assert!(
            popup_audit
                .records()
                .iter()
                .any(|r| r.purpose == "step_0_popup_bottom_right_2x"),
            "Legacy popup corner 2x zoom must still run when self_fix_enabled=false"
        );
    }

    // Verify experience store was NOT written/updated (uses is still 1)
    let persisted = store
        .load_entry("events-zh", "www.accupass.com", "活動卡片")
        .unwrap();
    assert_eq!(persisted.uses, 1);
    assert_eq!(persisted.wins, 1);
    assert_eq!(persisted.lesson, "經驗庫中的活動卡片提示");

    // Verify page-turn retry (focus_click + KeyCode::PageDown) ran on step 1 scroll
    assert_eq!(agent.input().scrolls.len(), 1);
    assert_eq!(agent.input().focus_clicks.len(), 1);
    assert!(agent.input().keys.contains(&KeyCode::PageDown));

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    assert!(
        !log_str.contains("\"event\":\"self_fix\""),
        "self_fix event must not be logged when self_fix_enabled=false"
    );
    assert!(
        log_str.contains("\"method\":\"pagedown_fallback\""),
        "page-turn fallback must still run when self_fix_enabled=false"
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn i_min_ocr_text_chars_rejects_short_digits_and_punctuation_from_experience_store() {
    use decision_cu::experience::{is_valid_ocr_text, MIN_OCR_TEXT_CHARS};

    assert_eq!(MIN_OCR_TEXT_CHARS, 2);

    // Short (< 2 non-whitespace chars), purely digits, or purely punctuation must be rejected
    for invalid in [
        "",
        "   ",
        "3",
        " 3 ",
        "a",
        "免",
        "12",
        "123",
        " 2026 ",
        "０１",
        "..",
        "---",
        "!@#",
        "「」",
        "（）",
        "+-",
    ] {
        assert!(
            !is_valid_ocr_text(invalid),
            "Expected OCR text {:?} to be rejected",
            invalid
        );
    }

    // Valid multi-char non-digit non-punctuation OCR texts must pass
    for valid in ["免費", "Go", "NT$0", "立即報名", "10月", " 免費活動 "] {
        assert!(
            is_valid_ocr_text(valid),
            "Expected OCR text {:?} to be accepted",
            valid
        );
    }

    let temp_dir = std::env::temp_dir().join("decision_cu_test_f10_i");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let store = ExperienceStore::new(temp_dir.join("experience"));

    // Direct record_outcome and write_entry must refuse short / digit-only / punctuation-only OCR lessons
    for bad_lesson in ["3", " 7 ", "12", "..."] {
        let res = store
            .record_outcome(
                "events-zh",
                "www.accupass.com",
                "票種選項",
                bad_lesson,
                "ocr",
                0.90,
                true,
                Some("step 0 click changed screen"),
            )
            .unwrap();
        assert!(
            res.is_none(),
            "Short/invalid OCR lesson {:?} must not be recorded",
            bad_lesson
        );
        assert!(
            store
                .load_entry("events-zh", "www.accupass.com", "票種選項")
                .is_none()
        );

        let write_res = store.write_entry(
            "events-zh",
            "www.accupass.com",
            &ExperienceEntry {
                cue: "票種選項".to_string(),
                lesson: bad_lesson.to_string(),
                evidence: "step 0 click changed screen".to_string(),
                confidence: 0.90,
                adopted_candidate: "ocr".to_string(),
                uses: 1,
                wins: 1,
            },
        );
        assert!(write_res.is_err());
    }

    // End-to-end Agent run where self-fix adopts an OCR candidate with text "3":
    // click succeeds and URL/screen changes, but "3" must NOT be written to the experience library.
    let img1 = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let mut img2 = img1.clone();
    for y in 200..900 {
        for x in 200..1100 {
            img2.put_pixel(x, y, Rgba([50, 160, 80, 255]));
        }
    }
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };

    let harness_ocr_only = HarnessOverrides {
        targets: vec![],
        candidate_order: vec![CandidateKind::Ocr],
        max_extra_requests_per_step: 1,
    };

    {
        let screen = MockScreen::sequence(vec![img1.clone(), img2.clone()], Some(window), 2.0);
        let backend = MockBackend::new(vec![
            make_choice_response("click", "數量加號", 0.1),
            make_box_response(0.30, 0.30, 0.40, 0.40, 0.10, 0.01),
            make_item_choice_response("0", 0.92),
            make_choice_response("done", "nothing to click", 0.95),
        ]);
        let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
            .with_preset(Preset::EventsZh)
            .with_done_condition(DoneCondition::CheckoutPage)
            .with_harness(harness_ocr_only.clone())
            .with_experience_store(store.clone());
        let browser = SequenceBrowser::new(vec![
            Some("https://www.accupass.com/event/123".to_string()),
            Some("https://www.accupass.com/event/123?step=2".to_string()),
        ]);
        let ocr = FakeOcr::new(vec![OcrLine::new("3", 0.98, 0.45, 0.45, 0.55, 0.55)]);
        let mut agent =
            Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
                .with_browser(browser)
                .with_ocr(ocr)
                .with_flights_preset(false)
                .with_start_url(Some("https://www.accupass.com/event/123".to_string()));
        let summary = agent.run("Select ticket").unwrap();
        assert_eq!(summary.stop_reason, StopReason::Done);
        assert!(
            store
                .load_entry("events-zh", "www.accupass.com", "數量加號")
                .is_none(),
            "OCR text '3' must never be recorded in the experience library"
        );
    }

    // Contrast: when OCR candidate has valid text "免費票", it IS recorded in the experience library.
    {
        let screen = MockScreen::sequence(vec![img1, img2], Some(window), 2.0);
        let backend = MockBackend::new(vec![
            make_choice_response("click", "票種選項", 0.1),
            make_box_response(0.30, 0.30, 0.40, 0.40, 0.10, 0.01),
            make_item_choice_response("0", 0.93),
            make_choice_response("done", "nothing to click", 0.95),
        ]);
        let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
            .with_preset(Preset::EventsZh)
            .with_done_condition(DoneCondition::CheckoutPage)
            .with_harness(harness_ocr_only)
            .with_experience_store(store.clone());
        let browser = SequenceBrowser::new(vec![
            Some("https://www.accupass.com/event/123".to_string()),
            Some("https://www.accupass.com/event/123?ticket=free".to_string()),
        ]);
        let ocr = FakeOcr::new(vec![OcrLine::new("免費票", 0.98, 0.45, 0.45, 0.55, 0.55)]);
        let mut agent =
            Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
                .with_browser(browser)
                .with_ocr(ocr)
                .with_flights_preset(false)
                .with_start_url(Some("https://www.accupass.com/event/123".to_string()));
        let summary = agent.run("Select free ticket").unwrap();
        assert_eq!(summary.stop_reason, StopReason::Done);
        let entry = store
            .load_entry("events-zh", "www.accupass.com", "票種選項")
            .expect("Valid OCR text '免費票' should be recorded");
        assert_eq!(entry.lesson, "免費票");
        assert_eq!(entry.adopted_candidate, "ocr");
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn j_no_self_fix_non_card_target_runs_label_description_legacy_reask() {
    use decision_cu::api::Question;
    use decision_cu::privacy::AuditLog;

    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let prepared = prepare_image(&img, Some(window), 2.0).unwrap();

    let harness = HarnessOverrides {
        targets: vec![HarnessTargetOverride {
            label: "搜尋或活動分類".to_string(),
            description: Some("HARNESS_REWRITTEN_DESC".to_string()),
            reask: vec!["HARNESS_CUSTOM_REASK_PROMPT".to_string()],
        }],
        candidate_order: vec![
            CandidateKind::Reask,
            CandidateKind::Experience,
            CandidateKind::Zoom,
            CandidateKind::Ocr,
        ],
        max_extra_requests_per_step: 4,
    };

    let backend = MockBackend::new(vec![
        // Initial box query: low confidence (0.11)
        make_box_response(0.30, 0.30, 0.40, 0.40, 0.11, 0.01),
        // Legacy non-card re-ask '<label> (<description>)': high confidence (0.88)
        make_box_response(0.45, 0.40, 0.55, 0.60, 0.88, 0.01),
    ]);

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_harness(harness)
        .with_self_fix(false);

    let mut audit_log = AuditLog::default();
    let (_bbox, box_conf, _sens, reasked) = decider
        .resolve_click_box(
            "Goal: find event",
            &prepared,
            "搜尋或活動分類",
            Some((&mut audit_log, 0)),
        )
        .unwrap();

    assert!(reasked);
    assert!((box_conf - 0.88).abs() < 1e-6);
    assert!(decider.last_self_fix().is_none());
    assert!(
        !audit_log.records().iter().any(|r| r.purpose.contains("2x")),
        "Non-card legacy re-ask must not trigger content 2x zoom"
    );

    let sent_reqs = decider.backend().requests.borrow();
    assert_eq!(sent_reqs.len(), 2);
    if let Some(Question::Box { instructions }) = sent_reqs[0].questions.get("target_box") {
        assert_eq!(instructions, "搜尋或活動分類");
    } else {
        panic!("Expected initial target_box question in req 0");
    }
    if let Some(Question::Box { instructions }) = sent_reqs[1].questions.get("target_box") {
        assert_eq!(
            instructions,
            "搜尋或活動分類 (首頁的活動分類或「近期活動」連結)"
        );
    } else {
        panic!("Expected '<label> (<description>)' legacy re-ask in req 1");
    }
}
