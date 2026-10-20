use decision_cu::agent::{Agent, AgentError, StopReason};
use decision_cu::api::{
    Answer, ApiError, ApiKey, BoundingBox, DecisionBackend, DecisionClient, Question, Request,
    Response, Usage, DEFAULT_MODEL,
};
use decision_cu::capture::{CaptureError, Rect, Screen};
use decision_cu::decider::{
    events_preset_targets, events_zh_preset_targets, flights_preset_targets,
    generic_preset_targets, load_targets_file, map_box_center_to_screen_pt, ActionKind,
    DoneCondition, Preset, TargetSpec, VisionDecider, CHECKOUT_QUESTION_INSTRUCTIONS,
    CHECKOUT_ZH_QUESTION_INSTRUCTIONS, CLOSEISH_QUESTION_INSTRUCTIONS,
    POPUP_PRESENT_QUESTION_INSTRUCTIONS, SENSITIVE_QUESTION_INSTRUCTIONS,
};
use decision_cu::input::{Input, InputError, KeyCode};
use decision_cu::logging::{redact, RunLogger};
use decision_cu::privacy::{prepare_image, PrivacyError, PreparedImage};
use decision_cu::safety::{SafetyError, SafetyPolicy};
use image::{Rgba, RgbaImage};
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};

struct MockScreen {
    img: RgbaImage,
    rect: Option<Rect>,
    scale: f64,
}

impl Screen for MockScreen {
    fn capture(&self) -> Result<RgbaImage, CaptureError> {
        Ok(self.img.clone())
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

    fn request_count(&self) -> usize {
        self.requests.borrow().len()
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
    moved_to: Vec<(f64, f64)>,
    typed: Vec<String>,
    keys: Vec<KeyCode>,
    scrolls: Vec<i32>,
    scrolls_at: Vec<(f64, f64, i32)>,
    modifier_releases: usize,
    select_alls: usize,
}

impl Input for MockInput {
    fn click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.clicks.push((x, y));
        self.release_modifiers()
    }

    fn focus_click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.focus_clicks.push((x, y));
        self.release_modifiers()
    }

    fn move_mouse(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        self.moved_to.push((x, y));
        Ok(())
    }

    fn scroll_vertical_at(&mut self, x: f64, y: f64, amount: i32) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.moved_to.push((x, y));
        self.scrolls.push(amount);
        self.scrolls_at.push((x, y, amount));
        self.release_modifiers()
    }

    fn type_text(&mut self, text: &str) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.typed.push(text.to_string());
        self.release_modifiers()
    }

    fn key(&mut self, key: KeyCode) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.keys.push(key);
        self.release_modifiers()
    }

    fn scroll_vertical(&mut self, amount: i32) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.scrolls.push(amount);
        self.release_modifiers()
    }

    fn release_modifiers(&mut self) -> Result<(), InputError> {
        self.modifier_releases += 1;
        Ok(())
    }

    fn select_all(&mut self) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.select_alls += 1;
        self.release_modifiers()
    }
}

struct SequenceBrowser {
    urls: RefCell<VecDeque<Option<String>>>,
    set_urls: Vec<String>,
    activated: usize,
    back_calls: usize,
}

impl SequenceBrowser {
    fn new(urls: Vec<Option<String>>) -> Self {
        Self {
            urls: RefCell::new(VecDeque::from(urls)),
            set_urls: Vec::new(),
            activated: 0,
            back_calls: 0,
        }
    }
}

impl decision_cu::browser::Browser for SequenceBrowser {
    fn active_tab_url(&self) -> Option<String> {
        self.urls.borrow_mut().pop_front().flatten()
    }

    fn set_active_tab_url(&mut self, url: &str) -> Result<(), decision_cu::browser::BrowserError> {
        self.set_urls.push(url.to_string());
        Ok(())
    }

    fn activate_chrome(&mut self) -> bool {
        self.activated += 1;
        true
    }

    fn go_back(&mut self) -> Result<(), decision_cu::browser::BrowserError> {
        self.back_calls += 1;
        Ok(())
    }
}

fn make_choice_response_with_results(kind: &str, target: &str, results_noul: f64) -> Response {
    let mut resp = make_choice_response(kind, target);
    resp.answers.insert(
        "results".to_string(),
        Answer::Noul {
            noul: results_noul,
        },
    );
    resp
}

fn make_choice_response(kind: &str, target: &str) -> Response {
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
    Response {
        answers,
        model: DEFAULT_MODEL.to_string(),
        usage: Usage {
            input_tokens: 100,
            output_tokens: 0,
        },
    }
}

fn make_box_response(top: f64, left: f64, bottom: f64, right: f64, confidence: f64) -> Response {
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
fn no_crop_no_send() {
    let img = RgbaImage::from_pixel(800, 600, Rgba([240, 240, 240, 255]));
    assert_eq!(prepare_image(&img, None, 2.0), Err(PrivacyError::NoCrop));

    let screen = MockScreen {
        img,
        rect: None,
        scale: 2.0,
    };
    let backend = MockBackend::new(vec![]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", Some("Search flights".to_string()))],
    );
    let input = MockInput::default();
    let policy = SafetyPolicy::default();

    let mut agent = Agent::new(screen, decider, input, policy);
    let summary = agent.run("Find flights from SFO to JFK").unwrap();

    assert_eq!(summary.stop_reason, StopReason::NoCrop);
    assert_eq!(summary.steps_taken, 0);
    assert_eq!(agent.audit_log().records().len(), 0);
}

#[test]
fn crop_too_small_rejected() {
    let img = RgbaImage::from_pixel(800, 600, Rgba([240, 240, 240, 255]));
    let tiny_rect = Rect {
        x: 10.0,
        y: 10.0,
        width: 20.0,
        height: 20.0,
    };
    assert_eq!(
        prepare_image(&img, Some(tiny_rect), 1.0),
        Err(PrivacyError::CropTooSmall {
            width: 20,
            height: 20
        })
    );
}

#[test]
fn tab_strip_click_blocked() {
    let policy = SafetyPolicy::default();
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 1200.0,
        height: 800.0,
    };

    // y = 110.0 is < window.y + 80.0 (130.0) -> blocked!
    let res = policy.check_click((300.0, 110.0), window);
    assert!(matches!(res, Err(SafetyError::ClickInTabStrip { .. })));

    // y = 135.0 is >= 130.0 -> allowed!
    assert!(policy.check_click((300.0, 135.0), window).is_ok());

    // Also test in full agent loop
    let img = RgbaImage::from_pixel(3000, 2000, Rgba([255, 255, 255, 255]));
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };
    // Box near top of window (top=0.01, bottom=0.05 -> center_y = 0.03 * 800 = 24pt from window top -> y=74 < 130)
    let backend = MockBackend::new(vec![
        make_choice_response("click", "search button"),
        make_box_response(0.01, 0.2, 0.05, 0.4, 0.95),
    ]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", Some("Blue button".to_string()))],
    );
    let mut agent = Agent::new(screen, decider, MockInput::default(), policy);
    let err = agent.run("Click search").unwrap_err();
    assert!(matches!(
        err,
        AgentError::Safety(SafetyError::ClickInTabStrip { .. })
    ));
}

#[test]
fn key_never_in_logs() {
    let fake_key_str = "sk-vsr-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefG";
    assert_eq!(fake_key_str.len(), 7 + 43);

    let key = ApiKey::new(fake_key_str);

    // Debug & Display on ApiKey
    let dbg_key = format!("{:?}", key);
    let disp_key = format!("{}", key);
    assert_eq!(dbg_key, "ApiKey(***)");
    assert_eq!(disp_key, "ApiKey(***)");
    assert!(!dbg_key.contains(fake_key_str));
    assert!(!disp_key.contains(fake_key_str));

    // Debug on DecisionClient
    let client = DecisionClient::with_endpoint(key.clone(), "http://127.0.0.1:1/invalid")
        .unwrap()
        .with_transport_retry_delay(std::time::Duration::ZERO);
    let dbg_client = format!("{:?}", client);
    assert!(!dbg_client.contains(fake_key_str));
    assert!(dbg_client.contains("ApiKey(***)"));

    // Error string from transport failure
    let mut q = BTreeMap::new();
    q.insert(
        "k".to_string(),
        Question::Box {
            instructions: "find it".to_string(),
        },
    );
    let req = Request {
        model: DEFAULT_MODEL.to_string(),
        state: "test".to_string(),
        images: vec!["data:image/jpeg;base64,AA==".to_string()],
        questions: q,
    };
    let err = client.ask(&req).unwrap_err();
    let err_str = format!("{} / {:?}", err, err);
    assert!(!err_str.contains(fake_key_str));

    // Redact function on arbitrary strings and custom keys
    let raw_line = format!(
        "{{\"msg\":\"leaked {} and custom_secret_xyz\"}}",
        fake_key_str
    );
    let custom_key = ApiKey::new("custom_secret_xyz");
    let redacted = redact(&raw_line, Some(&custom_key));
    assert!(!redacted.contains(fake_key_str));
    assert!(!redacted.contains("custom_secret_xyz"));
    assert!(redacted.contains("***"));

    // RunLogger redaction test
    let temp_dir = std::env::temp_dir().join("decision_cu_test_logs");
    let mut logger = RunLogger::new_in_dir(&temp_dir, Some(key)).unwrap();
    let written = logger
        .log_entry(&serde_json::json!({
            "event": "test",
            "accidental_leak": fake_key_str,
        }))
        .unwrap();
    assert!(!written.contains(fake_key_str));
    let on_disk = std::fs::read_to_string(logger.path()).unwrap();
    assert!(!on_disk.contains(fake_key_str));
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn waits_limit() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([200, 200, 200, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };
    // 3 consecutive waits -> max_consecutive_waits is 2, so 3rd wait stops with MaxWaitsExceeded
    let backend = MockBackend::new(vec![
        make_choice_response("wait", "nothing to click"),
        make_choice_response("wait", "nothing to click"),
        make_choice_response("wait", "nothing to click"),
        make_choice_response("done", "nothing to click"),
    ]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", None)],
    );
    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default());
    let summary = agent.run("Search flights").unwrap();
    assert_eq!(summary.stop_reason, StopReason::MaxWaitsExceeded);
    assert_eq!(summary.steps_taken, 3);

    // After 2 waits (step 0 and step 1), step 2's kind criteria must omit "wait"
    let reqs = agent.decider().backend().requests.borrow();
    match reqs[0].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => assert!(criteria.contains_key("wait")),
        _ => panic!("Expected choice question"),
    }
    match reqs[1].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => assert!(criteria.contains_key("wait")),
        _ => panic!("Expected choice question"),
    }
    match reqs[2].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => assert!(!criteria.contains_key("wait")),
        _ => panic!("Expected choice question"),
    }
}

#[test]
fn box_centre_mapping() {
    let prepared = PreparedImage {
        data_url: "data:image/jpeg;base64,".to_string(),
        crop_origin_px: (200, 100),
        crop_size_px: (1600, 1200),
        sent_size_px: (800, 600),
        scale_factor: 2.0,
    };
    let bbox = BoundingBox {
        top: 0.25,
        left: 0.50,
        bottom: 0.75,
        right: 0.90,
    };
    // center_x_frac = 0.70, center_y_frac = 0.50
    // crop_x_px = 200 + 0.70 * 1600 = 200 + 1120 = 1320 px -> / 2.0 scale = 660.0 pt
    // crop_y_px = 100 + 0.50 * 1200 = 100 + 600 = 700 px -> / 2.0 scale = 350.0 pt
    let (pt_x, pt_y) = map_box_center_to_screen_pt(&bbox, &prepared);
    assert!((pt_x - 660.0).abs() < 1e-6);
    assert!((pt_y - 350.0).abs() < 1e-6);
}

#[test]
fn reask_when_confidence_below_threshold() {
    let prepared = PreparedImage {
        data_url: "data:image/jpeg;base64,AA==".to_string(),
        crop_origin_px: (0, 0),
        crop_size_px: (1000, 800),
        sent_size_px: (1000, 800),
        scale_factor: 2.0,
    };

    let backend = MockBackend::new(vec![
        make_choice_response("click", "search button"),
        // First box response has confidence 0.15 (< 0.2)
        make_box_response(0.1, 0.1, 0.2, 0.2, 0.15),
        // Second box response (re-ask) has confidence 0.92
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.92),
    ]);

    let targets = vec![TargetSpec::new(
        "search button",
        Some("The blue pill-shaped Search button".to_string()),
    )];
    let decider = VisionDecider::new(backend, targets);
    let decision = decider
        .decide("Find flights", &["clicked origin".to_string()], &prepared)
        .unwrap();

    assert_eq!(decision.kind, ActionKind::Click);
    assert!(decision.reasked);
    assert!((decision.confidence - 0.92).abs() < 1e-6);
    assert_eq!(decider.backend().request_count(), 3);

    // Check that the 3rd request used the longer description
    let reqs = decider.backend().requests.borrow();
    match reqs[2].questions.get("target_box").unwrap() {
        Question::Box { instructions } => {
            assert!(instructions.contains("The blue pill-shaped Search button"));
        }
        _ => panic!("Expected box question on re-ask"),
    }
}

#[test]
fn request_and_response_json_shape_matches_docs() {
    let mut criteria = BTreeMap::new();
    criteria.insert("click".to_string(), Some("Click an element".to_string()));
    criteria.insert("done".to_string(), None);

    let mut questions = BTreeMap::new();
    questions.insert(
        "kind".to_string(),
        Question::Choice {
            instructions: "What next?".to_string(),
            criteria,
        },
    );
    questions.insert(
        "find_btn".to_string(),
        Question::Box {
            instructions: "search button".to_string(),
        },
    );

    let req = Request {
        model: "vllm-sr/d3-turbo".to_string(),
        state: "Goal: test".to_string(),
        images: vec!["data:image/jpeg;base64,xyz".to_string()],
        questions,
    };

    let val = serde_json::to_value(&req).unwrap();
    assert_eq!(val["model"], "vllm-sr/d3-turbo");
    assert_eq!(val["state"], "Goal: test");
    assert_eq!(val["images"][0], "data:image/jpeg;base64,xyz");
    assert_eq!(val["questions"]["kind"]["type"], "choice");
    assert_eq!(val["questions"]["kind"]["instructions"], "What next?");
    assert_eq!(
        val["questions"]["kind"]["criteria"]["click"],
        "Click an element"
    );
    assert!(val["questions"]["kind"]["criteria"]["done"].is_null());
    assert_eq!(val["questions"]["find_btn"]["type"], "box");
    assert_eq!(
        val["questions"]["find_btn"]["instructions"],
        "search button"
    );

    // Verify response deserialization from exact documented JSON shape
    let raw_resp = r#"{
        "answers": {
            "kind": {
                "type": "choice",
                "choice": "click",
                "probabilities": {"click": 0.98, "done": 0.02},
                "confidence": 0.98
            },
            "find_btn": {
                "type": "box",
                "box": {"top": 0.1, "left": 0.2, "bottom": 0.3, "right": 0.4},
                "confidence": 0.87
            }
        },
        "model": "vllm-sr/d3-turbo",
        "usage": {"input_tokens": 512, "output_tokens": 0}
    }"#;

    let parsed: Response = serde_json::from_str(raw_resp).unwrap();
    assert_eq!(parsed.model, "vllm-sr/d3-turbo");
    assert_eq!(parsed.usage.input_tokens, 512);
    assert_eq!(parsed.usage.output_tokens, 0);

    match parsed.answers.get("kind").unwrap() {
        Answer::Choice {
            choice, confidence, ..
        } => {
            assert_eq!(choice, "click");
            assert!((*confidence - 0.98).abs() < 1e-6);
        }
        _ => panic!("Expected choice answer"),
    }

    match parsed.answers.get("find_btn").unwrap() {
        Answer::Box { bbox, confidence } => {
            assert!((bbox.top - 0.1).abs() < 1e-6);
            assert!((bbox.left - 0.2).abs() < 1e-6);
            assert!((bbox.bottom - 0.3).abs() < 1e-6);
            assert!((bbox.right - 0.4).abs() < 1e-6);
            assert!((*confidence - 0.87).abs() < 1e-6);
        }
        _ => panic!("Expected box answer"),
    }
}

#[test]
fn url_off_prefix_nothing_sent_and_tab_event() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };
    let backend = MockBackend::new(vec![]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", None)],
    );
    let browser = SequenceBrowser::new(vec![
        Some("https://example.com/other".to_string()),
        Some("https://news.ycombinator.com".to_string()),
        Some("chrome://newtab".to_string()),
    ]);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_tab_events");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();
    let summary_path = logger.summary_path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser)
        .with_start_url(Some("https://www.google.com/travel/flights".to_string()))
        .with_start_url_wait(std::time::Duration::ZERO)
        .with_logger(logger);

    let summary = agent.run("Find flights").unwrap();
    assert!(matches!(summary.stop_reason, StopReason::SafetyBlocked(_)));
    assert_eq!(summary.steps_taken, 0);
    assert_eq!(summary.requests_sent, 0);
    assert_eq!(summary.images_sent, 0);
    assert_eq!(summary.tab_events, 3);
    assert_eq!(agent.decider().backend().request_count(), 0);
    assert_eq!(agent.audit_log().records().len(), 0);
    assert_eq!(agent.browser().set_urls.len(), 4);
    assert!(agent.browser().activated >= 1);

    let steps_content = std::fs::read_to_string(&steps_path).unwrap();
    assert!(steps_content.contains("\"tab_event\""));
    let summary_content = std::fs::read_to_string(&summary_path).unwrap();
    let summary_json: serde_json::Value = serde_json::from_str(&summary_content).unwrap();
    assert_eq!(summary_json["status"], "blocked");
    assert_eq!(summary_json["tab_events"], 3);
    assert_eq!(summary_json["requests_sent"], 0);
    assert_eq!(summary_json["images_sent"], 0);
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn fill_typing_after_click_selects_all_and_types_in_same_step() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    let backend = MockBackend::new(vec![
        make_choice_response_with_results("click", "origin field", 0.1),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
        make_choice_response_with_results("done", "nothing to click", 0.6),
    ]);
    let mut fills = BTreeMap::new();
    fills.insert("origin field".to_string(), "Osaka".to_string());

    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("origin field", Some("Where from?".to_string()))],
    )
    .with_type_values(fills);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default());
    let summary = agent.run("Search one-way flights from Osaka").unwrap();

    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.steps_taken, 2);
    assert_eq!(summary.clicks, 1);
    assert_eq!(agent.input().clicks.len(), 1);
    assert_eq!(agent.input().select_alls, 1);
    assert_eq!(agent.input().typed, vec!["Osaka".to_string()]);
    assert!(agent.input().keys.is_empty());
    assert_eq!(
        summary.history[0],
        "clicked and typed 'Osaka' into origin field"
    );

    let reqs = agent.decider().backend().requests.borrow();
    match reqs[0].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("type"));
            assert!(criteria.contains_key("press enter"));
        }
        _ => panic!("Expected choice question for kind"),
    }
}

#[test]
fn results_stop_after_two_consecutive_and_done_requires_half_noul() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    let backend = MockBackend::new(vec![
        make_choice_response_with_results("done", "nothing to click", 0.2),
        make_choice_response_with_results("scroll", "nothing to click", 0.85),
        make_choice_response_with_results("wait", "nothing to click", 0.91),
    ]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", None)],
    );

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default());
    let summary = agent.run("Find flights").unwrap();

    assert_eq!(summary.stop_reason, StopReason::ResultsPageWithPrices);
    assert_eq!(summary.steps_taken, 3);
    let summary_json = summary.to_summary_json();
    assert_eq!(summary_json.status, "done");
    assert_eq!(summary_json.reason, "results page with prices");
}

#[test]
fn summary_counters_and_step_log_fields() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 50.0,
        y: 40.0,
        width: 700.0,
        height: 500.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    let backend = MockBackend::new(vec![
        make_choice_response_with_results("click", "search button", 0.1),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.9),
        make_choice_response_with_results("press enter", "nothing to click", 0.3),
        make_choice_response_with_results("done", "nothing to click", 0.7),
    ]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", None)],
    );

    let temp_dir = std::env::temp_dir().join("decision_cu_test_summary_counters");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();
    let summary_path = logger.summary_path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_logger(logger);
    let summary = agent.run("Search flights").unwrap();

    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.steps_taken, 3);
    assert_eq!(summary.clicks, 1);
    assert_eq!(summary.requests_sent, 4);
    assert_eq!(summary.images_sent, 4);
    assert!(summary.all_images_cropped);
    assert_eq!(summary.tab_events, 0);
    assert!(summary.avg_seconds_per_step >= 0.0);
    assert_eq!(agent.input().keys, vec![KeyCode::Return]);

    let summary_str = std::fs::read_to_string(&summary_path).unwrap();
    let s_val: serde_json::Value = serde_json::from_str(&summary_str).unwrap();
    assert_eq!(s_val["status"], "done");
    assert_eq!(s_val["steps"], 3);
    assert_eq!(s_val["clicks"], 1);
    assert_eq!(s_val["requests_sent"], 4);
    assert_eq!(s_val["images_sent"], 4);
    assert_eq!(s_val["all_images_cropped"], true);
    assert_eq!(s_val["tab_events"], 0);

    let steps_str = std::fs::read_to_string(&steps_path).unwrap();
    let lines: Vec<&str> = steps_str.lines().collect();
    assert_eq!(lines.len(), 3);
    let step0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(step0["step"], 0);
    assert_eq!(step0["url"], "https://www.google.com/travel/flights");
    assert_eq!(step0["kind"], "click");
    assert_eq!(step0["target"], "search button");
    assert!(step0["box_crop_px"].is_object());
    assert!(step0["box_confidence"].is_number());
    assert_eq!(step0["reasked"], false);
    assert!(step0["click_point"].is_array());
    assert!(step0["seconds"].is_number());
    assert_eq!(step0["results_noul"], 0.1);
    assert!(step0["sent_image_crop_rect"].is_object());
    assert_eq!(step0["sent_size"], serde_json::json!([1400, 1000]));
    assert_eq!(step0["full_screen_size"], serde_json::json!([1600, 1200]));

    let key = ApiKey::new("sk-vsr-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefG");
    let client = DecisionClient::with_endpoint(key, "http://127.0.0.1:1/invalid")
        .unwrap()
        .with_transport_retry_delay(std::time::Duration::ZERO);
    assert_eq!(client.requests_sent(), 0);
    let mut q = BTreeMap::new();
    q.insert(
        "results".to_string(),
        Question::Noul {
            instructions: "results?".to_string(),
        },
    );
    let req = Request {
        model: DEFAULT_MODEL.to_string(),
        state: "Goal: test".to_string(),
        images: vec![],
        questions: q,
    };
    let _ = client.ask(&req);
    assert_eq!(client.requests_sent(), 2);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

struct FrontmostSequenceBrowser {
    frontmost_seq: RefCell<VecDeque<String>>,
    activated: usize,
}

impl FrontmostSequenceBrowser {
    fn new(seq: Vec<&str>) -> Self {
        Self {
            frontmost_seq: RefCell::new(seq.into_iter().map(|s| s.to_string()).collect()),
            activated: 0,
        }
    }
}

impl decision_cu::browser::Browser for FrontmostSequenceBrowser {
    fn active_tab_url(&self) -> Option<String> {
        Some("https://www.google.com/travel/flights".to_string())
    }

    fn set_active_tab_url(&mut self, _url: &str) -> Result<(), decision_cu::browser::BrowserError> {
        Ok(())
    }

    fn activate_chrome(&mut self) -> bool {
        self.activated += 1;
        true
    }

    fn frontmost_app(&self) -> Option<String> {
        self.frontmost_seq.borrow_mut().pop_front()
    }
}

#[test]
fn not_frontmost_sends_nothing_and_stops_blocked() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };
    let backend = MockBackend::new(vec![]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", None)],
    );
    let browser = FrontmostSequenceBrowser::new(vec!["Terminal", "Finder"]);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser);
    let summary = agent.run("Find flights").unwrap();

    assert!(matches!(summary.stop_reason, StopReason::SafetyBlocked(_)));
    assert_eq!(summary.steps_taken, 0);
    assert_eq!(summary.requests_sent, 0);
    assert_eq!(summary.images_sent, 0);
    assert_eq!(agent.decider().backend().request_count(), 0);
    assert_eq!(agent.audit_log().records().len(), 0);
    assert_eq!(agent.browser().activated, 2);
}

#[test]
fn click_blocked_when_chrome_loses_frontmost() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };
    let backend = MockBackend::new(vec![
        make_choice_response_with_results("click", "search button", 0.1),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
    ]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", None)],
    );
    // Frontmost at start of step ("Google Chrome"), but loses frontmost right before click ("Slack")
    let browser = FrontmostSequenceBrowser::new(vec!["Google Chrome", "Slack"]);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser);
    let summary = agent.run("Click search").unwrap();
    assert_eq!(
        summary.stop_reason,
        StopReason::SafetyBlocked("Chrome occluded by Slack".to_string())
    );
    assert!(agent.input().clicks.is_empty());
}

struct OccludedScreen {
    img: RgbaImage,
    rect: Option<Rect>,
    scale: f64,
    topmost_app: String,
}

impl Screen for OccludedScreen {
    fn capture(&self) -> Result<RgbaImage, CaptureError> {
        Ok(self.img.clone())
    }

    fn chrome_window_rect(&self) -> Option<Rect> {
        self.rect
    }

    fn scale_factor(&self) -> f64 {
        self.scale
    }

    fn topmost_app_at(&self, _x_pt: f64, _y_pt: f64) -> Option<String> {
        Some(self.topmost_app.clone())
    }
}

#[test]
fn occluded_click_blocked_with_no_input_events_and_logs_occluded() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = OccludedScreen {
        img,
        rect: Some(window),
        scale: 2.0,
        topmost_app: "iTerm2".to_string(),
    };
    let backend = MockBackend::new(vec![
        make_choice_response_with_results("click", "origin field", 0.1),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
    ]);
    let mut fills = BTreeMap::new();
    fills.insert("origin field".to_string(), "Osaka".to_string());
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("origin field", None)],
    )
    .with_type_values(fills);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_occluded");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_logger(logger);
    let summary = agent.run("Search flights").unwrap();

    assert_eq!(
        summary.stop_reason,
        StopReason::SafetyBlocked("Chrome occluded by iTerm2".to_string())
    );
    let s_json = summary.to_summary_json();
    assert_eq!(s_json.status, "blocked");
    assert_eq!(s_json.reason, "Chrome occluded by iTerm2");
    assert!(agent.input().clicks.is_empty());
    assert_eq!(agent.input().select_alls, 0);
    assert!(agent.input().typed.is_empty());
    assert!(agent.input().keys.is_empty());

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    assert!(log_str.contains("\"occluded\""));
    assert!(log_str.contains("\"iTerm2\""));
    let step_rec: serde_json::Value = log_str
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .find(|e| e.get("kind").is_some())
        .unwrap();
    assert_eq!(step_rec["executed"], false);
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn explore_url_goes_back_sends_nothing_and_adds_history_hint() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };
    // Only 1 response needed because step 0 is on /travel/explore (sends nothing, goes back)
    // and step 1 is back on /travel/flights
    let backend = MockBackend::new(vec![make_choice_response_with_results(
        "done",
        "nothing to click",
        0.7,
    )]);
    let decider = VisionDecider::new(
        backend,
        vec![TargetSpec::new("search button", None)],
    );
    let browser = SequenceBrowser::new(vec![
        Some("https://www.google.com/travel/explore?tfs=abc".to_string()),
        Some("https://www.google.com/travel/flights".to_string()),
    ]);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_explore_back");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser)
        .with_logger(logger);
    let summary = agent.run("Find flights").unwrap();

    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.requests_sent, 1);
    assert_eq!(summary.images_sent, 1);
    assert_eq!(agent.browser().back_calls, 1);
    assert_eq!(
        summary.history[0],
        "search went to Explore page; set destination and date before searching"
    );

    let reqs = agent.decider().backend().requests.borrow();
    assert_eq!(reqs.len(), 1);
    assert!(reqs[0]
        .state
        .contains("search went to Explore page; set destination and date before searching"));

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    assert!(log_str.contains("\"explore_back\""));
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn repeated_target_excluded_after_two_consecutive_steps_and_dropdown_reclicked_when_unchanged() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    // Step 0: click "trip type dropdown" (req 0 = choice, req 1 = box) -> screen unchanged so re-clicks once
    // Step 1: click "trip type dropdown" again (req 2 = choice -> NOT excluded yet after 1 step; req 3 = box)
    // Step 2: click "one way option" (req 4 = choice -> MUST exclude "trip type dropdown" after 2 consecutive steps; req 5 = box)
    // Step 3: done (req 6 = choice -> "trip type dropdown" no longer excluded since step 2 clicked "one way option")
    let backend = MockBackend::new(vec![
        make_choice_response_with_results("click", "trip type dropdown", 0.1),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
        make_choice_response_with_results("click", "trip type dropdown", 0.1),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
        make_choice_response_with_results("click", "one way option", 0.1),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
        make_choice_response_with_results("done", "nothing to click", 0.7),
    ]);

    let decider = VisionDecider::new(
        backend,
        vec![
            TargetSpec::new("trip type dropdown", None),
            TargetSpec::new("one way option", None),
            TargetSpec::new("origin field", None),
        ],
    );

    let temp_dir = std::env::temp_dir().join("decision_cu_test_reclick");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_logger(logger);
    let summary = agent.run("Select one way").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);

    // Step 0 and Step 1 clicked "trip type dropdown" on unchanged MockScreen, so each re-clicked once (2 + 2 + 1 = 5 clicks total)
    assert_eq!(summary.clicks, 5);
    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    assert!(log_str.contains("\"reclick_unchanged\""));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let reqs = agent.decider().backend().requests.borrow();

    // Step 1 choice request (reqs[2]): after only 1 click, "trip type dropdown" is still allowed
    assert!(reqs[2]
        .state
        .contains("Targets already clicked: trip type dropdown"));
    match reqs[2].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(criteria.contains_key("trip type dropdown"));
            assert!(criteria.contains_key("one way option"));
        }
        _ => panic!("Expected target choice question"),
    }

    // Step 2 choice request (reqs[4]): after 2 consecutive clicks on "trip type dropdown", it is excluded!
    assert!(reqs[4]
        .state
        .contains("Targets already clicked: trip type dropdown -> trip type dropdown"));
    match reqs[4].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("trip type dropdown"));
            assert!(criteria.contains_key("one way option"));
            assert!(criteria.contains_key("origin field"));
        }
        _ => panic!("Expected target choice question"),
    }

    // Step 3 choice request (reqs[6]): after clicking "one way option" at step 2, last 2 steps are not the same target
    match reqs[6].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(criteria.contains_key("trip type dropdown"));
            assert!(criteria.contains_key("one way option"));
        }
        _ => panic!("Expected target choice question"),
    }
}

#[test]
fn events_and_generic_preset_target_criteria_and_targets_file() {
    let prepared = PreparedImage {
        data_url: "data:image/jpeg;base64,AA==".to_string(),
        crop_origin_px: (0, 0),
        crop_size_px: (1000, 800),
        sent_size_px: (1000, 800),
        scale_factor: 2.0,
    };

    let events_targets = Preset::Events.default_targets();
    assert_eq!(events_targets, events_preset_targets());
    let backend = MockBackend::new(vec![
        make_choice_response_with_results("click", "events menu", 0.1),
        make_box_response(0.1, 0.2, 0.2, 0.3, 0.9),
    ]);
    let decider = VisionDecider::new(backend, events_targets);
    let decision = decider
        .decide("Find next TechCrunch event tickets", &[], &prepared)
        .unwrap();
    assert_eq!(decision.kind, ActionKind::Click);
    assert_eq!(decision.target.as_deref(), Some("events menu"));

    let reqs = decider.backend().requests.borrow();
    match reqs[0].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert_eq!(
                criteria.get("events menu").unwrap().as_deref(),
                Some("the 'Events' link in the site header/navigation")
            );
            assert_eq!(
                criteria.get("upcoming event card").unwrap().as_deref(),
                Some("the card or link for the next upcoming event, nearest future date")
            );
            assert_eq!(
                criteria.get("tickets link").unwrap().as_deref(),
                Some("a 'Tickets', 'Buy tickets', 'Get tickets' or 'Register' button/link for that event")
            );
            assert_eq!(
                criteria.get("cheapest ticket option").unwrap().as_deref(),
                Some("the lowest-priced ticket type row or its select/add button")
            );
            assert_eq!(
                criteria.get("quantity increase").unwrap().as_deref(),
                Some("the '+' button to set ticket quantity to 1")
            );
            assert_eq!(
                criteria.get("checkout button").unwrap().as_deref(),
                Some("a 'Checkout', 'Continue' or 'Next' button that proceeds to checkout")
            );
            // Popup close is not offered in target choices unless popup_present noul confirmed a popup
            assert!(!criteria.contains_key("popup close"));
            assert!(criteria.contains_key("nothing to click"));
        }
        _ => panic!("Expected target choice question"),
    }

    let generic_targets = Preset::Generic.default_targets();
    assert_eq!(generic_targets, generic_preset_targets());
    let generic_labels: Vec<&str> = generic_targets.iter().map(|t| t.label.as_str()).collect();
    assert_eq!(
        generic_labels,
        vec![
            "navigation link toward the goal",
            "primary call-to-action button toward the goal",
            "popup close",
        ]
    );

    assert_eq!(Preset::Flights.default_targets(), flights_preset_targets());

    let temp_path = std::env::temp_dir().join("decision_cu_privacy_targets_file.json");
    std::fs::write(
        &temp_path,
        r#"[{"label":"register button","description":"main register CTA"}]"#,
    )
    .unwrap();
    let from_file = load_targets_file(&temp_path).unwrap();
    assert_eq!(
        from_file,
        vec![TargetSpec::new(
            "register button",
            Some("main register CTA".to_string())
        )]
    );
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn checkout_page_done_after_two_consecutive_and_logs_noul_and_done_condition() {
    assert_eq!(
        Preset::Flights.default_done_condition(),
        DoneCondition::FlightsResults
    );
    assert_eq!(
        Preset::Events.default_done_condition(),
        DoneCondition::CheckoutPage
    );
    assert_eq!(
        Preset::Generic.default_done_condition(),
        DoneCondition::Model
    );
    assert_eq!(
        DoneCondition::from_name("flights-results").unwrap(),
        DoneCondition::FlightsResults
    );
    assert_eq!(
        DoneCondition::from_name("checkout-page").unwrap(),
        DoneCondition::CheckoutPage
    );
    assert_eq!(
        DoneCondition::from_name("model").unwrap(),
        DoneCondition::Model
    );
    assert!(DoneCondition::from_name("invalid").is_err());

    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    // Step 0: checkout noul 0.85 (1st high) -> clicks events menu
    // Step 1: checkout noul 0.40 (resets consecutive count) -> clicks upcoming event card
    // Step 2: checkout noul 0.82 (1st high) -> clicks tickets link
    // Step 3: checkout noul 0.91 (2nd consecutive >= 0.8) -> done immediately!
    let backend = MockBackend::new(vec![
        make_choice_response_with_results("click", "events menu", 0.85),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
        make_choice_response_with_results("click", "upcoming event card", 0.40),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
        make_choice_response_with_results("click", "tickets link", 0.82),
        make_box_response(0.4, 0.4, 0.6, 0.6, 0.95),
        make_choice_response_with_results("wait", "nothing to click", 0.91),
    ]);
    let decider = VisionDecider::new(backend, Preset::Events.default_targets())
        .with_done_condition(DoneCondition::CheckoutPage);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_checkout_done");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();
    let summary_path = logger.summary_path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_flights_preset(false)
        .with_start_url(Some("https://www.google.com/travel/flights".to_string()))
        .with_logger(logger);

    let summary = agent.run("Find event checkout").unwrap();
    assert_eq!(summary.stop_reason, StopReason::CheckoutPageWithPrices);
    assert_eq!(summary.done_condition, "checkout-page");
    assert_eq!(summary.steps_taken, 4);
    assert_eq!(summary.clicks, 3);

    // Verify checkout noul question instructions were sent on each step
    let reqs = agent.decider().backend().requests.borrow();
    match reqs[0].questions.get("results").unwrap() {
        Question::Noul { instructions } => {
            assert_eq!(instructions, CHECKOUT_QUESTION_INSTRUCTIONS);
        }
        _ => panic!("Expected noul question"),
    }

    // Verify step log records the noul each step and summary has done_condition
    let steps_str = std::fs::read_to_string(&steps_path).unwrap();
    let lines: Vec<&str> = steps_str.lines().collect();
    assert_eq!(lines.len(), 4);
    let s0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    let s1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    let s2: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
    let s3: serde_json::Value = serde_json::from_str(lines[3]).unwrap();
    assert_eq!(s0["results_noul"], 0.85);
    assert_eq!(s1["results_noul"], 0.40);
    assert_eq!(s2["results_noul"], 0.82);
    assert_eq!(s3["results_noul"], 0.91);
    assert_eq!(s3["executed"], false);

    let summary_str = std::fs::read_to_string(&summary_path).unwrap();
    let summary_val: serde_json::Value = serde_json::from_str(&summary_str).unwrap();
    assert_eq!(summary_val["status"], "done");
    assert_eq!(summary_val["done_condition"], "checkout-page");
    let _ = std::fs::remove_dir_all(&temp_dir);
}

fn make_box_response_with_sensitive(
    top: f64,
    left: f64,
    bottom: f64,
    right: f64,
    confidence: f64,
    sensitive_noul: f64,
) -> Response {
    let mut resp = make_box_response(top, left, bottom, right, confidence);
    resp.answers.insert(
        "sensitive".to_string(),
        Answer::Noul {
            noul: sensitive_noul,
        },
    );
    resp
}

#[test]
fn sensitive_target_blocked_logged_and_excluded_for_next_three_steps() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    // Step 0: chooses click "primary call-to-action button toward the goal", box returns sensitive_noul = 0.72 (>= 0.5)
    //         -> do not click, log 'blocked_sensitive', exclude target for next 3 steps (steps 1, 2, 3).
    // Step 1: chooses click "navigation link toward the goal", box returns sensitive_noul = 0.05 (< 0.5) -> clicks
    // Step 2: chooses scroll
    // Step 3: chooses wait
    // Step 4: after 3 steps, "primary call-to-action button toward the goal" is included again -> done
    let backend = MockBackend::new(vec![
        make_choice_response("click", "primary call-to-action button toward the goal"),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.72),
        make_choice_response("click", "navigation link toward the goal"),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.05),
        make_choice_response("scroll", "nothing to click"),
        make_choice_response("wait", "nothing to click"),
        make_choice_response("done", "nothing to click"),
    ]);

    let decider = VisionDecider::new(backend, Preset::Generic.default_targets())
        .with_done_condition(DoneCondition::Model);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_blocked_sensitive");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_logger(logger);
    let summary = agent.run("Navigate to event page").unwrap();

    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.steps_taken, 5);
    assert_eq!(summary.clicks, 1);
    assert_eq!(agent.input().clicks.len(), 1);

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    assert!(log_str.contains("\"blocked_sensitive\""));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let reqs = agent.decider().backend().requests.borrow();
    // reqs[1] is the box request for step 0: verify "sensitive" noul question in SAME request
    match reqs[1].questions.get("sensitive").unwrap() {
        Question::Noul { instructions } => {
            assert_eq!(instructions, SENSITIVE_QUESTION_INSTRUCTIONS);
        }
        _ => panic!("Expected sensitive noul question in box request"),
    }
    assert!(reqs[1].questions.contains_key("target_box"));

    // reqs[2] (step 1), reqs[4] (step 2), reqs[5] (step 3): target must be excluded!
    for &req_idx in &[2usize, 4, 5] {
        match reqs[req_idx].questions.get("target").unwrap() {
            Question::Choice { criteria, .. } => {
                assert!(
                    !criteria.contains_key("primary call-to-action button toward the goal"),
                    "Target should be excluded at request index {}",
                    req_idx
                );
            }
            _ => panic!("Expected target choice question"),
        }
    }

    // reqs[6] (step 4): after 3 steps, target is no longer excluded
    match reqs[6].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(criteria.contains_key("primary call-to-action button toward the goal"));
        }
        _ => panic!("Expected target choice question"),
    }
}

#[test]
fn checkout_button_allowed_at_most_once_per_run() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    // Step 0: click "checkout button" (1st time -> allowed and clicked)
    // Step 1: click "checkout button" (2nd time -> blocked, not clicked, logs 'checkout_once')
    // Step 2: done
    let backend = MockBackend::new(vec![
        make_choice_response_with_results("click", "checkout button", 0.2),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.1),
        make_choice_response_with_results("click", "checkout button", 0.3),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.1),
        make_choice_response_with_results("done", "nothing to click", 0.7),
    ]);

    let decider = VisionDecider::new(backend, Preset::Events.default_targets())
        .with_done_condition(DoneCondition::CheckoutPage);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_checkout_once");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_logger(logger);
    let summary = agent.run("Proceed to checkout").unwrap();

    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.steps_taken, 3);
    assert_eq!(summary.clicks, 1);
    assert_eq!(agent.input().clicks.len(), 1);

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    assert!(log_str.contains("\"checkout_once\""));
    let _ = std::fs::remove_dir_all(&temp_dir);
}

fn make_box_response_with_closeish(
    top: f64,
    left: f64,
    bottom: f64,
    right: f64,
    confidence: f64,
    closeish_noul: f64,
) -> Response {
    let mut resp = make_box_response(top, left, bottom, right, confidence);
    resp.answers.insert(
        "closeish".to_string(),
        Answer::Noul {
            noul: closeish_noul,
        },
    );
    resp
}

fn make_choice_response_with_results_and_popup(
    kind: &str,
    target: &str,
    results_noul: f64,
    popup_present_noul: f64,
) -> Response {
    let mut resp = make_choice_response_with_results(kind, target, results_noul);
    resp.answers.insert(
        "popup_present".to_string(),
        Answer::Noul {
            noul: popup_present_noul,
        },
    );
    resp
}

#[test]
fn dismiss_popup_refused_when_closeish_low_and_clicked_when_high() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img: img.clone(),
        rect: Some(window),
        scale: 2.0,
    };

    // Flights preset must NOT include "dismiss popup" in kind criteria and must NOT include "popup_present"
    let flights_decider = VisionDecider::new(
        MockBackend::new(vec![make_choice_response_with_results(
            "done",
            "nothing to click",
            0.7,
        )]),
        Preset::Flights.default_targets(),
    )
    .with_preset(Preset::Flights);
    let prepared = prepare_image(&img, Some(window), 2.0).unwrap();
    let _ = flights_decider
        .decide("Search flights", &[], &prepared)
        .unwrap();
    let flights_reqs = flights_decider.backend().requests.borrow();
    match flights_reqs[0].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("dismiss popup"));
        }
        _ => panic!("Expected kind choice question"),
    }
    assert!(!flights_reqs[0].questions.contains_key("popup_present"));

    // Events preset:
    // Step 0: 1st request has "popup_present" noul, "dismiss popup" is NOT in kind criteria yet (no prev step);
    //         returns scroll + popup_present = 0.85 (>= 0.6)
    // Step 1: prev popup_present = 0.85 >= 0.6 -> "dismiss popup" IS in kind criteria;
    //         chooses "dismiss popup" (popup_present = 0.80 >= 0.6) -> box for "popup close" + closeish = 0.3 (< 0.6) -> refused, logs 'dismiss_refused'
    // Step 2: prev popup_present = 0.80 >= 0.6 -> "dismiss popup" IS in kind criteria;
    //         chooses "dismiss popup" (popup_present = 0.40 < 0.6) -> box for "popup close" + closeish = 0.9 (>= 0.6) -> refused because this step's popup_present < 0.6!
    // Step 3: prev popup_present = 0.40 < 0.6 -> "dismiss popup" is NOT in kind criteria;
    //         clicks "events menu" (popup_present = 0.90 >= 0.6) -> resets wait counter
    // Step 4: prev popup_present = 0.90 >= 0.6 -> "dismiss popup" IS in kind criteria;
    //         chooses "dismiss popup" (popup_present = 0.92 >= 0.6) -> box for "popup close" + closeish = 0.9 (>= 0.6) -> clicked!
    // Step 5: done
    let backend = MockBackend::new(vec![
        // Step 0 (1 req)
        make_choice_response_with_results_and_popup("scroll", "nothing to click", 0.1, 0.85),
        // Step 1 (2 reqs): closeish = 0.3 < 0.6 -> refused
        make_choice_response_with_results_and_popup("dismiss popup", "nothing to click", 0.2, 0.80),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.95, 0.3),
        // Step 2 (2 reqs): popup_present = 0.40 < 0.6 (even though closeish = 0.9) -> refused
        make_choice_response_with_results_and_popup("dismiss popup", "nothing to click", 0.2, 0.40),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.95, 0.9),
        // Step 3 (2 reqs): prev popup_present = 0.40 < 0.6, so click "events menu", popup_present = 0.90
        make_choice_response_with_results_and_popup("click", "events menu", 0.2, 0.90),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.05),
        // Step 4 (2 reqs): prev popup_present = 0.90 >= 0.6, this step popup_present = 0.92 >= 0.6, closeish = 0.9 >= 0.6 -> executed!
        make_choice_response_with_results_and_popup("dismiss popup", "nothing to click", 0.2, 0.92),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.95, 0.9),
        // Step 5 (1 req)
        make_choice_response_with_results_and_popup("done", "nothing to click", 0.7, 0.1),
    ]);

    let decider = VisionDecider::new(backend, Preset::Events.default_targets())
        .with_preset(Preset::Events)
        .with_done_condition(DoneCondition::CheckoutPage);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_dismiss_popup");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_logger(logger);
    let summary = agent.run("Dismiss popup and proceed").unwrap();

    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.steps_taken, 6);
    // Clicked at step 3 ("events menu") and step 4 ("popup close") -> 2 clicks total
    assert_eq!(summary.clicks, 2);
    assert_eq!(agent.input().clicks.len(), 2);
    assert_eq!(summary.history[0], "scroll down");
    assert_eq!(summary.history[1], "wait");
    assert_eq!(summary.history[2], "wait");
    assert!(summary.history[3].starts_with("click 'events menu'"));
    assert!(summary.history[4].starts_with("click 'popup close'"));

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    let refused_count = log_str
        .lines()
        .filter(|l| l.contains("\"dismiss_refused\""))
        .count();
    assert_eq!(refused_count, 2);
    let _ = std::fs::remove_dir_all(&temp_dir);

    let reqs = agent.decider().backend().requests.borrow();
    // Step 0 first request (reqs[0]): has "popup_present" noul, does NOT have "dismiss popup" in kind criteria
    match reqs[0].questions.get("popup_present").unwrap() {
        Question::Noul { instructions } => {
            assert_eq!(instructions, POPUP_PRESENT_QUESTION_INSTRUCTIONS);
        }
        _ => panic!("Expected popup_present noul question"),
    }
    match reqs[0].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("dismiss popup"));
        }
        _ => panic!("Expected kind choice question"),
    }
    // Step 1 first request (reqs[1]): prev popup_present was 0.85 >= 0.6 -> "dismiss popup" IS in kind criteria
    match reqs[1].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => {
            assert_eq!(
                criteria.get("dismiss popup").unwrap().as_deref(),
                Some("Close a cookie, newsletter or promo popup")
            );
        }
        _ => panic!("Expected kind choice question"),
    }
    // Verify box request (reqs[2]) asks for "popup close" and "closeish" noul in the same request
    match reqs[2].questions.get("target_box").unwrap() {
        Question::Box { instructions } => {
            assert_eq!(instructions, "popup close");
        }
        _ => panic!("Expected target_box question"),
    }
    match reqs[2].questions.get("closeish").unwrap() {
        Question::Noul { instructions } => {
            assert_eq!(instructions, CLOSEISH_QUESTION_INSTRUCTIONS);
        }
        _ => panic!("Expected closeish noul question"),
    }
    // Step 3 first request (reqs[5]): prev popup_present was 0.40 < 0.6 -> "dismiss popup" is NOT in kind criteria
    match reqs[5].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("dismiss popup"));
        }
        _ => panic!("Expected kind choice question"),
    }
    // Step 4 first request (reqs[7]): prev popup_present was 0.90 >= 0.6 -> "dismiss popup" IS in kind criteria
    match reqs[7].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(criteria.contains_key("dismiss popup"));
        }
        _ => panic!("Expected kind choice question"),
    }
}

#[test]
fn low_box_conf_after_reask_does_not_click_logs_and_excludes_for_two_steps() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    // Step 0: click "events menu", 1st box conf 0.12 (< 0.2), re-ask conf 0.15 (< 0.2)
    //         -> do NOT click, log 'low_box_conf', exclude "events menu" for next 2 steps (steps 1 and 2), count as wait
    // Step 1: dismiss popup ("popup close"), 1st box conf 0.10 (< 0.2), re-ask conf 0.18 (< 0.2)
    //         -> do NOT click, log 'low_box_conf', exclude "popup close" for next 2 steps (steps 2 and 3), count as wait
    // Step 2: click "upcoming event card" (high conf 0.95) -> clicked! (resets consecutive waits)
    // Step 3: done ("events menu" is included again at step 3; "popup close" still excluded at step 3)
    let backend = MockBackend::new(vec![
        // Step 0 (3 requests: choice + box + re-ask box)
        make_choice_response_with_results("click", "events menu", 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.12, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.15, 0.05),
        // Step 1 (10 requests: choice + box + 6 specific re-asks + 2 corner crop asks)
        make_choice_response_with_results("dismiss popup", "nothing to click", 0.1),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.10, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.11, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.12, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.13, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.14, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.15, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.16, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.17, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.18, 0.9),
        // Step 2 (2 requests: choice + box)
        make_choice_response_with_results("click", "upcoming event card", 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.05),
        // Step 3 (1 request: choice)
        make_choice_response_with_results("done", "nothing to click", 0.7),
    ]);

    let decider = VisionDecider::new(backend, Preset::Events.default_targets())
        .with_preset(Preset::Events)
        .with_done_condition(DoneCondition::CheckoutPage);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_low_box_conf");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_logger(logger);
    let summary = agent.run("Find events").unwrap();

    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.steps_taken, 4);
    assert_eq!(summary.clicks, 1);
    assert_eq!(agent.input().clicks.len(), 1);
    assert_eq!(summary.history[0], "wait");
    assert_eq!(summary.history[1], "wait");
    assert!(summary.history[2].starts_with("click 'upcoming event card'"));

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    let entries: Vec<serde_json::Value> = log_str
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let low_conf_events: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|e| e.get("event").and_then(|v| v.as_str()) == Some("low_box_conf"))
        .collect();
    assert_eq!(low_conf_events.len(), 2);
    assert_eq!(low_conf_events[0]["target"], "events menu");
    assert!((low_conf_events[0]["confidence"].as_f64().unwrap() - 0.15).abs() < 1e-6);
    assert_eq!(low_conf_events[1]["target"], "popup close");
    assert!((low_conf_events[1]["confidence"].as_f64().unwrap() - 0.18).abs() < 1e-6);

    let step_records: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|e| e.get("kind").is_some())
        .collect();
    assert_eq!(step_records.len(), 4);
    assert_eq!(step_records[0]["executed"], false);
    assert_eq!(step_records[1]["executed"], false);
    assert_eq!(step_records[2]["executed"], true);
    assert_eq!(step_records[3]["executed"], false);
    let _ = std::fs::remove_dir_all(&temp_dir);

    let reqs = agent.decider().backend().requests.borrow();
    // Step 1 choice request is reqs[3]: "events menu" must be excluded
    match reqs[3].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("events menu"));
        }
        _ => panic!("Expected target choice question"),
    }
    // Step 2 choice request is reqs[13]: both "events menu" and "popup close" excluded,
    // and "wait" is omitted from kind criteria because steps 0 and 1 were wait-like non-actions!
    match reqs[13].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("events menu"));
            assert!(!criteria.contains_key("popup close"));
        }
        _ => panic!("Expected target choice question"),
    }
    match reqs[13].questions.get("kind").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("wait"));
        }
        _ => panic!("Expected kind choice question"),
    }
    // Step 3 choice request is reqs[15]: both "events menu" and "popup close" remain excluded for the rest of the run
    match reqs[15].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("events menu"));
            assert!(!criteria.contains_key("popup close"));
        }
        _ => panic!("Expected target choice question"),
    }
}

#[test]
fn summary_on_error_reports_done_condition_and_requests_sent_and_blocked_steps_have_executed_false() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 1200.0,
        height: 800.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    // Step 0: click "checkout button" -> allowed and executed (executed = true)
    // Step 1: click "checkout button" again -> blocked by checkout_once (executed = false)
    // Step 2: click "events menu" -> blocked by sensitive_noul = 0.8 (executed = false)
    // Step 3: dismiss popup -> blocked by closeish_noul = 0.2 (executed = false)
    // Step 4: click "tickets link", box response fails (no more mock responses -> Transport error after 10 requests sent!)
    let backend = MockBackend::new(vec![
        // Step 0 (2 reqs)
        make_choice_response_with_results("click", "checkout button", 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.05),
        // Step 1 (2 reqs)
        make_choice_response_with_results("click", "checkout button", 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.05),
        // Step 2 (2 reqs)
        make_choice_response_with_results("click", "events menu", 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.80),
        // Step 3 (2 reqs)
        make_choice_response_with_results("dismiss popup", "nothing to click", 0.1),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.95, 0.20),
        // Step 4 (1st req succeeds, 2nd req fails with Transport error)
        make_choice_response_with_results("click", "tickets link", 0.1),
    ]);

    let decider = VisionDecider::new(backend, Preset::Events.default_targets())
        .with_preset(Preset::Events)
        .with_done_condition(DoneCondition::CheckoutPage);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_error_summary_and_executed");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();
    let summary_path = logger.summary_path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_logger(logger);
    let err = agent.run("Buy event ticket").unwrap_err();
    assert!(matches!(err, AgentError::Decider(_)));
    assert_eq!(agent.decider().requests_sent(), 10);

    // Summary file written even on error exit must report effective done_condition ("checkout-page") and client counter (10)
    let summary_str = std::fs::read_to_string(&summary_path).unwrap();
    let summary_json: serde_json::Value = serde_json::from_str(&summary_str).unwrap();
    assert_eq!(summary_json["status"], "error");
    assert_eq!(summary_json["done_condition"], "checkout-page");
    assert_eq!(summary_json["requests_sent"], 10);

    // Verify step records' executed flags
    let steps_str = std::fs::read_to_string(&steps_path).unwrap();
    let step_records: Vec<serde_json::Value> = steps_str
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .filter(|e| e.get("kind").is_some())
        .collect();
    assert_eq!(step_records.len(), 4);
    assert_eq!(step_records[0]["executed"], true); // 1st checkout click executed
    assert_eq!(step_records[1]["executed"], false); // 2nd checkout click blocked (checkout_once)
    assert_eq!(step_records[2]["executed"], false); // blocked_sensitive
    assert_eq!(step_records[3]["executed"], false); // dismiss_refused

    let _ = std::fs::remove_dir_all(&temp_dir);
}

struct TabTrackingBrowser {
    urls: RefCell<VecDeque<Option<String>>>,
    tab_counts: RefCell<VecDeque<Option<usize>>>,
    window_ids: RefCell<VecDeque<Option<String>>>,
    set_urls: Vec<String>,
    back_calls: usize,
}

impl TabTrackingBrowser {
    fn new(
        urls: Vec<Option<String>>,
        tab_counts: Vec<Option<usize>>,
        window_ids: Vec<Option<String>>,
    ) -> Self {
        Self {
            urls: RefCell::new(VecDeque::from(urls)),
            tab_counts: RefCell::new(VecDeque::from(tab_counts)),
            window_ids: RefCell::new(VecDeque::from(window_ids)),
            set_urls: Vec::new(),
            back_calls: 0,
        }
    }
}

impl decision_cu::browser::Browser for TabTrackingBrowser {
    fn active_tab_url(&self) -> Option<String> {
        self.urls.borrow_mut().pop_front().flatten()
    }

    fn set_active_tab_url(&mut self, url: &str) -> Result<(), decision_cu::browser::BrowserError> {
        self.set_urls.push(url.to_string());
        Ok(())
    }

    fn tab_count(&self) -> Option<usize> {
        let mut q = self.tab_counts.borrow_mut();
        if q.len() > 1 {
            q.pop_front().flatten()
        } else {
            q.front().cloned().flatten()
        }
    }

    fn front_window_id(&self) -> Option<String> {
        let mut q = self.window_ids.borrow_mut();
        if q.len() > 1 {
            q.pop_front().flatten()
        } else {
            q.front().cloned().flatten()
        }
    }

    fn go_back(&mut self) -> Result<(), decision_cu::browser::BrowserError> {
        self.back_calls += 1;
        Ok(())
    }
}

#[test]
fn stops_on_new_tab_when_tab_count_increases_or_window_changes() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img: img.clone(),
        rect: Some(window),
        scale: 2.0,
    };

    // Case 1: tab_count increases from 2 at start -> 2 at step 0 -> 3 before step 1 capture
    let backend = MockBackend::new(vec![
        make_choice_response_with_results_and_popup("click", "events menu", 0.1, 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.05),
    ]);
    let decider = VisionDecider::new(backend, Preset::Events.default_targets())
        .with_preset(Preset::Events)
        .with_done_condition(DoneCondition::CheckoutPage);

    let browser = TabTrackingBrowser::new(
        vec![
            Some("https://example.com/events".to_string()),
            Some("https://tickets.partner.example/event-123".to_string()),
        ],
        vec![Some(2), Some(2), Some(3)],
        vec![Some("win-100".to_string())],
    );

    let temp_dir = std::env::temp_dir().join("decision_cu_test_new_tab_count");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();
    let summary_path = logger.summary_path().to_path_buf();

    let policy = SafetyPolicy::default()
        .with_allowed_prefixes(vec!["https://example.com".to_string()]);
    let mut agent = Agent::new(screen, decider, MockInput::default(), policy)
        .with_browser(browser)
        .with_flights_preset(false)
        .with_logger(logger);

    let summary = agent.run("Open event").unwrap();
    assert_eq!(
        summary.stop_reason,
        StopReason::SafetyBlocked("new tab opened".to_string())
    );
    assert_eq!(summary.steps_taken, 1);
    assert_eq!(summary.clicks, 1);
    // Must NOT navigate away or close anything
    assert!(agent.browser().set_urls.is_empty());
    assert_eq!(agent.browser().back_calls, 0);

    let summary_str = std::fs::read_to_string(&summary_path).unwrap();
    let summary_json: serde_json::Value = serde_json::from_str(&summary_str).unwrap();
    assert_eq!(summary_json["status"], "blocked");
    assert_eq!(summary_json["reason"], "new tab opened");

    let steps_str = std::fs::read_to_string(&steps_path).unwrap();
    let new_tab_events: Vec<serde_json::Value> = steps_str
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .filter(|e| e.get("event").and_then(|v| v.as_str()) == Some("new_tab"))
        .collect();
    assert_eq!(new_tab_events.len(), 1);
    assert_eq!(new_tab_events[0]["step"], 1);
    assert_eq!(
        new_tab_events[0]["url"],
        "https://tickets.partner.example/event-123"
    );
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Case 2: front_window_id changes at step 0 before capture -> nothing sent, stops blocked
    let screen2 = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };
    let decider2 = VisionDecider::new(MockBackend::new(vec![]), Preset::Generic.default_targets())
        .with_preset(Preset::Generic)
        .with_done_condition(DoneCondition::Model);
    let browser2 = TabTrackingBrowser::new(
        vec![Some("https://example.com/popup-window".to_string())],
        vec![Some(1), Some(1)],
        vec![Some("win-100".to_string()), Some("win-200".to_string())],
    );
    let policy2 = SafetyPolicy::default()
        .with_allowed_prefixes(vec!["https://example.com".to_string()]);
    let mut agent2 = Agent::new(screen2, decider2, MockInput::default(), policy2)
        .with_browser(browser2)
        .with_flights_preset(false);

    let summary2 = agent2.run("Browse site").unwrap();
    assert_eq!(
        summary2.stop_reason,
        StopReason::SafetyBlocked("new tab opened".to_string())
    );
    assert_eq!(summary2.steps_taken, 0);
    assert_eq!(summary2.requests_sent, 0);
}

#[test]
fn events_zh_preset_parsing_done_noul_checkout_once_popup_close_and_forbidden_words() {
    assert_eq!(Preset::from_name("events-zh").unwrap(), Preset::EventsZh);
    assert_eq!(Preset::EventsZh.as_str(), "events-zh");
    assert_eq!(
        Preset::EventsZh.default_done_condition(),
        DoneCondition::CheckoutPage
    );
    assert_eq!(Preset::EventsZh.checkout_once_label(), "下一步按鈕");
    assert_eq!(Preset::EventsZh.popup_close_label(), "關閉彈窗");

    let targets = Preset::EventsZh.default_targets();
    assert_eq!(targets, events_zh_preset_targets());
    let map: BTreeMap<&str, Option<&str>> = targets
        .iter()
        .map(|t| (t.label.as_str(), t.description.as_deref()))
        .collect();
    assert_eq!(
        map.get("活動卡片").copied().flatten(),
        Some("列表中顯示日期與標題的即將舉辦活動卡片")
    );
    assert_eq!(
        map.get("立即報名按鈕").copied().flatten(),
        Some("活動頁上的「立即報名」「購票」「報名」按鈕")
    );
    assert_eq!(
        map.get("票種選項").copied().flatten(),
        Some("票種列表中價格最低的那一列或其選擇按鈕")
    );
    assert_eq!(
        map.get("數量加號").copied().flatten(),
        Some("把票數設為 1 的「+」按鈕")
    );
    assert_eq!(
        map.get("下一步按鈕").copied().flatten(),
        Some("前往結帳的「下一步」「結帳」「確認」按鈕")
    );
    assert_eq!(
        map.get("關閉彈窗").copied().flatten(),
        Some("彈窗的「關閉」「X」「不用了」「拒絕」按鈕")
    );
    assert_eq!(
        map.get("搜尋或活動分類").copied().flatten(),
        Some("首頁的活動分類或「近期活動」連結")
    );

    // Forbidden Chinese words
    let policy = SafetyPolicy::default();
    for forbidden in ["登入", "註冊", "付款", "結帳完成", "送出訂單"] {
        assert!(
            policy.check_target(forbidden).is_err(),
            "Expected '{}' to be blocked",
            forbidden
        );
    }
    for t in &targets {
        assert!(
            policy.check_target(&t.label).is_ok(),
            "Expected preset target '{}' to be allowed",
            t.label
        );
    }

    // Verify events-zh done noul text, checkout-once for "下一步按鈕", and popup close for "關閉彈窗"
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    // Step 0: click "下一步按鈕" (1st time -> executed, popup_present = 0.9)
    // Step 1: click "下一步按鈕" (2nd time -> checkout_once blocked, popup_present = 0.9)
    // Step 2: dismiss popup -> queries "關閉彈窗" + closeish = 0.95 -> executed
    // Step 3: done
    let backend = MockBackend::new(vec![
        make_choice_response_with_results_and_popup("click", "下一步按鈕", 0.2, 0.9),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.05),
        make_choice_response_with_results_and_popup("click", "下一步按鈕", 0.3, 0.9),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.95, 0.05),
        make_choice_response_with_results_and_popup("dismiss popup", "nothing to click", 0.3, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.95, 0.95),
        make_choice_response_with_results_and_popup("done", "nothing to click", 0.7, 0.1),
    ]);

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(Preset::EventsZh.default_done_condition());

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default());
    let summary = agent.run("搜尋活動票券").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.steps_taken, 4);
    assert_eq!(summary.clicks, 2);
    assert!(summary.history[0].starts_with("click '下一步按鈕'"));
    assert_eq!(summary.history[1], "checkout_once '下一步按鈕'");
    assert!(summary.history[2].starts_with("click '關閉彈窗'"));

    let reqs = agent.decider().backend().requests.borrow();
    // Verify Chinese done noul text in 1st request
    match reqs[0].questions.get("results").unwrap() {
        Question::Noul { instructions } => {
            assert_eq!(instructions, CHECKOUT_ZH_QUESTION_INSTRUCTIONS);
            assert_eq!(instructions, "畫面是否為顯示票價的選擇票種或結帳頁面？");
        }
        _ => panic!("Expected results noul question"),
    }
    // Verify dismiss popup box query uses "關閉彈窗"
    match reqs[5].questions.get("target_box").unwrap() {
        Question::Box { instructions } => {
            assert_eq!(instructions, "關閉彈窗");
        }
        _ => panic!("Expected target_box question"),
    }
}

#[test]
fn retry_on_transport_error_once_and_no_transport_retry_on_http_status() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Barrier};

    let key = ApiKey::new("sk-vsr-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefG");
    let default_client = DecisionClient::new(key.clone()).unwrap();
    assert_eq!(
        default_client.timeout(),
        std::time::Duration::from_secs(decision_cu::api::DEFAULT_TIMEOUT_SECS)
    );
    assert_eq!(default_client.timeout(), std::time::Duration::from_secs(60));
    assert_eq!(
        default_client.transport_retry_delay(),
        std::time::Duration::from_secs(decision_cu::api::TRANSPORT_RETRY_DELAY_SECS)
    );
    assert_eq!(
        default_client.transport_retry_delay(),
        std::time::Duration::from_secs(2)
    );

    // Mock server 1: drops 1st TCP connection immediately (transport error), serves 200 OK on 2nd connection (retry once succeeds)
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let barrier_bg = Arc::clone(&barrier);

    let handle = std::thread::spawn(move || {
        barrier_bg.wait();
        // 1st connection: read start of request then close immediately without sending HTTP headers -> connection reset / unexpected EOF transport error
        if let Ok((mut stream1, _)) = listener.accept() {
            let mut buf = [0u8; 256];
            let _ = stream1.read(&mut buf);
            drop(stream1);
        }
        // 2nd connection: return valid 200 OK JSON response
        if let Ok((mut stream2, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let _ = stream2.read(&mut buf);
            let body = r#"{"answers":{"results":{"type":"noul","noul":0.88}},"model":"vllm-sr/d3-turbo","usage":{"input_tokens":10,"output_tokens":0}}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream2.write_all(resp.as_bytes());
            let _ = stream2.flush();
        }
    });

    barrier.wait();
    let client = DecisionClient::with_endpoint(key.clone(), format!("http://{}/v1/systemone", addr))
        .unwrap()
        .with_transport_retry_delay(std::time::Duration::from_millis(10));

    let mut q = BTreeMap::new();
    q.insert(
        "results".to_string(),
        Question::Noul {
            instructions: "test?".to_string(),
        },
    );
    let req = Request {
        model: DEFAULT_MODEL.to_string(),
        state: "Goal: test".to_string(),
        images: vec![],
        questions: q,
    };

    let resp = client.ask(&req).unwrap();
    assert_eq!(client.requests_sent(), 2);
    match resp.answers.get("results").unwrap() {
        Answer::Noul { noul } => assert!((*noul - 0.88).abs() < 1e-6),
        _ => panic!("Expected noul answer"),
    }
    let _ = handle.join();

    // Mock server 2: HTTP 400 status error must NOT be retried (only 1 request sent)
    let listener2 = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr2 = listener2.local_addr().unwrap();
    let handle2 = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener2.accept() {
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let body = r#"{"error":{"type":"invalid_request_error","code":"bad_request","message":"invalid field"}}"#;
            let resp = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });

    let client2 =
        DecisionClient::with_endpoint(key, format!("http://{}/v1/systemone", addr2))
            .unwrap()
            .with_transport_retry_delay(std::time::Duration::from_millis(10));
    let err = client2.ask(&req).unwrap_err();
    assert!(matches!(err, ApiError::RemoteError { status: 400, .. }));
    assert_eq!(client2.requests_sent(), 1);
    let _ = handle2.join();
}

#[test]
fn acc2_fixes_low_conf_exclusion_noop_cap_scroll_target_and_popup_target_gate() {
    use decision_cu::decider::compute_excluded_targets;

    // 1. Verify compute_excluded_targets is fed by low_box_conf events and excludes primary targets for 3 steps
    let step_events = vec![Some("low_box_conf:活動卡片".to_string())];
    assert_eq!(
        compute_excluded_targets(&step_events),
        vec!["活動卡片".to_string()]
    );
    let step_events_2 = vec![
        Some("low_box_conf:活動卡片".to_string()),
        None,
    ];
    assert_eq!(
        compute_excluded_targets(&step_events_2),
        vec!["活動卡片".to_string()]
    );
    let step_events_3 = vec![
        Some("low_box_conf:活動卡片".to_string()),
        None,
        None,
    ];
    assert_eq!(
        compute_excluded_targets(&step_events_3),
        vec!["活動卡片".to_string()]
    );
    let step_events_4 = vec![
        Some("low_box_conf:活動卡片".to_string()),
        None,
        None,
        None,
    ];
    assert!(compute_excluded_targets(&step_events_4).is_empty());

    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img: img.clone(),
        rect: Some(window),
        scale: 2.0,
    };

    let backend = MockBackend::new(vec![
        // Step 0: click "活動卡片" -> low_box_conf (popup_present = 0.2) (4 box asks: initial + 2 specific + content 2x crop)
        make_choice_response_with_results_and_popup("click", "活動卡片", 0.1, 0.2),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.11, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.12, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.13, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.14, 0.05),
        // Step 1: scroll with raw target "關閉彈窗" -> normalized to "向下捲動" (popup_present = 0.85)
        make_choice_response_with_results_and_popup("scroll", "關閉彈窗", 0.1, 0.85),
        // Step 2: wait (popup_present = 0.1)
        make_choice_response_with_results_and_popup("wait", "nothing to click", 0.1, 0.1),
        // Step 3: wait (popup_present = 0.1)
        make_choice_response_with_results_and_popup("wait", "nothing to click", 0.1, 0.1),
        // Step 4: click "搜尋或活動分類" -> low_box_conf (1st consecutive no-op; auxiliary target: 2 box asks)
        make_choice_response_with_results_and_popup("click", "搜尋或活動分類", 0.1, 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.10, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.12, 0.05),
        // Step 5: click "立即報名按鈕" -> low_box_conf (2nd consecutive no-op; primary target: 4 box asks)
        make_choice_response_with_results_and_popup("click", "立即報名按鈕", 0.1, 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.09, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.11, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.13, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.15, 0.05),
        // Step 6: click "票種選項" -> low_box_conf (3rd consecutive no-op -> blocked!; primary target: 4 box asks)
        make_choice_response_with_results_and_popup("click", "票種選項", 0.1, 0.1),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.08, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.10, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.11, 0.05),
        make_box_response_with_sensitive(0.4, 0.4, 0.6, 0.6, 0.13, 0.05),
    ]);

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_acc2_fixes");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();
    let summary_path = logger.summary_path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com".to_string()))
        .with_logger(logger);

    let summary = agent.run("Find Accupass event").unwrap();
    assert_eq!(
        summary.stop_reason,
        StopReason::SafetyBlocked("3 consecutive no-op steps (low box confidence)".to_string())
    );
    assert_eq!(summary.steps_taken, 7);
    assert_eq!(summary.clicks, 0);

    let summary_json = summary.to_summary_json();
    assert_eq!(summary_json.status, "blocked");
    assert_eq!(
        summary_json.reason,
        "3 consecutive no-op steps (low box confidence)"
    );

    let summary_file_str = std::fs::read_to_string(&summary_path).unwrap();
    let summary_file_val: serde_json::Value = serde_json::from_str(&summary_file_str).unwrap();
    assert_eq!(summary_file_val["status"], "blocked");
    assert_eq!(
        summary_file_val["reason"],
        "3 consecutive no-op steps (low box confidence)"
    );

    // Check step log records
    let steps_str = std::fs::read_to_string(&steps_path).unwrap();
    let step_records: Vec<serde_json::Value> = steps_str
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .filter(|e| e.get("kind").is_some())
        .collect();
    assert_eq!(step_records.len(), 7);
    // Step 1 was scroll: logged target must be "向下捲動", never "關閉彈窗"
    assert_eq!(step_records[1]["kind"], "scroll");
    assert_eq!(step_records[1]["target"], "向下捲動");
    assert_eq!(
        step_records[1]["decision"]["target"],
        serde_json::Value::String("向下捲動".to_string())
    );
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Inspect target choice questions sent to backend
    let reqs = agent.decider().backend().requests.borrow();
    // Step 0 choice request (reqs[0]): "關閉彈窗" NOT offered because popup_present not confirmed
    match reqs[0].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(criteria.contains_key("活動卡片"));
            assert!(!criteria.contains_key("關閉彈窗"));
        }
        _ => panic!("Expected target choice question"),
    }
    // Step 1 choice request (reqs[5]): "活動卡片" excluded (1st of 3 steps), "關閉彈窗" still NOT offered (step 0 popup_present = 0.2)
    match reqs[5].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("活動卡片"));
            assert!(!criteria.contains_key("關閉彈窗"));
        }
        _ => panic!("Expected target choice question"),
    }
    // Step 2 choice request (reqs[6]): "活動卡片" STILL excluded (2nd of 3 steps), "關閉彈窗" IS offered (step 1 popup_present = 0.85 >= 0.6)
    match reqs[6].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("活動卡片"));
            assert!(criteria.contains_key("關閉彈窗"));
        }
        _ => panic!("Expected target choice question"),
    }
    // Step 3 choice request (reqs[7]): "活動卡片" STILL excluded (3rd of 3 steps), "關閉彈窗" NOT offered (step 2 popup_present = 0.1 < 0.6)
    match reqs[7].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("活動卡片"));
            assert!(!criteria.contains_key("關閉彈窗"));
        }
        _ => panic!("Expected target choice question"),
    }
    // Step 4 choice request (reqs[8]): "活動卡片" is BACK in target choices after 3 excluded steps!
    match reqs[8].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(criteria.contains_key("活動卡片"));
        }
        _ => panic!("Expected target choice question"),
    }
    // Step 5 choice request (reqs[11]): "搜尋或活動分類" excluded (from step 4 low_box_conf)
    match reqs[11].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("搜尋或活動分類"));
        }
        _ => panic!("Expected target choice question"),
    }
    // Step 6 choice request (reqs[16]): both "搜尋或活動分類" (2nd step) and "立即報名按鈕" (1st step) excluded
    match reqs[16].questions.get("target").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(!criteria.contains_key("搜尋或活動分類"));
            assert!(!criteria.contains_key("立即報名按鈕"));
        }
        _ => panic!("Expected target choice question"),
    }
}

struct SequenceScreen {
    images: RefCell<VecDeque<RgbaImage>>,
    last_img: RgbaImage,
    rect: Option<Rect>,
    scale: f64,
}

impl SequenceScreen {
    fn new(images: Vec<RgbaImage>, rect: Option<Rect>, scale: f64) -> Self {
        let last_img = images
            .last()
            .cloned()
            .unwrap_or_else(|| RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255])));
        Self {
            images: RefCell::new(VecDeque::from(images)),
            last_img,
            rect,
            scale,
        }
    }
}

impl Screen for SequenceScreen {
    fn capture(&self) -> Result<RgbaImage, CaptureError> {
        Ok(self
            .images
            .borrow_mut()
            .pop_front()
            .unwrap_or_else(|| self.last_img.clone()))
    }

    fn chrome_window_rect(&self) -> Option<Rect> {
        self.rect
    }

    fn scale_factor(&self) -> f64 {
        self.scale
    }
}

fn make_changed_img(patch_color: [u8; 4]) -> RgbaImage {
    let mut img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    for y in 200..400 {
        for x in 200..500 {
            img.put_pixel(x, y, Rgba(patch_color));
        }
    }
    img
}

#[test]
fn f5_scroll_location_and_pagedown_fallback() {
    use decision_cu::input::DEFAULT_SCROLL_DELTA_PX;

    let base_img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let scrolled_img = make_changed_img([30, 30, 30, 255]);
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 800.0,
        height: 600.0,
    };

    let screen = SequenceScreen::new(
        vec![
            base_img.clone(),     // Step 0 initial capture
            scrolled_img.clone(), // Step 0 after wheel scroll (changed!)
            scrolled_img.clone(), // Step 1 initial capture
            scrolled_img.clone(), // Step 1 after wheel scroll (unchanged -> triggers PageDown fallback!)
            base_img.clone(),     // Step 1 after PageDown (changed!)
            base_img.clone(),     // Step 2 initial capture
        ],
        Some(window),
        2.0,
    );

    let backend = MockBackend::new(vec![
        make_choice_response_with_results("scroll", "nothing to click", 0.1),
        make_choice_response_with_results("scroll", "nothing to click", 0.2),
        make_choice_response_with_results("done", "nothing to click", 0.8),
    ]);
    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_f5_scroll");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com".to_string()))
        .with_logger(logger);

    let summary = agent.run("Scroll Accupass homepage").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.steps_taken, 3);

    assert_eq!(
        agent.input().scrolls_at,
        vec![
            (500.0, 390.0, DEFAULT_SCROLL_DELTA_PX),
            (500.0, 390.0, DEFAULT_SCROLL_DELTA_PX),
        ]
    );
    assert_eq!(DEFAULT_SCROLL_DELTA_PX, -400);

    assert_eq!(agent.input().focus_clicks, vec![(112.0, 390.0)]);
    assert_eq!(agent.input().keys, vec![KeyCode::PageDown]);

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    let scroll_events: Vec<serde_json::Value> = log_str
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .filter(|e| e.get("event").and_then(|v| v.as_str()) == Some("scroll"))
        .collect();
    assert_eq!(scroll_events.len(), 2);
    assert_eq!(scroll_events[0]["method"], "wheel");
    assert_eq!(scroll_events[0]["screen_changed"], true);
    assert_eq!(scroll_events[1]["method"], "pagedown_fallback");
    assert_eq!(scroll_events[1]["screen_changed"], true);
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn f5_noop_counting_and_stop_on_scroll_dismiss_alternation_and_rest_of_run_exclusion() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };

    let screen = MockScreen {
        img: img.clone(),
        rect: Some(window),
        scale: 2.0,
    };
    let backend = MockBackend::new(vec![
        make_choice_response_with_results_and_popup("scroll", "向下捲動", 0.1, 0.9),
        make_choice_response_with_results_and_popup("dismiss popup", "關閉彈窗", 0.1, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.05, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.06, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.07, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.08, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.09, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.10, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.11, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.12, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.14, 0.9),
        make_choice_response_with_results_and_popup("scroll", "向下捲動", 0.1, 0.9),
    ]);
    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com".to_string()));

    let summary = agent.run("Find Accupass event").unwrap();
    assert_eq!(
        summary.stop_reason,
        StopReason::SafetyBlocked("3 consecutive no-op steps (low box confidence)".to_string())
    );
    assert_eq!(summary.steps_taken, 3);
    assert_eq!(summary.to_summary_json().status, "blocked");

    let events_6_steps = vec![
        Some("low_box_conf:關閉彈窗".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    assert!(
        decision_cu::decider::compute_excluded_targets(&events_6_steps)
            .contains(&"關閉彈窗".to_string())
    );
}

#[test]
fn f5_popup_escape_then_specific_reask_then_corner_crop_mapping_and_box_width_guard() {
    use decision_cu::decider::{
        POPUP_REASK_AD_CLOSE_DESC, POPUP_REASK_COOKIE_DESC, POPUP_REASK_COOKIE_GOT_IT_DESC,
    };

    let base_img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let after_esc_img = make_changed_img([50, 180, 90, 255]);
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 800.0,
        height: 600.0,
    };

    let screen_esc = SequenceScreen::new(
        vec![
            base_img.clone(),
            after_esc_img.clone(),
            after_esc_img.clone(),
            base_img.clone(),
            base_img.clone(),
        ],
        Some(window),
        2.0,
    );
    let backend_esc = MockBackend::new(vec![
        make_choice_response_with_results_and_popup("scroll", "向下捲動", 0.1, 0.9),
        make_choice_response_with_results_and_popup("dismiss popup", "關閉彈窗", 0.2, 0.9),
        make_choice_response_with_results_and_popup("done", "nothing to click", 0.8, 0.1),
    ]);
    let decider_esc = VisionDecider::new(backend_esc, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);
    let mut agent_esc = Agent::new(
        screen_esc,
        decider_esc,
        MockInput::default(),
        SafetyPolicy::default(),
    )
    .with_flights_preset(false)
    .with_start_url(Some("https://www.accupass.com".to_string()));
    let summary_esc = agent_esc.run("Dismiss via Escape").unwrap();
    assert_eq!(summary_esc.stop_reason, StopReason::Done);
    assert_eq!(summary_esc.clicks, 0);
    assert!(agent_esc.input().keys.contains(&KeyCode::Escape));
    assert_eq!(agent_esc.decider().requests_sent(), 3);

    let screen_crop = SequenceScreen::new(
        vec![
            base_img.clone(),
            after_esc_img.clone(),
            after_esc_img.clone(),
            after_esc_img.clone(),
            base_img.clone(),
            base_img.clone(),
            base_img.clone(),
            base_img.clone(),
        ],
        Some(window),
        2.0,
    );
    let backend_crop = MockBackend::new(vec![
        make_choice_response_with_results_and_popup("scroll", "向下捲動", 0.1, 0.9),
        make_choice_response_with_results_and_popup("dismiss popup", "關閉彈窗", 0.2, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.10, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.11, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.11, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.11, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.11, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.12, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.15, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.18, 0.9),
        make_box_response_with_closeish(0.6, 0.1, 0.8, 0.2, 0.88, 0.95),
        make_choice_response_with_results_and_popup("dismiss popup", "關閉彈窗", 0.3, 0.9),
        make_box_response_with_closeish(0.7, 0.1, 0.9, 0.4, 0.92, 0.95),
        make_choice_response_with_results_and_popup("done", "nothing to click", 0.85, 0.1),
    ]);
    let decider_crop = VisionDecider::new(backend_crop, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);
    let mut agent_crop = Agent::new(
        screen_crop,
        decider_crop,
        MockInput::default(),
        SafetyPolicy::default(),
    )
    .with_flights_preset(false)
    .with_start_url(Some("https://www.accupass.com".to_string()));

    let summary_crop = agent_crop.run("Dismiss via crop and reject wide ad box").unwrap();
    assert_eq!(summary_crop.stop_reason, StopReason::Done);
    assert_eq!(summary_crop.clicks, 1);
    assert_eq!(agent_crop.input().clicks.len(), 1);
    let (cx, cy) = agent_crop.input().clicks[0];
    assert!((cx - 160.0).abs() < 1e-6, "expected cx=160.0, got {}", cx);
    assert!((cy - 560.0).abs() < 1e-6, "expected cy=560.0, got {}", cy);

    let reqs = agent_crop.decider().backend().requests.borrow();
    match reqs[7].questions.get("target_box").unwrap() {
        Question::Box { instructions } => assert_eq!(instructions, POPUP_REASK_COOKIE_DESC),
        _ => panic!("Expected box question"),
    }
    match reqs[8].questions.get("target_box").unwrap() {
        Question::Box { instructions } => assert_eq!(instructions, POPUP_REASK_AD_CLOSE_DESC),
        _ => panic!("Expected box question"),
    }
    match reqs[9].questions.get("target_box").unwrap() {
        Question::Box { instructions } => assert_eq!(instructions, POPUP_REASK_COOKIE_GOT_IT_DESC),
        _ => panic!("Expected box question"),
    }
    match reqs[10].questions.get("target_box").unwrap() {
        Question::Box { instructions } => assert_eq!(instructions, POPUP_REASK_AD_CLOSE_DESC),
        _ => panic!("Expected box question"),
    }
    let audit_purposes: Vec<&str> = agent_crop
        .audit_log()
        .records()
        .iter()
        .map(|r| r.purpose.as_str())
        .collect();
    assert!(audit_purposes.contains(&"step_1_popup_bottom_right_2x"));
    assert!(audit_purposes.contains(&"step_1_popup_bottom_left_2x"));
}

#[test]
fn f6_card_boxing_reask_content_crop_2x_mapping_wide_card_ok_and_short_primary_exclusion() {
    use decision_cu::decider::{
        CARD_REASK_FIRST_CARD_IMAGE_DESC, CARD_REASK_TITLE_TEXT_DESC,
        PRIMARY_LOW_BOX_CONF_EXCLUDE_STEPS,
    };

    let base_img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 800.0,
        height: 600.0,
    };

    let screen = MockScreen {
        img: base_img.clone(),
        rect: Some(window),
        scale: 2.0,
    };
    let backend = MockBackend::new(vec![
        make_choice_response_with_results_and_popup("click", "活動卡片", 0.3, 0.1),
        make_box_response(0.1, 0.2, 0.4, 0.5, 0.11),
        make_box_response(0.1, 0.2, 0.4, 0.5, 0.14),
        make_box_response(0.1, 0.2, 0.4, 0.5, 0.18),
        make_box_response(0.25, 0.10, 0.75, 0.40, 0.85),
        make_choice_response_with_results_and_popup("done", "nothing to click", 0.9, 0.1),
    ]);
    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);
    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_flights_preset(false)
        .with_start_url(Some("https://www.google.com/travel/flights".to_string()));

    let summary = agent.run("Click Accupass card via content crop").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.clicks, 1);
    assert_eq!(agent.input().clicks.len(), 1);

    let (cx, cy) = agent.input().clicks[0];
    assert!((cx - 300.0).abs() < 1e-4, "expected cx=300.0, got {}", cx);
    assert!((cy - 390.0).abs() < 1e-4, "expected cy=390.0, got {}", cy);

    let reqs = agent.decider().backend().requests.borrow();
    match reqs[1].questions.get("target_box").unwrap() {
        Question::Box { instructions } => assert_eq!(instructions, "活動卡片"),
        _ => panic!("Expected box question"),
    }
    match reqs[2].questions.get("target_box").unwrap() {
        Question::Box { instructions } => {
            assert_eq!(instructions, CARD_REASK_FIRST_CARD_IMAGE_DESC)
        }
        _ => panic!("Expected box question"),
    }
    match reqs[3].questions.get("target_box").unwrap() {
        Question::Box { instructions } => assert_eq!(instructions, CARD_REASK_TITLE_TEXT_DESC),
        _ => panic!("Expected box question"),
    }
    match reqs[4].questions.get("target_box").unwrap() {
        Question::Box { instructions } => {
            assert_eq!(instructions, CARD_REASK_FIRST_CARD_IMAGE_DESC)
        }
        _ => panic!("Expected box question"),
    }
    let audit_purposes: Vec<&str> = agent
        .audit_log()
        .records()
        .iter()
        .map(|r| r.purpose.as_str())
        .collect();
    assert!(audit_purposes.contains(&"step_0_content_2x"));

    assert_eq!(PRIMARY_LOW_BOX_CONF_EXCLUDE_STEPS, 3);
    let events_within_3 = vec![
        Some("low_box_conf:活動卡片".to_string()),
        None,
        None,
    ];
    assert!(
        decision_cu::decider::compute_excluded_targets(&events_within_3)
            .contains(&"活動卡片".to_string())
    );
    let events_after_4 = vec![
        Some("low_box_conf:活動卡片".to_string()),
        Some("low_box_conf:搜尋或活動分類".to_string()),
        None,
        None,
        None,
        None,
    ];
    let excluded_after_4 = decision_cu::decider::compute_excluded_targets(&events_after_4);
    assert!(
        !excluded_after_4.contains(&"活動卡片".to_string()),
        "Primary target 活動卡片 must re-enter available targets after 3 steps"
    );
    assert!(
        excluded_after_4.contains(&"搜尋或活動分類".to_string()),
        "Auxiliary target 搜尋或活動分類 may remain excluded for the rest of the run"
    );
}

#[test]
fn f6_cookie_banner_specific_descriptions_and_bottom_right_corner_crop() {
    use decision_cu::decider::{
        POPUP_REASK_ACCEPT_DESC, POPUP_REASK_AGREE_DESC, POPUP_REASK_COOKIE_CLOSE_X_DESC,
        POPUP_REASK_COOKIE_DESC, POPUP_REASK_COOKIE_GOT_IT_DESC,
    };

    let base_img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 800.0,
        height: 600.0,
    };

    let screen_desc = MockScreen {
        img: base_img.clone(),
        rect: Some(window),
        scale: 2.0,
    };
    let backend_desc = MockBackend::new(vec![
        make_choice_response_with_results_and_popup("scroll", "向下捲動", 0.1, 0.9),
        make_choice_response_with_results_and_popup("dismiss popup", "關閉彈窗", 0.2, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.10, 0.9),
        make_box_response_with_closeish(0.80, 0.80, 0.90, 0.90, 0.82, 0.95),
        make_choice_response_with_results_and_popup("done", "nothing to click", 0.9, 0.1),
    ]);
    let decider_desc = VisionDecider::new(backend_desc, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);
    let mut agent_desc = Agent::new(
        screen_desc,
        decider_desc,
        MockInput::default(),
        SafetyPolicy::default(),
    )
    .with_flights_preset(false)
    .with_start_url(Some("https://www.accupass.com".to_string()));

    let summary_desc = agent_desc.run("Dismiss cookie banner via got-it description").unwrap();
    assert_eq!(summary_desc.stop_reason, StopReason::Done);
    assert_eq!(summary_desc.clicks, 1);

    let screen_br = MockScreen {
        img: base_img,
        rect: Some(window),
        scale: 2.0,
    };
    let backend_br = MockBackend::new(vec![
        make_choice_response_with_results_and_popup("scroll", "向下捲動", 0.1, 0.9),
        make_choice_response_with_results_and_popup("dismiss popup", "關閉彈窗", 0.2, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.08, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.09, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.10, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.11, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.12, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.13, 0.9),
        make_box_response_with_closeish(0.4, 0.4, 0.5, 0.5, 0.14, 0.9),
        make_box_response_with_closeish(0.6, 0.5, 0.8, 0.7, 0.90, 0.95),
        make_choice_response_with_results_and_popup("done", "nothing to click", 0.9, 0.1),
    ]);
    let decider_br = VisionDecider::new(backend_br, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);
    let mut agent_br = Agent::new(
        screen_br,
        decider_br,
        MockInput::default(),
        SafetyPolicy::default(),
    )
    .with_flights_preset(false)
    .with_start_url(Some("https://www.accupass.com".to_string()));

    let summary_br = agent_br.run("Dismiss cookie banner via bottom-right corner crop").unwrap();
    assert_eq!(summary_br.stop_reason, StopReason::Done);
    assert_eq!(summary_br.clicks, 1);
    let (cx, cy) = agent_br.input().clicks[0];
    assert!((cx - 740.0).abs() < 1e-4, "expected cx=740.0, got {}", cx);
    assert!((cy - 560.0).abs() < 1e-4, "expected cy=560.0, got {}", cy);

    let reqs = agent_br.decider().backend().requests.borrow();
    let expected_descs = [
        "關閉彈窗",
        POPUP_REASK_COOKIE_GOT_IT_DESC,
        POPUP_REASK_ACCEPT_DESC,
        POPUP_REASK_AGREE_DESC,
        POPUP_REASK_COOKIE_CLOSE_X_DESC,
        POPUP_REASK_COOKIE_DESC,
    ];
    for (idx, expected) in expected_descs.iter().enumerate() {
        match reqs[2 + idx].questions.get("target_box").unwrap() {
            Question::Box { instructions } => assert_eq!(instructions, expected),
            _ => panic!("Expected box question at index {}", 2 + idx),
        }
    }
}

#[test]
fn f6_no_available_targets_wait_counts_as_noop_and_stops_blocked() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    let backend = MockBackend::new(vec![
        make_choice_response("click", "活動卡片"),
        make_box_response(0.2, 0.2, 0.4, 0.4, 0.05),
        make_box_response(0.2, 0.2, 0.4, 0.4, 0.08),
        make_box_response(0.2, 0.2, 0.4, 0.4, 0.10),
        make_box_response(0.2, 0.2, 0.4, 0.4, 0.12),
        make_choice_response("wait", "nothing to click"),
    ]);
    let decider = VisionDecider::new(
        backend,
        vec![decision_cu::decider::TargetSpec::new("活動卡片", None)],
    )
    .with_preset(Preset::EventsZh)
    .with_done_condition(DoneCondition::CheckoutPage);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_flights_preset(false)
        .with_start_url(Some("https://www.google.com/travel/flights".to_string()));

    let summary = agent.run("Stop when no available targets").unwrap();
    assert_eq!(
        summary.stop_reason,
        StopReason::SafetyBlocked("no available targets".to_string())
    );
    assert_eq!(summary.steps_taken, 2);
    assert_eq!(
        agent.policy().consecutive_waits(),
        0,
        "Wait when no available targets must count as no-op, not wait"
    );
    let json = summary.to_summary_json();
    assert_eq!(json.status, "blocked");
    assert_eq!(json.reason, "no available targets");
}




