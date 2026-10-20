use decision_cu::agent::{Agent, AgentError, StopReason};
use decision_cu::api::{
    Answer, ApiError, ApiKey, BoundingBox, DecisionBackend, DecisionClient, Question, Request,
    Response, Usage, DEFAULT_MODEL,
};
use decision_cu::capture::{CaptureError, Rect, Screen};
use decision_cu::decider::{
    map_box_center_to_screen_pt, ActionKind, TargetSpec, VisionDecider,
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
}

#[derive(Default)]
struct MockInput {
    clicks: Vec<(f64, f64)>,
    typed: Vec<String>,
    keys: Vec<KeyCode>,
    scrolls: Vec<i32>,
    modifier_releases: usize,
}

impl Input for MockInput {
    fn click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.clicks.push((x, y));
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
    let client = DecisionClient::with_endpoint(key.clone(), "http://127.0.0.1:1/invalid").unwrap();
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
