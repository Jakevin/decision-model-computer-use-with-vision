use decision_cu::agent::{Agent, StopReason};
use decision_cu::api::{
    Answer, ApiError, BoundingBox, DecisionBackend, Question, Request, Response, Usage,
    DEFAULT_MODEL,
};
use decision_cu::browser::{Browser, BrowserError};
use decision_cu::capture::{CaptureError, Rect, Screen};
use decision_cu::decider::{
    map_box_center_to_screen_pt, DoneCondition, Preset, VisionDecider,
};
use decision_cu::input::{Input, InputError, KeyCode};
use decision_cu::logging::RunLogger;
use decision_cu::ocr::{
    filter_ocr_candidates, parse_vision_jsonl, vision_to_topleft, FakeOcr, OcrLine,
};
use decision_cu::privacy::prepare_image;
use decision_cu::safety::SafetyPolicy;
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
}

impl Input for MockInput {
    fn click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        self.clicks.push((x, y));
        Ok(())
    }
    fn move_mouse(&mut self, _x: f64, _y: f64) -> Result<(), InputError> {
        Ok(())
    }
    fn type_text(&mut self, _text: &str) -> Result<(), InputError> {
        Ok(())
    }
    fn key(&mut self, _key: KeyCode) -> Result<(), InputError> {
        Ok(())
    }
    fn scroll_vertical(&mut self, _amount: i32) -> Result<(), InputError> {
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
}

impl SequenceBrowser {
    fn new(urls: Vec<Option<String>>) -> Self {
        Self {
            urls: RefCell::new(VecDeque::from(urls)),
        }
    }
}

impl Browser for SequenceBrowser {
    fn active_tab_url(&self) -> Option<String> {
        self.urls.borrow_mut().pop_front().flatten()
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
    answers.insert("popup_present".to_string(), Answer::Noul { noul: 0.1 });
    answers.insert("footer".to_string(), Answer::Noul { noul: 0.1 });
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

fn make_price_box_response(
    top: f64,
    left: f64,
    bottom: f64,
    right: f64,
    confidence: f64,
) -> Response {
    let mut answers = BTreeMap::new();
    answers.insert(
        "price_box".to_string(),
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

fn count_target_box_requests(requests: &[Request]) -> usize {
    requests
        .iter()
        .filter(|r| {
            r.questions
                .get("target_box")
                .map(|q| matches!(q, Question::Box { .. }))
                .unwrap_or(false)
        })
        .count()
}

#[test]
fn f9_vision_y_flip_mapping_to_screen_pt_including_negative_origin_window() {
    let (top, left, bottom, right) = vision_to_topleft(0.25, 0.30, 0.50, 0.20);
    assert!((top - 0.50).abs() < 1e-9);
    assert!((left - 0.25).abs() < 1e-9);
    assert!((bottom - 0.70).abs() < 1e-9);
    assert!((right - 0.75).abs() < 1e-9);

    let line = OcrLine::from_vision("立即報名", 0.96, 0.25, 0.30, 0.50, 0.20);
    assert_eq!(line.region(), "center");

    let jsonl =
        r#"{"text":"2026 台北音樂節","confidence":0.95,"x":0.10,"y":0.55,"w":0.40,"h":0.10}"#;
    let parsed = parse_vision_jsonl(jsonl).unwrap();
    assert_eq!(parsed.len(), 1);
    assert!((parsed[0].top - 0.35).abs() < 1e-9);
    assert!((parsed[0].bottom - 0.45).abs() < 1e-9);

    let neg_rect = Rect {
        x: -1440.0,
        y: -300.0,
        width: 576.0,
        height: 360.0,
    };
    let img = RgbaImage::from_pixel(720, 450, Rgba([220, 220, 220, 255]));
    let prepared = prepare_image(&img, Some(neg_rect), 2.0).unwrap();
    let (x_pt, y_pt) = map_box_center_to_screen_pt(&line.bbox(), &prepared);
    assert!((x_pt - (-1152.0)).abs() < 1e-6);
    assert!((y_pt - (-84.0)).abs() < 1e-6);
}

#[test]
fn f9_choosing_ocr_item_clicks_center_without_calling_ask_box() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: -1000.0,
        y: -200.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    let card_line = OcrLine::new("2026 台北國際科技論壇", 0.94, 0.30, 0.10, 0.40, 0.50);
    let ocr = FakeOcr::new(vec![card_line]);

    let backend = MockBackend::new(vec![
        make_choice_response("click", "活動卡片", 0.2),
        make_item_choice_response("0", 0.93),
        make_choice_response("done", "nothing to click", 0.9),
    ]);

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);
    let browser = SequenceBrowser::new(vec![
        Some("https://www.accupass.com".to_string()),
        Some("https://www.accupass.com/event/251021".to_string()),
    ]);

    let temp_dir = std::env::temp_dir().join("decision_cu_test_f9_ocr_no_box");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let logger = RunLogger::new_in_dir(&temp_dir, None).unwrap();
    let steps_path = logger.path().to_path_buf();

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser)
        .with_ocr(ocr)
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com".to_string()))
        .with_logger(logger);

    let summary = agent.run("Click card via OCR anchor").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.clicks, 1);

    let reqs = agent.decider().backend().requests.borrow();
    assert_eq!(count_target_box_requests(&reqs), 0);

    let (click_x, click_y) = agent.input().clicks[0];
    assert!((click_x - (-760.0)).abs() < 1e-6);
    assert!((click_y - 10.0).abs() < 1e-6);

    let log_str = std::fs::read_to_string(&steps_path).unwrap();
    let entries: Vec<serde_json::Value> = log_str
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .collect();
    assert!(entries
        .iter()
        .any(|e| e.get("event").and_then(|v| v.as_str()) == Some("ocr")));
    assert!(entries
        .iter()
        .any(|e| e.get("click_source").and_then(|v| v.as_str()) == Some("ocr")));
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn f9_non_chrome_outside_content_occluded_and_low_conf_text_excluded() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 800.0,
        height: 600.0,
    };
    let prepared = prepare_image(&img, Some(window), 2.0).unwrap();

    let raw_lines = vec![
        OcrLine::new("https://www.accupass.com", 0.99, 0.02, 0.10, 0.08, 0.50),
        OcrLine::new("Outside Window", 0.95, 0.30, -0.10, 0.38, 0.20),
        OcrLine::new("模糊標題", 0.25, 0.30, 0.10, 0.38, 0.40),
        OcrLine::new("允許", 0.98, 0.70, 0.70, 0.78, 0.85),
        OcrLine::new("sk-vsr-secret", 0.91, 0.25, 0.10, 0.32, 0.35),
        OcrLine::new("AI 開發者大會", 0.93, 0.45, 0.15, 0.52, 0.45),
    ];

    let candidates = filter_ocr_candidates(&raw_lines, &prepared, window, 80.0, |pt| {
        !(pt.0 > 600.0 && pt.1 > 400.0)
    });

    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].text, "***");
    assert_eq!(candidates[1].text, "AI 開發者大會");
}

#[test]
fn f9_tried_ocr_item_not_reoffered_on_same_url() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    let card_a = OcrLine::new("無效卡片標題A", 0.95, 0.25, 0.10, 0.32, 0.40);
    let card_b = OcrLine::new("有效活動卡片B", 0.92, 0.45, 0.10, 0.52, 0.40);
    let ocr = FakeOcr::new(vec![card_a, card_b]);

    let backend = MockBackend::new(vec![
        make_choice_response("click", "活動卡片", 0.1),
        make_item_choice_response("0", 0.92),
        make_choice_response("click", "活動卡片", 0.2),
        make_item_choice_response("0", 0.94),
        make_choice_response("done", "nothing to click", 0.9),
    ]);

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);
    let browser = SequenceBrowser::new(vec![
        Some("https://www.accupass.com".to_string()),
        Some("https://www.accupass.com".to_string()),
        Some("https://www.accupass.com/event/999".to_string()),
    ]);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_browser(browser)
        .with_ocr(ocr)
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com".to_string()));

    let summary = agent.run("Try second card when first card is no-op").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.clicks, 2);
    assert_eq!(agent.decider().tried_ocr_items().len(), 1);

    let reqs = agent.decider().backend().requests.borrow();
    assert_eq!(count_target_box_requests(&reqs), 0);
    match reqs[3].questions.get("item").unwrap() {
        Question::Choice { criteria, .. } => {
            assert!(criteria.contains_key("0"));
            assert!(!criteria.contains_key("1"));
            assert!(criteria.get("0").unwrap().as_ref().unwrap().contains("有效活動卡片B"));
        }
        _ => panic!("Expected item Choice question in step 1"),
    }
}

#[test]
fn f9_login_ocr_lines_stop_blocked_login_required() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    let ocr = FakeOcr::new(vec![
        OcrLine::new("電子信箱", 0.96, 0.30, 0.25, 0.36, 0.45),
        OcrLine::new("密碼", 0.95, 0.42, 0.25, 0.48, 0.35),
    ]);

    let backend = MockBackend::new(vec![]);
    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_ocr(ocr)
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com/event/251021".to_string()));

    let summary = agent.run("Stop when OCR sees login form").unwrap();
    assert_eq!(
        summary.stop_reason,
        StopReason::SafetyBlocked("login required".to_string())
    );
    assert_eq!(summary.clicks, 0);
    assert_eq!(agent.decider().requests_sent(), 0);
}

#[test]
fn f9_ocr_item_none_falls_back_to_box() {
    let img = RgbaImage::from_pixel(1600, 1200, Rgba([240, 240, 240, 255]));
    let window = Rect {
        x: 100.0,
        y: 50.0,
        width: 800.0,
        height: 600.0,
    };
    let screen = MockScreen {
        img,
        rect: Some(window),
        scale: 2.0,
    };

    let ocr = FakeOcr::new(vec![OcrLine::new(
        "立即報名",
        0.95,
        0.60,
        0.70,
        0.68,
        0.90,
    )]);

    let backend = MockBackend::new(vec![
        make_choice_response("click", "立即報名按鈕", 0.2),
        make_item_choice_response("none", 0.85),
        make_box_response(0.60, 0.70, 0.70, 0.90, 0.91),
        make_price_box_response(0.40, 0.20, 0.50, 0.45, 0.10),
        make_choice_response("done", "nothing to click", 0.9),
    ]);

    let decider = VisionDecider::new(backend, Preset::EventsZh.default_targets())
        .with_preset(Preset::EventsZh)
        .with_done_condition(DoneCondition::CheckoutPage);

    let mut agent = Agent::new(screen, decider, MockInput::default(), SafetyPolicy::default())
        .with_ocr(ocr)
        .with_flights_preset(false)
        .with_start_url(Some("https://www.accupass.com/event/251021".to_string()));

    let summary = agent.run("Fallback to box when item choice is none").unwrap();
    assert_eq!(summary.stop_reason, StopReason::Done);
    assert_eq!(summary.clicks, 1);

    let reqs = agent.decider().backend().requests.borrow();
    assert_eq!(count_target_box_requests(&reqs), 1);
}
