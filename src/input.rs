use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyCode {
    Return,
    Escape,
    Tab,
    Backspace,
    DownArrow,
    UpArrow,
}

#[derive(Debug, Error)]
pub enum InputError {
    #[error("Input initialization or execution error: {0}")]
    Platform(String),
}

pub trait Input {
    fn click(&mut self, x: f64, y: f64) -> Result<(), InputError>;
    fn type_text(&mut self, text: &str) -> Result<(), InputError>;
    fn key(&mut self, key: KeyCode) -> Result<(), InputError>;
    fn scroll_vertical(&mut self, amount: i32) -> Result<(), InputError>;
    fn release_modifiers(&mut self) -> Result<(), InputError>;
    fn select_all(&mut self) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.release_modifiers()
    }
}

#[cfg(target_os = "macos")]
pub struct EnigoInput {
    enigo: enigo::Enigo,
}

#[cfg(target_os = "macos")]
impl EnigoInput {
    pub fn new() -> Result<Self, InputError> {
        let settings = enigo::Settings::default();
        let enigo =
            enigo::Enigo::new(&settings).map_err(|e| InputError::Platform(e.to_string()))?;
        Ok(Self { enigo })
    }
}

#[cfg(target_os = "macos")]
impl Input for EnigoInput {
    fn click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        use enigo::{Button, Coordinate, Direction, Mouse};
        self.release_modifiers()?;
        self.enigo
            .move_mouse(x.round() as i32, y.round() as i32, Coordinate::Abs)
            .map_err(|e| InputError::Platform(e.to_string()))?;
        let res = self
            .enigo
            .button(Button::Left, Direction::Click)
            .map_err(|e| InputError::Platform(e.to_string()));
        let _ = self.release_modifiers();
        res
    }

    fn type_text(&mut self, text: &str) -> Result<(), InputError> {
        use enigo::Keyboard;
        use std::thread;
        use std::time::Duration;
        self.release_modifiers()?;
        let mut res = Ok(());
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            let s = ch.to_string();
            if let Err(e) = self.enigo.text(&s) {
                res = Err(InputError::Platform(e.to_string()));
                break;
            }
            if chars.peek().is_some() {
                thread::sleep(Duration::from_millis(40));
            }
        }
        let _ = self.release_modifiers();
        res
    }

    fn key(&mut self, key: KeyCode) -> Result<(), InputError> {
        use enigo::{Direction, Key, Keyboard};
        self.release_modifiers()?;
        let k = match key {
            KeyCode::Return => Key::Return,
            KeyCode::Escape => Key::Escape,
            KeyCode::Tab => Key::Tab,
            KeyCode::Backspace => Key::Backspace,
            KeyCode::DownArrow => Key::DownArrow,
            KeyCode::UpArrow => Key::UpArrow,
        };
        let res = self
            .enigo
            .key(k, Direction::Click)
            .map_err(|e| InputError::Platform(e.to_string()));
        let _ = self.release_modifiers();
        res
    }

    fn scroll_vertical(&mut self, amount: i32) -> Result<(), InputError> {
        use enigo::{Axis, Mouse};
        self.release_modifiers()?;
        let res = self
            .enigo
            .scroll(amount, Axis::Vertical)
            .map_err(|e| InputError::Platform(e.to_string()));
        let _ = self.release_modifiers();
        res
    }

    fn release_modifiers(&mut self) -> Result<(), InputError> {
        use enigo::{Direction, Key, Keyboard};
        for modifier in [Key::Shift, Key::Control, Key::Alt, Key::Meta] {
            let _ = self.enigo.key(modifier, Direction::Release);
        }
        Ok(())
    }

    fn select_all(&mut self) -> Result<(), InputError> {
        use enigo::{Direction, Key, Keyboard};
        self.release_modifiers()?;
        let _ = self.enigo.key(Key::Meta, Direction::Press);
        let res = self
            .enigo
            .key(Key::Unicode('a'), Direction::Click)
            .map_err(|e| InputError::Platform(e.to_string()));
        let _ = self.enigo.key(Key::Meta, Direction::Release);
        let _ = self.release_modifiers();
        res
    }
}
