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
    PageDown,
    PageUp,
}

pub const DEFAULT_SCROLL_DELTA_PX: i32 = -400;
pub const DEFAULT_SCROLL_UP_DELTA_PX: i32 = 400;

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
    fn move_mouse(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        let _ = (x, y);
        Ok(())
    }
    fn scroll_vertical_at(&mut self, x: f64, y: f64, amount: i32) -> Result<(), InputError> {
        self.move_mouse(x, y)?;
        self.scroll_vertical(amount)
    }
    fn focus_click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        self.click(x, y)
    }
    fn release_modifiers(&mut self) -> Result<(), InputError>;
    fn select_all(&mut self) -> Result<(), InputError> {
        self.release_modifiers()?;
        self.release_modifiers()
    }
}

#[cfg(target_os = "macos")]
pub struct QuartzInput;

#[cfg(target_os = "macos")]
pub type EnigoInput = QuartzInput;

#[cfg(target_os = "macos")]
impl QuartzInput {
    pub fn new() -> Result<Self, InputError> {
        use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
        let _ = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|()| InputError::Platform("Failed to create CGEventSource".to_string()))?;
        Ok(Self)
    }

    fn event_source(
    ) -> Result<core_graphics::event_source::CGEventSource, InputError> {
        use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
        CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|()| InputError::Platform("Failed to create CGEventSource".to_string()))
    }
}

#[cfg(target_os = "macos")]
impl Default for QuartzInput {
    fn default() -> Self {
        Self
    }
}

#[cfg(target_os = "macos")]
impl Input for QuartzInput {
    fn click(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        use core_graphics::event::{
            CGEvent, CGEventFlags, CGEventTapLocation, CGEventType, CGMouseButton,
        };
        use core_graphics::geometry::CGPoint;
        use std::thread;
        use std::time::Duration;

        self.release_modifiers()?;
        let source = Self::event_source()?;
        let pt = CGPoint::new(x, y);

        let moved = CGEvent::new_mouse_event(
            source.clone(),
            CGEventType::MouseMoved,
            pt,
            CGMouseButton::Left,
        )
        .map_err(|()| InputError::Platform("Failed to create MouseMoved CGEvent".to_string()))?;
        moved.set_flags(CGEventFlags::CGEventFlagNull);
        moved.post(CGEventTapLocation::HID);
        thread::sleep(Duration::from_millis(40));

        let down_res = CGEvent::new_mouse_event(
            source.clone(),
            CGEventType::LeftMouseDown,
            pt,
            CGMouseButton::Left,
        )
        .map_err(|()| InputError::Platform("Failed to create LeftMouseDown CGEvent".to_string()))
        .map(|ev| {
            ev.set_flags(CGEventFlags::CGEventFlagNull);
            ev.post(CGEventTapLocation::HID);
        });
        thread::sleep(Duration::from_millis(40));

        let up_res = CGEvent::new_mouse_event(
            source,
            CGEventType::LeftMouseUp,
            pt,
            CGMouseButton::Left,
        )
        .map_err(|()| InputError::Platform("Failed to create LeftMouseUp CGEvent".to_string()))
        .map(|ev| {
            ev.set_flags(CGEventFlags::CGEventFlagNull);
            ev.post(CGEventTapLocation::HID);
        });
        thread::sleep(Duration::from_millis(40));

        let _ = self.release_modifiers();
        down_res?;
        up_res
    }

    fn type_text(&mut self, text: &str) -> Result<(), InputError> {
        use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};
        use std::thread;
        use std::time::Duration;

        self.release_modifiers()?;
        let source = Self::event_source()?;
        let mut buf = [0u8; 4];

        for ch in text.chars() {
            let s = ch.encode_utf8(&mut buf);
            let down = CGEvent::new_keyboard_event(source.clone(), 0, true).map_err(|()| {
                InputError::Platform("Failed to create key down CGEvent".to_string())
            })?;
            down.set_flags(CGEventFlags::CGEventFlagNull);
            down.set_string(s);
            down.post(CGEventTapLocation::HID);

            let up = CGEvent::new_keyboard_event(source.clone(), 0, false)
                .map_err(|()| InputError::Platform("Failed to create key up CGEvent".to_string()))?;
            up.set_flags(CGEventFlags::CGEventFlagNull);
            up.set_string(s);
            up.post(CGEventTapLocation::HID);

            thread::sleep(Duration::from_millis(30));
        }

        let _ = self.release_modifiers();
        Ok(())
    }

    fn key(&mut self, key: KeyCode) -> Result<(), InputError> {
        use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};

        self.release_modifiers()?;
        let source = Self::event_source()?;
        let code: CGKeyCode = match key {
            KeyCode::Return => 36,
            KeyCode::Escape => 53,
            KeyCode::Tab => 48,
            KeyCode::Backspace => 51,
            KeyCode::DownArrow => 125,
            KeyCode::UpArrow => 126,
            KeyCode::PageDown => 121,
            KeyCode::PageUp => 116,
        };

        let down = CGEvent::new_keyboard_event(source.clone(), code, true)
            .map_err(|()| InputError::Platform("Failed to create key down CGEvent".to_string()))?;
        down.set_flags(CGEventFlags::CGEventFlagNull);
        down.post(CGEventTapLocation::HID);

        let up = CGEvent::new_keyboard_event(source, code, false)
            .map_err(|()| InputError::Platform("Failed to create key up CGEvent".to_string()))?;
        up.set_flags(CGEventFlags::CGEventFlagNull);
        up.post(CGEventTapLocation::HID);

        let _ = self.release_modifiers();
        Ok(())
    }

    fn move_mouse(&mut self, x: f64, y: f64) -> Result<(), InputError> {
        use core_graphics::event::{
            CGEvent, CGEventFlags, CGEventTapLocation, CGEventType, CGMouseButton,
        };
        use core_graphics::geometry::CGPoint;

        let source = Self::event_source()?;
        let pt = CGPoint::new(x, y);
        let moved = CGEvent::new_mouse_event(
            source,
            CGEventType::MouseMoved,
            pt,
            CGMouseButton::Left,
        )
        .map_err(|()| InputError::Platform("Failed to create MouseMoved CGEvent".to_string()))?;
        moved.set_flags(CGEventFlags::CGEventFlagNull);
        moved.post(CGEventTapLocation::HID);
        Ok(())
    }

    fn scroll_vertical(&mut self, amount: i32) -> Result<(), InputError> {
        use core_graphics::event::{
            CGEvent, CGEventFlags, CGEventTapLocation, ScrollEventUnit,
        };

        self.release_modifiers()?;
        let source = Self::event_source()?;
        let pixel_delta = if amount.abs() <= 20 {
            amount * 80
        } else {
            amount
        };
        let ev = CGEvent::new_scroll_event(source, ScrollEventUnit::PIXEL, 1, pixel_delta, 0, 0)
            .map_err(|()| InputError::Platform("Failed to create scroll CGEvent".to_string()))?;
        ev.set_flags(CGEventFlags::CGEventFlagNull);
        ev.post(CGEventTapLocation::HID);
        let _ = self.release_modifiers();
        Ok(())
    }

    fn scroll_vertical_at(&mut self, x: f64, y: f64, amount: i32) -> Result<(), InputError> {
        use core_graphics::event::{
            CGEvent, CGEventFlags, CGEventTapLocation, CGEventType, CGMouseButton, ScrollEventUnit,
        };
        use core_graphics::geometry::CGPoint;
        use std::thread;
        use std::time::Duration;

        #[link(name = "CoreGraphics", kind = "framework")]
        extern "C" {
            fn CGEventSetLocation(
                event: core_graphics::sys::CGEventRef,
                location: CGPoint,
            );
        }

        self.release_modifiers()?;
        let source = Self::event_source()?;
        let pt = CGPoint::new(x, y);

        let moved = CGEvent::new_mouse_event(
            source.clone(),
            CGEventType::MouseMoved,
            pt,
            CGMouseButton::Left,
        )
        .map_err(|()| InputError::Platform("Failed to create MouseMoved CGEvent".to_string()))?;
        moved.set_flags(CGEventFlags::CGEventFlagNull);
        moved.post(CGEventTapLocation::HID);
        thread::sleep(Duration::from_millis(40));

        let pixel_delta = if amount.abs() <= 20 {
            amount * 80
        } else {
            amount
        };
        let ev = CGEvent::new_scroll_event(source, ScrollEventUnit::PIXEL, 1, pixel_delta, 0, 0)
            .map_err(|()| InputError::Platform("Failed to create scroll CGEvent".to_string()))?;
        unsafe {
            let ev_ref = *(&ev as *const CGEvent as *const core_graphics::sys::CGEventRef);
            CGEventSetLocation(ev_ref, pt);
        }
        ev.set_flags(CGEventFlags::CGEventFlagNull);
        ev.post(CGEventTapLocation::HID);
        let _ = self.release_modifiers();
        Ok(())
    }

    fn release_modifiers(&mut self) -> Result<(), InputError> {
        use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};

        let source = Self::event_source()?;
        // Shift = 56, Control = 59, Option = 58, Command = 55
        for code in [56 as CGKeyCode, 59, 58, 55] {
            if let Ok(up) = CGEvent::new_keyboard_event(source.clone(), code, false) {
                up.set_flags(CGEventFlags::CGEventFlagNull);
                up.post(CGEventTapLocation::HID);
            }
        }
        Ok(())
    }

    fn select_all(&mut self) -> Result<(), InputError> {
        use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};

        self.release_modifiers()?;
        let source = Self::event_source()?;

        let down = CGEvent::new_keyboard_event(source.clone(), 0, true).map_err(|()| {
            InputError::Platform("Failed to create select_all down CGEvent".to_string())
        })?;
        down.set_flags(CGEventFlags::CGEventFlagCommand);
        down.set_string("a");
        down.post(CGEventTapLocation::HID);

        let up = CGEvent::new_keyboard_event(source, 0, false).map_err(|()| {
            InputError::Platform("Failed to create select_all up CGEvent".to_string())
        })?;
        up.set_flags(CGEventFlags::CGEventFlagCommand);
        up.set_string("a");
        up.post(CGEventTapLocation::HID);

        self.release_modifiers()
    }
}
