use image::RgbaImage;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("No primary monitor found")]
    NoMonitor,
    #[error("Capture failed: {0}")]
    Platform(String),
}

pub trait Screen {
    fn capture(&self) -> Result<RgbaImage, CaptureError>;
    fn chrome_window_rect(&self) -> Option<Rect>;
    fn scale_factor(&self) -> f64 {
        2.0
    }
}

#[cfg(target_os = "macos")]
pub struct MacScreen;

#[cfg(target_os = "macos")]
impl MacScreen {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(target_os = "macos")]
impl Default for MacScreen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "macos")]
impl Screen for MacScreen {
    fn capture(&self) -> Result<RgbaImage, CaptureError> {
        let monitors = xcap::Monitor::all().map_err(|e| CaptureError::Platform(e.to_string()))?;
        let monitor = monitors
            .into_iter()
            .find(|m| m.is_primary().unwrap_or(false))
            .or_else(|| xcap::Monitor::all().ok()?.into_iter().next())
            .ok_or(CaptureError::NoMonitor)?;
        monitor
            .capture_image()
            .map_err(|e| CaptureError::Platform(e.to_string()))
    }

    fn chrome_window_rect(&self) -> Option<Rect> {
        let windows = xcap::Window::all().ok()?;
        for win in windows {
            if win.is_minimized().unwrap_or(false) {
                continue;
            }
            let app_name = win.app_name().unwrap_or_default();
            let width = win.width().unwrap_or(0);
            let height = win.height().unwrap_or(0);
            if app_name.contains("Google Chrome") && width > 50 && height > 50 {
                let x = win.x().unwrap_or(0);
                let y = win.y().unwrap_or(0);
                return Some(Rect {
                    x: x as f64,
                    y: y as f64,
                    width: width as f64,
                    height: height as f64,
                });
            }
        }
        None
    }

    fn scale_factor(&self) -> f64 {
        if let Ok(monitors) = xcap::Monitor::all() {
            if let Some(m) = monitors
                .into_iter()
                .find(|m| m.is_primary().unwrap_or(false))
            {
                if let Ok(sf) = m.scale_factor() {
                    return sf as f64;
                }
            }
        }
        2.0
    }
}
