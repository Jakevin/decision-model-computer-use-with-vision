use image::RgbaImage;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MIN_CHROME_WINDOW_WIDTH_PT: f64 = 400.0;
pub const MIN_CHROME_WINDOW_HEIGHT_PT: f64 = 300.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WinInfo {
    pub app_name: String,
    pub title: String,
    pub is_minimized: bool,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

pub fn select_main_window_index(windows: &[WinInfo]) -> Option<usize> {
    windows
        .iter()
        .enumerate()
        .filter(|(_, w)| {
            !w.is_minimized
                && w.app_name.contains("Google Chrome")
                && !w.title.trim().is_empty()
                && w.width >= MIN_CHROME_WINDOW_WIDTH_PT
                && w.height >= MIN_CHROME_WINDOW_HEIGHT_PT
        })
        .max_by(|(_, a), (_, b)| {
            let area_a = a.width * a.height;
            let area_b = b.width * b.height;
            area_a
                .partial_cmp(&area_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

pub fn select_main_window(windows: &[WinInfo]) -> Option<Rect> {
    let idx = select_main_window_index(windows)?;
    let w = &windows[idx];
    Some(Rect {
        x: w.x,
        y: w.y,
        width: w.width,
        height: w.height,
    })
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
        let windows = xcap::Window::all().map_err(|e| CaptureError::Platform(e.to_string()))?;
        let infos: Vec<WinInfo> = windows
            .iter()
            .map(|win| WinInfo {
                app_name: win.app_name().unwrap_or_default(),
                title: win.title().unwrap_or_default(),
                is_minimized: win.is_minimized().unwrap_or(false),
                x: win.x().unwrap_or(0) as f64,
                y: win.y().unwrap_or(0) as f64,
                width: win.width().unwrap_or(0) as f64,
                height: win.height().unwrap_or(0) as f64,
            })
            .collect();
        if let Some(idx) = select_main_window_index(&infos) {
            windows[idx]
                .capture_image()
                .map_err(|e| CaptureError::Platform(e.to_string()))
        } else {
            Ok(RgbaImage::new(1, 1))
        }
    }

    fn chrome_window_rect(&self) -> Option<Rect> {
        let windows = xcap::Window::all().ok()?;
        let infos: Vec<WinInfo> = windows
            .into_iter()
            .map(|win| WinInfo {
                app_name: win.app_name().unwrap_or_default(),
                title: win.title().unwrap_or_default(),
                is_minimized: win.is_minimized().unwrap_or(false),
                x: win.x().unwrap_or(0) as f64,
                y: win.y().unwrap_or(0) as f64,
                width: win.width().unwrap_or(0) as f64,
                height: win.height().unwrap_or(0) as f64,
            })
            .collect();
        select_main_window(&infos)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_main_window_ignores_popups_and_picks_largest() {
        let windows = vec![
            // Tiny popup bubble came first
            WinInfo {
                app_name: "Google Chrome".to_string(),
                title: "Translate bubble".to_string(),
                is_minimized: false,
                x: 200.0,
                y: 100.0,
                width: 288.0,
                height: 87.0,
            },
            // Large untitled overlay / helper window
            WinInfo {
                app_name: "Google Chrome".to_string(),
                title: "   ".to_string(),
                is_minimized: false,
                x: 0.0,
                y: 0.0,
                width: 1400.0,
                height: 1000.0,
            },
            // Smaller Chrome window
            WinInfo {
                app_name: "Google Chrome".to_string(),
                title: "Secondary".to_string(),
                is_minimized: false,
                x: 50.0,
                y: 50.0,
                width: 600.0,
                height: 500.0,
            },
            // Main browser window (largest valid)
            WinInfo {
                app_name: "Google Chrome".to_string(),
                title: "Google Flights - Google Chrome".to_string(),
                is_minimized: false,
                x: 80.0,
                y: 40.0,
                width: 1200.0,
                height: 916.0,
            },
            // Minimized even larger window
            WinInfo {
                app_name: "Google Chrome".to_string(),
                title: "Minimized Flights".to_string(),
                is_minimized: true,
                x: 0.0,
                y: 0.0,
                width: 1920.0,
                height: 1080.0,
            },
        ];

        let rect = select_main_window(&windows).unwrap();
        assert_eq!(
            rect,
            Rect {
                x: 80.0,
                y: 40.0,
                width: 1200.0,
                height: 916.0,
            }
        );
    }
}
