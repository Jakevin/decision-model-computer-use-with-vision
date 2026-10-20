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
    #[serde(default)]
    pub window_id: u32,
    pub app_name: String,
    pub title: String,
    pub is_minimized: bool,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone)]
pub struct BoundWindowCapture {
    pub window_id: u32,
    pub image: RgbaImage,
    pub rect: Option<Rect>,
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

pub fn select_main_window_with_id(windows: &[WinInfo]) -> Option<(u32, Rect)> {
    let idx = select_main_window_index(windows)?;
    let w = &windows[idx];
    Some((
        w.window_id,
        Rect {
            x: w.x,
            y: w.y,
            width: w.width,
            height: w.height,
        },
    ))
}

pub fn select_main_window(windows: &[WinInfo]) -> Option<Rect> {
    select_main_window_with_id(windows).map(|(_, rect)| rect)
}

pub fn select_topmost_app_at(windows: &[WinInfo], x_pt: f64, y_pt: f64) -> Option<String> {
    for w in windows {
        if w.is_minimized || w.width <= 1.0 || w.height <= 1.0 {
            continue;
        }
        let app = w.app_name.trim();
        if app.is_empty() || app == "Window Server" || app == "Dock" {
            continue;
        }
        if x_pt >= w.x && x_pt <= w.x + w.width && y_pt >= w.y && y_pt <= w.y + w.height {
            return Some(app.to_string());
        }
    }
    None
}

#[cfg(target_os = "macos")]
pub fn topmost_app_at(x_pt: f64, y_pt: f64) -> Option<String> {
    let windows = xcap::Window::all().ok()?;
    let infos: Vec<WinInfo> = windows
        .into_iter()
        .map(|win| WinInfo {
            window_id: win.id().unwrap_or(0),
            app_name: win.app_name().unwrap_or_default(),
            title: win.title().unwrap_or_default(),
            is_minimized: win.is_minimized().unwrap_or(false),
            x: win.x().unwrap_or(0) as f64,
            y: win.y().unwrap_or(0) as f64,
            width: win.width().unwrap_or(0) as f64,
            height: win.height().unwrap_or(0) as f64,
        })
        .collect();
    select_topmost_app_at(&infos, x_pt, y_pt)
}

#[cfg(not(target_os = "macos"))]
pub fn topmost_app_at(_x_pt: f64, _y_pt: f64) -> Option<String> {
    None
}

pub fn changed_fraction(a: &RgbaImage, b: &RgbaImage) -> f64 {
    use image::imageops::{grayscale, resize, FilterType};

    let gray_a = grayscale(a);
    let gray_b = grayscale(b);
    let thumb_a = resize(&gray_a, 320, 180, FilterType::Triangle);
    let thumb_b = resize(&gray_b, 320, 180, FilterType::Triangle);

    let mut diff_pixels = 0usize;
    let total = (320 * 180) as f64;
    for (pa, pb) in thumb_a.pixels().zip(thumb_b.pixels()) {
        if (pa[0] as i32 - pb[0] as i32).abs() > 24 {
            diff_pixels += 1;
        }
    }
    (diff_pixels as f64) / total
}

pub fn is_changed(a: &RgbaImage, b: &RgbaImage) -> bool {
    changed_fraction(a, b) > 0.002
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
    fn capture_window(&self) -> Result<BoundWindowCapture, CaptureError> {
        Ok(BoundWindowCapture {
            window_id: 0,
            image: self.capture()?,
            rect: self.chrome_window_rect(),
        })
    }
    fn scale_factor(&self) -> f64 {
        2.0
    }
    fn topmost_app_at(&self, _x_pt: f64, _y_pt: f64) -> Option<String> {
        Some("Google Chrome".to_string())
    }
}

#[cfg(target_os = "macos")]
pub struct MacScreen {
    last_window: std::sync::Mutex<Option<(u32, Rect)>>,
}

#[cfg(target_os = "macos")]
impl MacScreen {
    pub fn new() -> Self {
        Self {
            last_window: std::sync::Mutex::new(None),
        }
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
    fn capture_window(&self) -> Result<BoundWindowCapture, CaptureError> {
        let windows = xcap::Window::all().map_err(|e| CaptureError::Platform(e.to_string()))?;
        let infos: Vec<WinInfo> = windows
            .iter()
            .map(|win| WinInfo {
                window_id: win.id().unwrap_or(0),
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
            let info = &infos[idx];
            let rect = Rect {
                x: info.x,
                y: info.y,
                width: info.width,
                height: info.height,
            };
            let image = windows[idx]
                .capture_image()
                .map_err(|e| CaptureError::Platform(e.to_string()))?;
            *self.last_window.lock().unwrap() = Some((info.window_id, rect));
            Ok(BoundWindowCapture {
                window_id: info.window_id,
                image,
                rect: Some(rect),
            })
        } else {
            *self.last_window.lock().unwrap() = None;
            Ok(BoundWindowCapture {
                window_id: 0,
                image: RgbaImage::new(1, 1),
                rect: None,
            })
        }
    }

    fn capture(&self) -> Result<RgbaImage, CaptureError> {
        self.capture_window().map(|c| c.image)
    }

    fn chrome_window_rect(&self) -> Option<Rect> {
        let cached = *self.last_window.lock().unwrap();
        let windows = xcap::Window::all().ok()?;
        let infos: Vec<WinInfo> = windows
            .into_iter()
            .map(|win| WinInfo {
                window_id: win.id().unwrap_or(0),
                app_name: win.app_name().unwrap_or_default(),
                title: win.title().unwrap_or_default(),
                is_minimized: win.is_minimized().unwrap_or(false),
                x: win.x().unwrap_or(0) as f64,
                y: win.y().unwrap_or(0) as f64,
                width: win.width().unwrap_or(0) as f64,
                height: win.height().unwrap_or(0) as f64,
            })
            .collect();
        if let Some((bound_id, bound_rect)) = cached {
            if bound_id != 0 {
                if let Some(w) = infos.iter().find(|w| w.window_id == bound_id) {
                    return Some(Rect {
                        x: w.x,
                        y: w.y,
                        width: w.width,
                        height: w.height,
                    });
                }
                return Some(bound_rect);
            }
        }
        select_main_window(&infos)
    }

    fn scale_factor(&self) -> f64 {
        2.0
    }

    fn topmost_app_at(&self, x_pt: f64, y_pt: f64) -> Option<String> {
        topmost_app_at(x_pt, y_pt)
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
                window_id: 101,
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
                window_id: 102,
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
                window_id: 103,
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
                window_id: 104,
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
                window_id: 105,
                app_name: "Google Chrome".to_string(),
                title: "Minimized Flights".to_string(),
                is_minimized: true,
                x: 0.0,
                y: 0.0,
                width: 1920.0,
                height: 1080.0,
            },
        ];

        let (win_id, rect) = select_main_window_with_id(&windows).unwrap();
        assert_eq!(win_id, 104);
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

    #[test]
    fn changed_fraction_and_is_changed_on_identical_and_bright_patch() {
        let base = RgbaImage::from_pixel(1200, 900, image::Rgba([40, 40, 40, 255]));
        let identical = base.clone();
        assert_eq!(changed_fraction(&base, &identical), 0.0);
        assert!(!is_changed(&base, &identical));

        // A 60x60 bright patch on 1200x900 (~0.33% area > 0.002 threshold)
        let mut with_patch = base.clone();
        for y in 100..160 {
            for x in 200..260 {
                with_patch.put_pixel(x, y, image::Rgba([240, 240, 240, 255]));
            }
        }
        let frac = changed_fraction(&base, &with_patch);
        assert!(frac > 0.002, "expected fraction > 0.002, got {}", frac);
        assert!(is_changed(&base, &with_patch));
    }
}
