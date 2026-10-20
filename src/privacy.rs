use crate::capture::Rect;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::{crop_imm, resize, FilterType};
use image::{DynamicImage, RgbaImage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAX_PIXELS: u64 = 1_600_000;
pub const MIN_CROP_PX: u32 = 50;
pub const JPEG_QUALITY: u8 = 85;
pub const DEFAULT_TOOLBAR_PT: f64 = 80.0;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PrivacyError {
    #[error("No crop rectangle provided; refusing to capture or send full screen")]
    NoCrop,
    #[error("Cropped region is too small ({width}x{height} px, minimum is 50x50 px)")]
    CropTooSmall { width: u32, height: u32 },
    #[error("Image encoding failed: {0}")]
    EncodeError(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedImage {
    pub data_url: String,
    pub crop_origin_px: (u32, u32),
    pub crop_size_px: (u32, u32),
    pub sent_size_px: (u32, u32),
    pub scale_factor: f64,
    #[serde(default)]
    pub window_rect: Option<Rect>,
    #[serde(default)]
    pub capture_size_px: (u32, u32),
    #[serde(skip)]
    pub source_rgba: Option<std::sync::Arc<RgbaImage>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditRecord {
    pub crop_origin_px: (u32, u32),
    pub crop_size_px: (u32, u32),
    pub sent_size_px: (u32, u32),
    pub purpose: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    records: Vec<AuditRecord>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    pub fn record(&mut self, img: &PreparedImage, purpose: impl Into<String>) {
        self.records.push(AuditRecord {
            crop_origin_px: img.crop_origin_px,
            crop_size_px: img.crop_size_px,
            sent_size_px: img.sent_size_px,
            purpose: purpose.into(),
        });
    }

    pub fn records(&self) -> &[AuditRecord] {
        &self.records
    }
}

pub fn prepare_image(
    full: &RgbaImage,
    crop: Option<Rect>,
    scale: f64,
) -> Result<PreparedImage, PrivacyError> {
    let rect = crop.ok_or(PrivacyError::NoCrop)?;

    let full_w = full.width();
    let full_h = full.height();
    let win_w_u32 = rect.width.round().max(0.0) as u32;
    let win_h_u32 = rect.height.round().max(0.0) as u32;

    if full_w < MIN_CROP_PX
        || full_h < MIN_CROP_PX
        || win_w_u32 < MIN_CROP_PX
        || win_h_u32 < MIN_CROP_PX
    {
        return Err(PrivacyError::CropTooSmall {
            width: full_w.min(win_w_u32),
            height: full_h.min(win_h_u32),
        });
    }

    let crop_origin_px = (0, 0);
    let crop_w = full_w;
    let crop_h = full_h;
    let eff_scale = if rect.width > 0.0 {
        (full_w as f64) / rect.width
    } else if scale > 0.0 {
        scale
    } else {
        1.0
    };

    let total_px = (crop_w as u64) * (crop_h as u64);
    let (sent_rgba, sent_w, sent_h) = if total_px > MAX_PIXELS {
        let ratio = (MAX_PIXELS as f64 / total_px as f64).sqrt();
        let new_w = ((crop_w as f64 * ratio).floor() as u32).max(1);
        let new_h = ((crop_h as f64 * ratio).floor() as u32).max(1);
        let resized = resize(full, new_w, new_h, FilterType::Lanczos3);
        (resized, new_w, new_h)
    } else {
        (full.clone(), crop_w, crop_h)
    };

    let rgb = DynamicImage::ImageRgba8(sent_rgba).into_rgb8();
    let mut jpeg_bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_bytes, JPEG_QUALITY);
    encoder
        .encode_image(&rgb)
        .map_err(|e| PrivacyError::EncodeError(e.to_string()))?;

    let b64 = BASE64_STANDARD.encode(&jpeg_bytes);
    let data_url = format!("data:image/jpeg;base64,{}", b64);

    Ok(PreparedImage {
        data_url,
        crop_origin_px,
        crop_size_px: (crop_w, crop_h),
        sent_size_px: (sent_w, sent_h),
        scale_factor: eff_scale,
        window_rect: Some(rect),
        capture_size_px: (full_w, full_h),
        source_rgba: Some(std::sync::Arc::new(full.clone())),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CornerRegion {
    BottomRight,
    BottomLeft,
}

impl CornerRegion {
    pub fn offset_and_scale_frac(&self) -> (f64, f64, f64, f64) {
        match self {
            CornerRegion::BottomRight => (0.5, 0.5, 0.5, 0.5),
            CornerRegion::BottomLeft => (0.0, 0.5, 0.5, 0.5),
        }
    }
}

pub fn corner_crop_rect_px(full_w: u32, full_h: u32, corner: CornerRegion) -> (u32, u32, u32, u32) {
    let full_w = full_w.max(2);
    let full_h = full_h.max(2);
    let sub_w = (full_w / 2).max(1);
    let sub_h = (full_h / 2).max(1);
    let sub_x = match corner {
        CornerRegion::BottomRight => full_w - sub_w,
        CornerRegion::BottomLeft => 0,
    };
    let sub_y = full_h - sub_h;
    (sub_x, sub_y, sub_w, sub_h)
}

pub fn corner_offset_and_scale_frac(
    window_img: &PreparedImage,
    corner: CornerRegion,
) -> (f64, f64, f64, f64) {
    let full_w = window_img.crop_size_px.0.max(2);
    let full_h = window_img.crop_size_px.1.max(2);
    let (sub_x, sub_y, sub_w, sub_h) = corner_crop_rect_px(full_w, full_h, corner);
    (
        (sub_x as f64) / (full_w as f64),
        (sub_y as f64) / (full_h as f64),
        (sub_w as f64) / (full_w as f64),
        (sub_h as f64) / (full_h as f64),
    )
}

pub fn content_crop_rect_px(window_img: &PreparedImage) -> (u32, u32, u32, u32) {
    let full_w = window_img.crop_size_px.0.max(2);
    let full_h = window_img.crop_size_px.1.max(2);
    let scale_y = if let Some(win) = window_img.window_rect {
        if win.height > 0.0 {
            (full_h as f64) / win.height
        } else if window_img.scale_factor > 0.0 {
            window_img.scale_factor
        } else {
            1.0
        }
    } else if window_img.scale_factor > 0.0 {
        window_img.scale_factor
    } else {
        1.0
    };
    let toolbar_px = (DEFAULT_TOOLBAR_PT * scale_y).round().max(0.0) as u32;
    let sub_x = 0u32;
    let sub_y = toolbar_px.min(full_h / 2).min(full_h - 1);
    let sub_w = full_w;
    let sub_h = (full_h - sub_y).max(1);
    (sub_x, sub_y, sub_w, sub_h)
}

pub fn content_region_offset_and_scale_frac(window_img: &PreparedImage) -> (f64, f64, f64, f64) {
    let full_w = window_img.crop_size_px.0.max(2);
    let full_h = window_img.crop_size_px.1.max(2);
    let (sub_x, sub_y, sub_w, sub_h) = content_crop_rect_px(window_img);
    (
        (sub_x as f64) / (full_w as f64),
        (sub_y as f64) / (full_h as f64),
        (sub_w as f64) / (full_w as f64),
        (sub_h as f64) / (full_h as f64),
    )
}

fn crop_and_zoom_from_original(
    window_img: &PreparedImage,
    sub_x: u32,
    sub_y: u32,
    sub_w: u32,
    sub_h: u32,
) -> Result<PreparedImage, PrivacyError> {
    let fallback_rgba;
    let src_rgba: &RgbaImage = if let Some(ref orig) = window_img.source_rgba {
        orig.as_ref()
    } else {
        let w = window_img.crop_size_px.0.max(2);
        let h = window_img.crop_size_px.1.max(2);
        fallback_rgba = RgbaImage::from_pixel(w, h, image::Rgba([240, 240, 240, 255]));
        &fallback_rgba
    };

    let img_w = src_rgba.width().max(2);
    let img_h = src_rgba.height().max(2);
    let cx = sub_x.min(img_w - 1);
    let cy = sub_y.min(img_h - 1);
    let cw = sub_w.min(img_w - cx).max(1);
    let ch = sub_h.min(img_h - cy).max(1);

    let sub_crop = crop_imm(src_rgba, cx, cy, cw, ch).to_image();

    let target_w = (cw * 2).max(1);
    let target_h = (ch * 2).max(1);
    let target_px = (target_w as u64) * (target_h as u64);
    let (sent_w, sent_h) = if target_px > MAX_PIXELS {
        let ratio = (MAX_PIXELS as f64 / target_px as f64).sqrt();
        (
            ((target_w as f64 * ratio).floor() as u32).max(1),
            ((target_h as f64 * ratio).floor() as u32).max(1),
        )
    } else {
        (target_w, target_h)
    };

    let zoomed = if sent_w == cw && sent_h == ch {
        sub_crop
    } else {
        resize(&sub_crop, sent_w, sent_h, FilterType::Lanczos3)
    };

    let rgb = DynamicImage::ImageRgba8(zoomed).into_rgb8();
    let mut jpeg_bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_bytes, JPEG_QUALITY);
    encoder
        .encode_image(&rgb)
        .map_err(|e| PrivacyError::EncodeError(e.to_string()))?;

    let b64 = BASE64_STANDARD.encode(&jpeg_bytes);
    let data_url = format!("data:image/jpeg;base64,{}", b64);

    let origin_x_px = window_img.crop_origin_px.0 + cx;
    let origin_y_px = window_img.crop_origin_px.1 + cy;
    let capture_size_px = if window_img.capture_size_px.0 > 0 && window_img.capture_size_px.1 > 0 {
        window_img.capture_size_px
    } else {
        window_img.crop_size_px
    };

    Ok(PreparedImage {
        data_url,
        crop_origin_px: (origin_x_px, origin_y_px),
        crop_size_px: (cw, ch),
        sent_size_px: (sent_w, sent_h),
        scale_factor: window_img.scale_factor,
        window_rect: window_img.window_rect,
        capture_size_px,
        source_rgba: window_img.source_rgba.clone(),
    })
}

pub fn prepare_corner_zoom_2x(
    window_img: &PreparedImage,
    corner: CornerRegion,
) -> Result<PreparedImage, PrivacyError> {
    let full_w = window_img.crop_size_px.0.max(2);
    let full_h = window_img.crop_size_px.1.max(2);
    let (sub_x, sub_y, sub_w, sub_h) = corner_crop_rect_px(full_w, full_h, corner);
    crop_and_zoom_from_original(window_img, sub_x, sub_y, sub_w, sub_h)
}

pub fn prepare_content_zoom_2x(window_img: &PreparedImage) -> Result<PreparedImage, PrivacyError> {
    let (sub_x, sub_y, sub_w, sub_h) = content_crop_rect_px(window_img);
    crop_and_zoom_from_original(window_img, sub_x, sub_y, sub_w, sub_h)
}

pub fn encode_jpeg(img: &RgbaImage, quality: u8) -> Result<Vec<u8>, PrivacyError> {
    let rgb = DynamicImage::ImageRgba8(img.clone()).into_rgb8();
    let mut jpeg_bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_bytes, quality);
    encoder
        .encode_image(&rgb)
        .map_err(|e| PrivacyError::EncodeError(e.to_string()))?;
    Ok(jpeg_bytes)
}
