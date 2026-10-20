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
    let scale = if scale <= 0.0 { 1.0 } else { scale };

    let full_w = full.width();
    let full_h = full.height();

    let raw_x0 = (rect.x * scale).round().max(0.0) as u32;
    let raw_y0 = (rect.y * scale).round().max(0.0) as u32;
    let raw_x1 = ((rect.x + rect.width).max(0.0) * scale).round() as u32;
    let raw_y1 = ((rect.y + rect.height).max(0.0) * scale).round() as u32;

    let expected_w = raw_x1.saturating_sub(raw_x0);
    let expected_h = raw_y1.saturating_sub(raw_y0);

    let is_window_only_capture = (full.width() == expected_w && full.height() == expected_h)
        || ((raw_x0 > 0 || raw_y0 > 0) && (raw_x1 > full.width() || raw_y1 > full.height()));

    let (cropped, crop_origin_px, crop_w, crop_h) = if is_window_only_capture {
        if full.width() < MIN_CROP_PX || full.height() < MIN_CROP_PX || expected_w < MIN_CROP_PX || expected_h < MIN_CROP_PX {
            return Err(PrivacyError::CropTooSmall {
                width: full.width().min(expected_w),
                height: full.height().min(expected_h),
            });
        }
        (full.clone(), (raw_x0, raw_y0), expected_w, expected_h)
    } else {
        let x0 = raw_x0.min(full_w);
        let y0 = raw_y0.min(full_h);
        let x1 = raw_x1.min(full_w).max(x0);
        let y1 = raw_y1.min(full_h).max(y0);

        let w = x1 - x0;
        let h = y1 - y0;

        if w < MIN_CROP_PX || h < MIN_CROP_PX {
            return Err(PrivacyError::CropTooSmall {
                width: w,
                height: h,
            });
        }

        (crop_imm(full, x0, y0, w, h).to_image(), (x0, y0), w, h)
    };

    let total_px = (crop_w as u64) * (crop_h as u64);
    let (sent_rgba, sent_w, sent_h) = if total_px > MAX_PIXELS {
        let ratio = (MAX_PIXELS as f64 / total_px as f64).sqrt();
        let new_w = ((crop_w as f64 * ratio).floor() as u32).max(1);
        let new_h = ((crop_h as f64 * ratio).floor() as u32).max(1);
        let resized = resize(&cropped, new_w, new_h, FilterType::Lanczos3);
        (resized, new_w, new_h)
    } else {
        (cropped, crop_w, crop_h)
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
        scale_factor: scale,
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

pub fn prepare_corner_zoom_2x(
    window_img: &PreparedImage,
    corner: CornerRegion,
) -> Result<PreparedImage, PrivacyError> {
    let b64_str = window_img
        .data_url
        .strip_prefix("data:image/jpeg;base64,")
        .or_else(|| window_img.data_url.split_once(',').map(|(_, b)| b))
        .unwrap_or(&window_img.data_url);
    let decoded_bytes = BASE64_STANDARD
        .decode(b64_str.trim())
        .map_err(|e| PrivacyError::EncodeError(format!("base64 decode error: {}", e)))?;

    let rgba = if let Ok(dyn_img) = image::load_from_memory(&decoded_bytes) {
        dyn_img.to_rgba8()
    } else {
        let w = window_img.sent_size_px.0.max(100);
        let h = window_img.sent_size_px.1.max(100);
        RgbaImage::from_pixel(w, h, image::Rgba([240, 240, 240, 255]))
    };

    let full_w = rgba.width().max(2);
    let full_h = rgba.height().max(2);
    let sub_w = (full_w / 2).max(1);
    let sub_h = (full_h / 2).max(1);
    let sub_x = match corner {
        CornerRegion::BottomRight => full_w - sub_w,
        CornerRegion::BottomLeft => 0,
    };
    let sub_y = full_h - sub_h;

    let sub_crop = crop_imm(&rgba, sub_x, sub_y, sub_w, sub_h).to_image();
    let zoom_w = (sub_w * 2).max(1);
    let zoom_h = (sub_h * 2).max(1);
    let zoomed = resize(&sub_crop, zoom_w, zoom_h, FilterType::Triangle);

    let rgb = DynamicImage::ImageRgba8(zoomed).into_rgb8();
    let mut jpeg_bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_bytes, JPEG_QUALITY);
    encoder
        .encode_image(&rgb)
        .map_err(|e| PrivacyError::EncodeError(e.to_string()))?;

    let b64 = BASE64_STANDARD.encode(&jpeg_bytes);
    let data_url = format!("data:image/jpeg;base64,{}", b64);

    let (left_frac, top_frac, w_frac, h_frac) = corner.offset_and_scale_frac();
    let origin_x_px =
        window_img.crop_origin_px.0 + ((window_img.crop_size_px.0 as f64) * left_frac).round() as u32;
    let origin_y_px =
        window_img.crop_origin_px.1 + ((window_img.crop_size_px.1 as f64) * top_frac).round() as u32;
    let crop_w_px = ((window_img.crop_size_px.0 as f64) * w_frac).round().max(1.0) as u32;
    let crop_h_px = ((window_img.crop_size_px.1 as f64) * h_frac).round().max(1.0) as u32;

    Ok(PreparedImage {
        data_url,
        crop_origin_px: (origin_x_px, origin_y_px),
        crop_size_px: (crop_w_px, crop_h_px),
        sent_size_px: (zoom_w, zoom_h),
        scale_factor: window_img.scale_factor,
    })
}
