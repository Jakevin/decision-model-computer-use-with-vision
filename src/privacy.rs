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

    let x0 = raw_x0.min(full_w);
    let y0 = raw_y0.min(full_h);
    let x1 = raw_x1.min(full_w).max(x0);
    let y1 = raw_y1.min(full_h).max(y0);

    let crop_w = x1 - x0;
    let crop_h = y1 - y0;

    if crop_w < MIN_CROP_PX || crop_h < MIN_CROP_PX {
        return Err(PrivacyError::CropTooSmall {
            width: crop_w,
            height: crop_h,
        });
    }

    let cropped = crop_imm(full, x0, y0, crop_w, crop_h).to_image();

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
        crop_origin_px: (x0, y0),
        crop_size_px: (crop_w, crop_h),
        sent_size_px: (sent_w, sent_h),
        scale_factor: scale,
    })
}
