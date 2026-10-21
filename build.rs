use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=tools/vision_ocr.swift");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let swift_src = PathBuf::from("tools/vision_ocr.swift");

    if target_os == "macos" && swift_src.exists() {
        if let Ok(out_dir) = env::var("OUT_DIR") {
            let out_bin = PathBuf::from(out_dir).join("vision_ocr");
            let status = Command::new("swiftc")
                .arg(&swift_src)
                .arg("-o")
                .arg(&out_bin)
                .status();

            if let Ok(s) = status {
                if s.success() && out_bin.exists() {
                    println!("cargo:rustc-env=VISION_OCR_BIN={}", out_bin.display());
                    return;
                }
            }
        }
    }

    println!("cargo:rustc-env=VISION_OCR_BIN=");
}
