use crate::api::ApiKey;
use regex::Regex;
use serde::Serialize;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn redact(input: &str, current_key: Option<&ApiKey>) -> String {
    let mut out = input.to_string();
    if let Some(key) = current_key {
        let secret = key.expose_secret();
        if !secret.is_empty() {
            out = out.replace(secret, "***");
        }
    }
    if let Ok(re) = Regex::new(r"sk-vsr-[0-9A-Za-z]{43}") {
        out = re.replace_all(&out, "***").into_owned();
    }
    out
}

pub struct RunLogger {
    path: PathBuf,
    file: File,
    api_key: Option<ApiKey>,
}

impl RunLogger {
    pub fn new_in_runs_dir(api_key: Option<ApiKey>) -> io::Result<Self> {
        Self::new_in_dir(Path::new("./runs"), api_key)
    }

    pub fn new_in_dir(dir: &Path, api_key: Option<ApiKey>) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = dir.join(format!("run-{}.jsonl", ts));
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            file,
            api_key,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn log_entry<T: Serialize>(&mut self, entry: &T) -> io::Result<String> {
        let raw = serde_json::to_string(entry)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let clean = redact(&raw, self.api_key.as_ref());
        writeln!(self.file, "{}", clean)?;
        self.file.flush()?;
        Ok(clean)
    }
}
