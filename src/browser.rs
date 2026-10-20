use std::process::Command;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BrowserError {
    #[error("osascript execution failed: {0}")]
    CommandFailed(String),
}

pub trait Browser {
    fn active_tab_url(&self) -> Option<String>;
    fn set_active_tab_url(&mut self, url: &str) -> Result<(), BrowserError>;
    fn activate_chrome(&mut self) -> Result<(), BrowserError> {
        Ok(())
    }
    fn frontmost_app(&self) -> Option<String> {
        Some("Google Chrome".to_string())
    }
    fn is_chrome_frontmost(&self) -> bool {
        self.frontmost_app()
            .as_deref()
            .map(|n| n.trim() == "Google Chrome")
            .unwrap_or(false)
    }
}

impl<T: Browser + ?Sized> Browser for Box<T> {
    fn active_tab_url(&self) -> Option<String> {
        (**self).active_tab_url()
    }

    fn set_active_tab_url(&mut self, url: &str) -> Result<(), BrowserError> {
        (**self).set_active_tab_url(url)
    }

    fn activate_chrome(&mut self) -> Result<(), BrowserError> {
        (**self).activate_chrome()
    }

    fn frontmost_app(&self) -> Option<String> {
        (**self).frontmost_app()
    }

    fn is_chrome_frontmost(&self) -> bool {
        (**self).is_chrome_frontmost()
    }
}

impl<T: Browser + ?Sized> Browser for &mut T {
    fn active_tab_url(&self) -> Option<String> {
        (**self).active_tab_url()
    }

    fn set_active_tab_url(&mut self, url: &str) -> Result<(), BrowserError> {
        (**self).set_active_tab_url(url)
    }

    fn activate_chrome(&mut self) -> Result<(), BrowserError> {
        (**self).activate_chrome()
    }

    fn frontmost_app(&self) -> Option<String> {
        (**self).frontmost_app()
    }

    fn is_chrome_frontmost(&self) -> bool {
        (**self).is_chrome_frontmost()
    }
}

pub fn active_tab_url() -> Option<String> {
    let output = Command::new("osascript")
        .arg("-e")
        .arg("tell application \"Google Chrome\" to get URL of active tab of front window")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if url.is_empty() {
        None
    } else {
        Some(url)
    }
}

pub fn set_active_tab_url(url: &str) -> Result<(), BrowserError> {
    let escaped = url.replace('\\', "\\\\").replace('"', "\\\"");
    let script = format!(
        "tell application \"Google Chrome\" to set URL of active tab of front window to \"{}\"",
        escaped
    );
    let output = Command::new("osascript")
        .arg("-e")
        .arg(&script)
        .output()
        .map_err(|e| BrowserError::CommandFailed(e.to_string()))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(BrowserError::CommandFailed(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ))
    }
}

pub fn activate_chrome() -> Result<(), BrowserError> {
    let output = Command::new("osascript")
        .arg("-e")
        .arg("tell application \"Google Chrome\" to activate")
        .output()
        .map_err(|e| BrowserError::CommandFailed(e.to_string()))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(BrowserError::CommandFailed(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ))
    }
}

pub fn frontmost_app() -> Option<String> {
    let output = Command::new("osascript")
        .arg("-e")
        .arg("tell application \"System Events\" to get name of first process whose frontmost is true")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

pub fn is_chrome_frontmost() -> bool {
    frontmost_app()
        .as_deref()
        .map(|n| n == "Google Chrome")
        .unwrap_or(false)
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MacBrowser;

impl MacBrowser {
    pub fn new() -> Self {
        Self
    }
}

impl Browser for MacBrowser {
    fn active_tab_url(&self) -> Option<String> {
        active_tab_url()
    }

    fn set_active_tab_url(&mut self, url: &str) -> Result<(), BrowserError> {
        set_active_tab_url(url)
    }

    fn activate_chrome(&mut self) -> Result<(), BrowserError> {
        activate_chrome()
    }

    fn frontmost_app(&self) -> Option<String> {
        frontmost_app()
    }
}

#[derive(Debug, Clone)]
pub struct DefaultAllowedBrowser {
    pub url: Option<String>,
    pub set_urls: Vec<String>,
    pub activated: usize,
}

impl Default for DefaultAllowedBrowser {
    fn default() -> Self {
        Self {
            url: Some("https://www.google.com/travel/flights".to_string()),
            set_urls: Vec::new(),
            activated: 0,
        }
    }
}

impl Browser for DefaultAllowedBrowser {
    fn active_tab_url(&self) -> Option<String> {
        self.url.clone()
    }

    fn set_active_tab_url(&mut self, url: &str) -> Result<(), BrowserError> {
        self.url = Some(url.to_string());
        self.set_urls.push(url.to_string());
        Ok(())
    }

    fn activate_chrome(&mut self) -> Result<(), BrowserError> {
        self.activated += 1;
        Ok(())
    }
}
