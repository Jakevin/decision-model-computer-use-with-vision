use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::thread;
use std::time::Duration;
use thiserror::Error;

pub const DEFAULT_MODEL: &str = "vllm-sr/d3-turbo";
pub const DEFAULT_ENDPOINT: &str = "https://decision.vllm-sr.ai/v1/systemone";

#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn from_env() -> Result<Self, ApiError> {
        let val = std::env::var("VSR_API_KEY").map_err(|_| ApiError::MissingApiKey)?;
        if val.trim().is_empty() {
            return Err(ApiError::MissingApiKey);
        }
        Ok(Self(val))
    }

    pub fn new(val: impl Into<String>) -> Self {
        Self(val.into())
    }

    pub(crate) fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(***)")
    }
}

impl fmt::Display for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(***)")
    }
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("VSR_API_KEY environment variable is missing or empty")]
    MissingApiKey,
    #[error("A box question requires exactly one image in the request (got {0})")]
    InvalidImageCountForBox(usize),
    #[error("HTTP transport error: {0}")]
    Transport(String),
    #[error("API error (status {status}, type {error_type}, code {code}): {message}")]
    RemoteError {
        status: u16,
        error_type: String,
        code: String,
        message: String,
    },
    #[error("Unexpected response format (status {status}): {details}")]
    Decode { status: u16, details: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Choice {
        instructions: String,
        criteria: BTreeMap<String, Option<String>>,
    },
    Box {
        instructions: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Request {
    pub model: String,
    pub state: String,
    pub images: Vec<String>,
    pub questions: BTreeMap<String, Question>,
}

impl Request {
    pub fn validate(&self) -> Result<(), ApiError> {
        let has_box = self
            .questions
            .values()
            .any(|q| matches!(q, Question::Box { .. }));
        if has_box && self.images.len() != 1 {
            return Err(ApiError::InvalidImageCountForBox(self.images.len()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BoundingBox {
    pub top: f64,
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
}

impl BoundingBox {
    pub fn center(&self) -> (f64, f64) {
        ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        #[serde(default)]
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Box {
        #[serde(rename = "box")]
        bbox: BoundingBox,
        confidence: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Response {
    pub answers: BTreeMap<String, Answer>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    #[serde(rename = "type", default)]
    pub error_type: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub message: String,
}

pub trait DecisionBackend {
    fn ask(&self, req: &Request) -> Result<Response, ApiError>;
}

pub struct DecisionClient {
    api_key: ApiKey,
    endpoint: String,
    http: Client,
    max_retries: usize,
}

impl fmt::Debug for DecisionClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionClient")
            .field("api_key", &self.api_key)
            .field("endpoint", &self.endpoint)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl DecisionClient {
    pub fn new(api_key: ApiKey) -> Result<Self, ApiError> {
        Self::with_endpoint(api_key, DEFAULT_ENDPOINT)
    }

    pub fn with_endpoint(api_key: ApiKey, endpoint: impl Into<String>) -> Result<Self, ApiError> {
        let secret = api_key.expose_secret().to_string();
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| ApiError::Transport(sanitize_secret(&e.to_string(), &secret)))?;
        Ok(Self {
            api_key,
            endpoint: endpoint.into(),
            http,
            max_retries: 2,
        })
    }

    pub fn api_key(&self) -> &ApiKey {
        &self.api_key
    }
}

fn sanitize_secret(input: &str, secret: &str) -> String {
    if secret.is_empty() {
        input.to_string()
    } else {
        input.replace(secret, "***")
    }
}

impl DecisionBackend for DecisionClient {
    fn ask(&self, req: &Request) -> Result<Response, ApiError> {
        req.validate()?;
        let secret = self.api_key.expose_secret();
        let auth_value = format!("Bearer {}", secret);

        let mut attempt = 0usize;
        loop {
            let resp_res = self
                .http
                .post(&self.endpoint)
                .header(AUTHORIZATION, &auth_value)
                .header(CONTENT_TYPE, "application/json")
                .json(req)
                .send();

            let resp = match resp_res {
                Ok(r) => r,
                Err(e) => {
                    return Err(ApiError::Transport(sanitize_secret(&e.to_string(), secret)));
                }
            };

            let status = resp.status().as_u16();
            let retry_after_secs = resp
                .headers()
                .get(RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(1);

            let body_text = resp
                .text()
                .map_err(|e| ApiError::Transport(sanitize_secret(&e.to_string(), secret)))?;

            if (status == 429 || (500..=599).contains(&status)) && attempt < self.max_retries {
                attempt += 1;
                thread::sleep(Duration::from_secs(retry_after_secs.min(10)));
                continue;
            }

            if (200..300).contains(&status) {
                let parsed: Response = serde_json::from_str(&body_text).map_err(|e| {
                    ApiError::Decode {
                        status,
                        details: sanitize_secret(&e.to_string(), secret),
                    }
                })?;
                return Ok(parsed);
            }

            if let Ok(env) = serde_json::from_str::<ErrorEnvelope>(&body_text) {
                return Err(ApiError::RemoteError {
                    status,
                    error_type: sanitize_secret(&env.error.error_type, secret),
                    code: sanitize_secret(&env.error.code, secret),
                    message: sanitize_secret(&env.error.message, secret),
                });
            }

            return Err(ApiError::Decode {
                status,
                details: sanitize_secret(&body_text, secret),
            });
        }
    }
}
