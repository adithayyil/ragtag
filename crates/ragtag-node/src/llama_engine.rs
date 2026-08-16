//! Thin HTTP client for an external llama.cpp server.
//!
//! Rather than linking llama.cpp into the Rust binary, this engine talks to
//! `llama-server` (or any OpenAI-compatible llama.cpp endpoint) over HTTP.
//! The user is responsible for starting the server with the desired model
//! before this engine is invoked.
//!
//! This is intentionally a full-model generation step: true pipeline-parallel
//! layer splitting would require changes inside llama.cpp itself, which is out
//! of scope for this scaffold.

use crate::Engine;
use anyhow::{Context, Result};
use async_trait::async_trait;
use ragtag_proto::pb::ActivationChunk;
use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct LlamaEngine {
    client: Client,
    url: Arc<String>,
}

#[derive(Deserialize)]
struct ChatCompletionChoice {
    message: AssistantMessage,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

impl LlamaEngine {
    pub fn new(server_url: String) -> Self {
        Self {
            client: Client::new(),
            url: Arc::new(server_url.trim_end_matches('/').to_string()),
        }
    }
}

#[async_trait]
impl Engine for LlamaEngine {
    async fn forward(&self, chunk: ActivationChunk) -> Result<ActivationChunk> {
        let prompt = String::from_utf8_lossy(&chunk.payload).to_string();

        let body = serde_json::json!({
            "model": "local",
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
            "stream": false,
        });

        let response: ChatCompletionResponse = self
            .client
            .post(format!("{}/v1/chat/completions", self.url))
            .json(&body)
            .send()
            .await
            .context("call llama-server")?
            .error_for_status()
            .context("llama-server error")?
            .json()
            .await
            .context("decode llama-server response")?;

        let content = response
            .choices
            .into_iter()
            .next()
            .context("empty choices from llama-server")?
            .message
            .content;

        Ok(ActivationChunk {
            request_id: chunk.request_id,
            sequence_no: chunk.sequence_no + 1,
            payload: content.into_bytes(),
        })
    }
}
