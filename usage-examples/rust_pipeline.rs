//! Lite Guardrail Layer - Rust Pipeline Example.
//!
//! Demonstrates integrating prompt verification before invoking an LLM.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct PredictRequest<'a> {
    pub system_prompt: &'a str,
    pub user_prompt: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct PredictResponse {
    pub decision: String,
    pub execution_time_ms: f64,
}

pub struct GuardrailClient {
    api_base: String,
}

impl GuardrailClient {
    pub fn new(api_base: impl Into<String>) -> Self {
        Self { api_base: api_base.into() }
    }

    pub async fn check_prompt(&self, system_prompt: &str, user_prompt: &str) -> Result<PredictResponse, Box<dyn std::error::Error>> {
        let url = format!("{}/v1/predict/base", self.api_base);
        let client = reqwest::Client::new();
        let payload = PredictRequest { system_prompt, user_prompt };

        let resp = client.post(&url)
            .json(&payload)
            .send()
            .await?
            .json::<PredictResponse>()
            .await?;

        if resp.decision == "blocked" {
            return Err("Guardrail blocked prompt due to safety policy violation".into());
        }

        Ok(resp)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Lite Guardrail Layer - Rust Example ===");
    let guardrail = GuardrailClient::new("http://localhost:8000");

    let system_prompt = "You are a customer assistant.";
    let user_prompt = "How can I track my order?";

    println!("\n[1] Screening prompt: '{}'", user_prompt);
    match guardrail.check_prompt(system_prompt, user_prompt).await {
        Ok(res) => {
            println!(" -> Passed guardrail ({} ms). Forwarding to LLM...", res.execution_time_ms);
        }
        Err(e) => {
            println!(" -> [BLOCKED]: {}", e);
        }
    }

    Ok(())
}
