mod embedding;
mod error;
mod file_processor;
mod generate_input;
mod obsidian;

use crate::embedding::EmbeddingRequestBuilder;
use crate::file_processor::EmbeddingRow;
use crate::file_processor::EMBEDDING_FILE_PATH;
use crate::obsidian::Notice;

use anyhow::anyhow;
use anyhow::Context;
use anyhow::Result;
use embedding::EmbeddingRequest;
use embedding::EmbeddingResponse;
use error::SemanticSearchError;
use error::WrappedError;
use file_processor::FileProcessor;
use js_sys::JsString;
use log::debug;
use ndarray::Array1;
use obsidian::App;
use serde::Deserialize;
use serde::Serialize;
use wasm_bindgen::prelude::*;
use crate::obsidian::SemanticSearchSettings;

use crate::embedding::EmbeddingInput;

#[wasm_bindgen]
pub struct GenerateEmbeddingsCommand {
    file_processor: FileProcessor,
    client: Client,
    num_batches: u32,
}

#[wasm_bindgen]
pub struct CostEstimateResponse {
    pub nfiles: i64,
    pub cost: f32,
}

#[wasm_bindgen]
impl GenerateEmbeddingsCommand {
    #[wasm_bindgen(constructor)]
    pub fn new(app: App, settings: SemanticSearchSettings) -> GenerateEmbeddingsCommand {
        let file_processor = FileProcessor::new(app.vault());
        let client = Client::new();
        let num_batches = settings.numBatches();
        GenerateEmbeddingsCommand {
            file_processor,
            client,
            num_batches,
        }
    }

    pub async fn get_embeddings(&self) -> Result<(), SemanticSearchError> {
        let (_, modified_input, mut reusable_embeddings) =
            self.file_processor.read_modified_input().await?;
        self.file_processor.delete_embeddings().await?;

        let mut num_processed = 0;
        let num_batches = self.num_batches;
        let mut batch = 1;
        let num_records = modified_input.len();
        debug!("Found {} records.", num_records);
        let batch_size = (num_records as f64 / num_batches as f64).ceil() as usize;
        let mut with_headers = true;

        while num_processed < num_records {
            let num_to_process = if batch == num_batches {
                num_records - num_processed
            } else {
                batch_size
            };

            let records = &modified_input[num_processed..num_processed + num_to_process].to_vec();
            debug!(
                "Processing batch {}: {} to {}",
                batch,
                num_processed,
                num_processed + num_to_process
            );

            let request = self.client.create_embedding_request(records.into())?;
            let response = self.client.post_embedding_request(&request).await?;
            debug!("Sucessfully obtained {} embeddings", response.data.len());

            if records.len() != response.data.len() {
                return Err(SemanticSearchError(anyhow!(
                    "Requested for {} embeddings but got {}",
                    records.len(),
                    response.data.len()
                )));
            }

            let mut embedding_rows: Vec<EmbeddingRow> = Vec::with_capacity(num_to_process);

            records.into_iter().enumerate().for_each(|(i, record)| {
                let embedding = response
                    .data
                    .get(i)
                    .map(|res| {
                        res.embedding
                            .clone()
                            .into_iter()
                            .map(|f| f.to_string())
                            .collect::<Vec<String>>()
                            .join(",")
                    })
                    .expect("Length of records and response data should be aligned");

                embedding_rows.push(EmbeddingRow {
                    name: record.name.to_string(),
                    mtime: record.mtime.to_string(),
                    header: record.section.to_string(),
                    embedding,
                });
            });

            embedding_rows.append(&mut reusable_embeddings);
            self.file_processor
                .write_embedding_csv(embedding_rows, with_headers)
                .await?;

            num_processed += num_to_process;
            batch += 1;
            with_headers = false;
        }

        debug!("Saved embeddings to {}", EMBEDDING_FILE_PATH);
        Ok(())
    }

    pub async fn check_embedding_file_exists(&self) -> bool {
        return self
            .file_processor
            .check_file_exists_at_path(EMBEDDING_FILE_PATH)
            .await;
    }
}

#[wasm_bindgen]
pub struct QueryCommand {
    file_processor: FileProcessor,
    client: Client,
}

struct Embedding<'a> {
    row: &'a EmbeddingRow,
    score: f32,
}

#[wasm_bindgen]
impl QueryCommand {
    async fn get_similarity(&self, query: String) -> Result<Vec<Suggestions>, SemanticSearchError> {
        let rows = self.file_processor.read_embedding_csv().await?;
        let response = self.client.get_embedding(query.into()).await?;
        debug!("Sucessfully obtained {} embeddings", response.data.len());
        let query_embedding = response.data[0].clone().embedding;

        let mut embeddings: Vec<Embedding> = Vec::with_capacity(rows.len());
        for row in &rows {
            let deserialized = deserialize_embeddings(&row.embedding).with_context(|| format!("Failed to deserialize embedding for file: {} and section: {} with embedding: {}", &row.name, &row.header, &row.embedding))?;
            embeddings.push(Embedding {
                score: cosine_similarity(query_embedding.clone(), deserialized),
                row: &row,
            });
        }

        embeddings.sort_unstable_by(|row1: &Embedding, row2: &Embedding| {
            row1.score
                .partial_cmp(&row2.score)
                .expect("scores should be comparable")
        });
        embeddings.reverse();
        let ranked = embeddings
            .iter()
            .map(|e| Suggestions {
                name: e.row.name.to_string(),
                header: e.row.header.to_string(),
            })
            .collect();
        Ok(ranked)
    }
}

fn deserialize_embeddings(embedding: &str) -> Result<Vec<f32>> {
    embedding
        .split(",")
        .map(|s| {
            s.parse::<f32>()
                .context("Embedding should be comma-separated list of f32")
        })
        .collect()
}

fn cosine_similarity(left: Vec<f32>, right: Vec<f32>) -> f32 {
    let a1 = Array1::from_vec(left);
    let a2 = Array1::from_vec(right);
    a1.dot(&a2) / a1.dot(&a1).sqrt() * a2.dot(&a2).sqrt()
}

#[derive(Deserialize, Serialize)]
pub struct Suggestions {
    name: String,
    header: String,
}

#[wasm_bindgen]
pub async fn get_suggestions(
    app: &obsidian::App,
    query: JsString,
) -> Result<JsValue, JsError> {
    let query_string = query.as_string().unwrap();
    let file_processor = FileProcessor::new(app.vault());
    let client = Client::new();
    let query_cmd = QueryCommand {
        file_processor,
        client,
    };
    let mut ranked_suggestions = query_cmd.get_similarity(query_string).await?;
    ranked_suggestions.truncate(10);
    Ok(serde_wasm_bindgen::to_value(&ranked_suggestions)?)
}

#[derive(Debug, Clone)]
pub struct Client {
    api_base: String,
}

/// Default v1 API base url
pub const API_BASE: &str = "http://localhost:10554";

impl Client {
    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    fn new() -> Self {
        Self {
            api_base: API_BASE.to_string(),
        }
    }

    pub async fn get_embedding(
        &self,
        input: EmbeddingInput,
    ) -> Result<EmbeddingResponse, SemanticSearchError> {
        let request = self.create_embedding_request(input)?;
        let response = self.post_embedding_request(request).await?;
        Ok(response)
    }

    fn create_embedding_request(&self, input: EmbeddingInput) -> Result<EmbeddingRequest> {
        let embedding_request = EmbeddingRequestBuilder::default()
            .url(self.api_base().to_string())
            .input(input)
            .build()
            .context("Failed to build embedding request")?;
        Ok(embedding_request)
    }

    async fn post_embedding_request<I: serde::ser::Serialize>(
        &self,
        request: I,
    ) -> Result<EmbeddingResponse> {
        let path = "/embeddings";
        let url = format!("{}{path}", self.api_base());

        let request = reqwest::Client::new().post(&url).json(&request).build()?;

        let reqwest_client = reqwest::Client::new();
        let response = reqwest_client
            .execute(request)
            .await
            .context(format!("Failed POST request to {}", url))?;

        let status = response.status();
        let bytes = response.bytes().await?;

        if !status.is_success() {
            let wrapped_error: WrappedError = serde_json::from_slice(bytes.as_ref())?;
            return Err(anyhow!(wrapped_error));
        }

        let response: EmbeddingResponse = serde_json::from_slice(bytes.as_ref())
            .context("Failed deserializing embedding response")?;
        Ok(response)
    }
}

#[wasm_bindgen]
pub fn onload(_plugin: &obsidian::Plugin) {
    console_log::init_with_level(log::Level::Debug).expect("");
    debug!("Semantic Search Loaded!");
}
