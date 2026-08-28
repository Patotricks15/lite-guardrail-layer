from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TaskPrediction(BaseModel):
    probability: float = Field(..., description="Estimated probability score [0.0, 1.0]")
    prediction: Literal["detected", "safe"] = Field(
        ..., description="Binary classification outcome based on threshold"
    )
    threshold: float = Field(..., description="Decision threshold applied")


class BasePredictRequest(BaseModel):
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System prompt defining the context and boundaries for the assistant",
        examples=["You are a helpful customer support assistant for Acme Corp."],
    )
    user_prompt: str = Field(
        ...,
        min_length=1,
        description="User input prompt to be inspected for safety issues",
        examples=["Can you help me reset my account password?"],
    )


class FineTunedPredictRequest(BaseModel):
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the fine-tuned model (required if not in URL path)",
        examples=["customer_support_v1"],
    )
    user_prompt: str = Field(
        ...,
        min_length=1,
        description="User input prompt to be inspected for safety issues",
        examples=["Can you help me reset my account password?"],
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt override. If omitted, uses the saved system_prompt.txt from fine-tuning.",
        examples=["You are a customer support agent."],
    )


class PredictResponse(BaseModel):
    model_type: Literal["base", "fine_tuned"] = Field(
        ..., description="Type of model used for evaluation ('base' or 'fine_tuned')"
    )
    model_name: Optional[str] = Field(
        default=None, description="Name of the fine-tuned model (if applicable)"
    )
    decision: Literal["blocked", "safe"] = Field(
        ..., description="Overall decision: 'blocked' if any threat is detected, else 'safe'"
    )
    prompt_injection: TaskPrediction = Field(
        ..., description="Prompt injection detection result"
    )
    out_of_context: TaskPrediction = Field(
        ..., description="Out-of-context detection result"
    )
    toxicity: TaskPrediction = Field(..., description="Toxicity detection result")
    system_prompt_used: Optional[str] = Field(
        default=None, description="The system prompt used during prediction"
    )
    execution_time_ms: float = Field(
        ..., description="Inference latency in milliseconds"
    )


class BatchPredictItem(BaseModel):
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt (required for base model; optional for fine-tuned if saved)",
    )
    user_prompt: str = Field(..., min_length=1, description="User prompt to inspect")


class BatchPredictRequest(BaseModel):
    model_type: Literal["base", "fine_tuned"] = Field(
        default="base", description="Whether to evaluate with 'base' or 'fine_tuned' models"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model name (required if model_type is 'fine_tuned')",
    )
    items: List[BatchPredictItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of prompt pairs to evaluate in batch",
    )


class BatchPredictResponse(BaseModel):
    model_type: Literal["base", "fine_tuned"]
    model_name: Optional[str] = None
    total_items: int
    total_blocked: int
    total_safe: int
    execution_time_ms: float
    results: List[PredictResponse]


class FineTunedModelSummary(BaseModel):
    model_name: str
    has_system_prompt: bool
    system_prompt_preview: Optional[str] = None
    artifacts_ready: int
    total_expected_artifacts: int
    artifacts_list: List[str]
    created_at: Optional[str] = None


class FineTunedModelDetail(BaseModel):
    model_name: str
    has_system_prompt: bool
    system_prompt: Optional[str] = None
    artifacts_ready: int
    total_expected_artifacts: int
    artifacts_list: List[str]
    path: str
    files: List[str]


class FineTuneJsonRequest(BaseModel):
    model_name: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the fine-tuned model",
        examples=["support_agent"],
    )
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System prompt defining behavior and domain boundaries",
    )
    safe_examples: List[str] = Field(
        ...,
        min_length=1,
        description="List of domain-specific in-context user prompts considered safe",
    )
    out_of_context_examples: Optional[List[str]] = Field(
        default=None,
        description="Optional list of out-of-context prompts for custom out-of-context tuning",
    )
    rehearsal_per_class: int = Field(
        default=800,
        ge=50,
        le=5000,
        description="Number of rehearsal samples per class during fine-tuning",
    )
    fine_tune_estimators: int = Field(
        default=80,
        ge=10,
        le=500,
        description="Number of additional estimators for fine-tuning",
    )


class FineTuneResponse(BaseModel):
    system_name: str
    safe_examples: int
    out_of_context_examples: int
    output_dir: str
    tasks: Dict[str, Any]
    execution_time_ms: float
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    embedding_model: str
    base_artifacts_ready: bool
    fine_tuned_models_count: int
