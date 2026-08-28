from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import List, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import pandas as pd

from fine_tune_system_safes import run_fine_tuning
from predict import (
    DEFAULT_FINE_TUNED_ROOT,
    DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
    DEFAULT_PROMPT_INJECTION_ARTIFACT,
    DEFAULT_TOXICITY_ARTIFACT,
    get_fine_tuned_model_dir,
    load_artifact,
    load_fine_tuned_system_prompt,
    load_model,
    predict,
    predict_fine_tuned,
)
from preprocessing import DEFAULT_EMBEDDING_MODEL
from schemas import (
    BasePredictRequest,
    BatchPredictItem,
    BatchPredictRequest,
    BatchPredictResponse,
    FineTunedModelDetail,
    FineTunedModelSummary,
    FineTunedPredictRequest,
    FineTuneJsonRequest,
    FineTuneResponse,
    HealthResponse,
    PredictResponse,
    TaskPrediction,
)

API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up embedding model and verify base artifacts at startup."""
    try:
        load_model(DEFAULT_EMBEDDING_MODEL)
        load_artifact(str(DEFAULT_PROMPT_INJECTION_ARTIFACT.resolve()), "prompt_injection")
        load_artifact(str(DEFAULT_OUT_OF_CONTEXT_ARTIFACT.resolve()), "out_of_context")
        load_artifact(str(DEFAULT_TOXICITY_ARTIFACT.resolve()), "toxicity")
    except Exception as e:
        print(f"[Warning] Failed to warm up models during startup: {e}")
    yield


app = FastAPI(
    title="Open Guardrail Layer API",
    description=(
        "Production-ready Guardrail Layer API for LLMs.\n\n"
        "Detects **Prompt Injection**, **Out-of-Context queries**, and **Toxicity** "
        "using calibrated XGBoost classifiers with multi-lingual sentence embeddings.\n\n"
        "Supports both **Base Models** and **Domain-Adapted Fine-Tuned Models**."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Health & Service Metadata
# ==========================================


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns service health status and loaded model information."""
    base_ready = all(
        path.is_file()
        for path in (
            DEFAULT_PROMPT_INJECTION_ARTIFACT,
            DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
            DEFAULT_TOXICITY_ARTIFACT,
        )
    )
    fine_tuned_count = 0
    if DEFAULT_FINE_TUNED_ROOT.exists():
        fine_tuned_count = sum(
            1 for p in DEFAULT_FINE_TUNED_ROOT.iterdir() if p.is_dir()
        )

    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        service="open-guardrail-layer",
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        base_artifacts_ready=base_ready,
        fine_tuned_models_count=fine_tuned_count,
    )


# ==========================================
# Base Model Prediction Endpoints
# ==========================================


@app.post(
    "/v1/predict/base",
    response_model=PredictResponse,
    summary="Predict using Base Guardrail Models",
    tags=["Base Models"],
)
@app.post(
    "/v1/predict",
    response_model=PredictResponse,
    include_in_schema=False,
)
@app.post(
    "/predict",
    response_model=PredictResponse,
    include_in_schema=False,
)
async def predict_base_endpoint(request: BasePredictRequest):
    """
    Evaluates a user prompt against a given system prompt using the pre-trained base models.
    
    Checks for:
    - **Prompt Injection**: Attempts to hijack system instructions
    - **Out of Context**: User prompt diverging from the specified system prompt persona/domain
    - **Toxicity**: Harmful, toxic, or abusive language
    """
    if not request.system_prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="system_prompt must not be empty",
        )
    if not request.user_prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_prompt must not be empty",
        )

    start_time = perf_counter()
    try:
        raw_result = predict(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
        )
        latency = round((perf_counter() - start_time) * 1000, 3)

        return PredictResponse(
            model_type="base",
            model_name=None,
            decision=raw_result["decision"],
            prompt_injection=TaskPrediction(**raw_result["prompt_injection"]),
            out_of_context=TaskPrediction(**raw_result["out_of_context"]),
            toxicity=TaskPrediction(**raw_result["toxicity"]),
            system_prompt_used=request.system_prompt,
            execution_time_ms=latency,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )


# ==========================================
# Fine-Tuned Model Prediction Endpoints
# ==========================================


@app.post(
    "/v1/predict/fine-tuned/{model_name}",
    response_model=PredictResponse,
    summary="Predict using a specific Fine-Tuned Model (URL path)",
    tags=["Fine-Tuned Models"],
)
async def predict_fine_tuned_path_endpoint(
    model_name: str, request: FineTunedPredictRequest
):
    """
    Evaluates a prompt using a fine-tuned model specified in the URL path.
    
    If `system_prompt` is not provided in the request body, the model's saved
    `system_prompt.txt` will automatically be loaded and used.
    """
    return _execute_fine_tuned_predict(
        model_name=model_name,
        user_prompt=request.user_prompt,
        system_prompt_override=request.system_prompt,
    )


@app.post(
    "/v1/predict/fine-tuned",
    response_model=PredictResponse,
    summary="Predict using Fine-Tuned Model (Model Name in Body)",
    tags=["Fine-Tuned Models"],
)
async def predict_fine_tuned_body_endpoint(request: FineTunedPredictRequest):
    """
    Evaluates a prompt using a fine-tuned model specified in the request body.
    """
    if not request.model_name or not request.model_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_name must be provided in request body or path",
        )
    return _execute_fine_tuned_predict(
        model_name=request.model_name,
        user_prompt=request.user_prompt,
        system_prompt_override=request.system_prompt,
    )


def _execute_fine_tuned_predict(
    model_name: str,
    user_prompt: str,
    system_prompt_override: Optional[str],
) -> PredictResponse:
    if not user_prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_prompt must not be empty",
        )

    try:
        model_dir = get_fine_tuned_model_dir(model_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fine-tuned model '{model_name}' not found",
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )

    system_prompt = system_prompt_override
    if not system_prompt or not system_prompt.strip():
        system_prompt = load_fine_tuned_system_prompt(model_name)
        if not system_prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model '{model_name}' does not have a saved system_prompt.txt. "
                    "Please provide system_prompt in the request body."
                ),
            )

    start_time = perf_counter()
    try:
        raw_result = predict_fine_tuned(
            model_name=model_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        latency = round((perf_counter() - start_time) * 1000, 3)

        return PredictResponse(
            model_type="fine_tuned",
            model_name=model_name,
            decision=raw_result["decision"],
            prompt_injection=TaskPrediction(**raw_result["prompt_injection"]),
            out_of_context=TaskPrediction(**raw_result["out_of_context"]),
            toxicity=TaskPrediction(**raw_result["toxicity"]),
            system_prompt_used=raw_result.get("system_prompt_used", system_prompt),
            execution_time_ms=latency,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fine-tuned inference error: {exc}",
        )


# ==========================================
# Batch Prediction Endpoint
# ==========================================


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch Evaluation of Multiple Prompts",
    tags=["Batch Predictions"],
)
async def batch_predict_endpoint(request: BatchPredictRequest):
    """
    Evaluates multiple user prompts in batch using either base or fine-tuned models.
    """
    start_time = perf_counter()
    results: List[PredictResponse] = []

    for item in request.items:
        if request.model_type == "fine_tuned":
            if not request.model_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="model_name is required when model_type is 'fine_tuned'",
                )
            res = _execute_fine_tuned_predict(
                model_name=request.model_name,
                user_prompt=item.user_prompt,
                system_prompt_override=item.system_prompt,
            )
        else:
            if not item.system_prompt or not item.system_prompt.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="system_prompt is required for base model batch items",
                )
            raw = predict(
                system_prompt=item.system_prompt,
                user_prompt=item.user_prompt,
            )
            res = PredictResponse(
                model_type="base",
                model_name=None,
                decision=raw["decision"],
                prompt_injection=TaskPrediction(**raw["prompt_injection"]),
                out_of_context=TaskPrediction(**raw["out_of_context"]),
                toxicity=TaskPrediction(**raw["toxicity"]),
                system_prompt_used=item.system_prompt,
                execution_time_ms=0.0,
            )
        results.append(res)

    total_latency = round((perf_counter() - start_time) * 1000, 3)
    blocked_count = sum(1 for r in results if r.decision == "blocked")

    return BatchPredictResponse(
        model_type=request.model_type,
        model_name=request.model_name,
        total_items=len(results),
        total_blocked=blocked_count,
        total_safe=len(results) - blocked_count,
        execution_time_ms=total_latency,
        results=results,
    )


# ==========================================
# Fine-Tuned Models Management
# ==========================================


@app.get(
    "/v1/models/fine-tuned",
    response_model=List[FineTunedModelSummary],
    summary="List All Fine-Tuned Models",
    tags=["Fine-Tuned Management"],
)
async def list_fine_tuned_models_endpoint():
    """Lists all available fine-tuned models and their statuses."""
    if not DEFAULT_FINE_TUNED_ROOT.exists():
        return []

    summaries: List[FineTunedModelSummary] = []
    for model_dir in sorted(DEFAULT_FINE_TUNED_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue

        artifacts = [p.name for p in sorted(model_dir.glob("*_model.pkl"))]
        prompt_path = model_dir / "system_prompt.txt"
        has_prompt = prompt_path.is_file()
        preview = None
        if has_prompt:
            content = prompt_path.read_text(encoding="utf-8").strip()
            preview = (content[:120] + "...") if len(content) > 120 else content

        created_ts = datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat()

        summaries.append(
            FineTunedModelSummary(
                model_name=model_dir.name,
                has_system_prompt=has_prompt,
                system_prompt_preview=preview,
                artifacts_ready=len(artifacts),
                total_expected_artifacts=3,
                artifacts_list=artifacts,
                created_at=created_ts,
            )
        )

    return summaries


@app.get(
    "/v1/models/fine-tuned/{model_name}",
    response_model=FineTunedModelDetail,
    summary="Get Fine-Tuned Model Details",
    tags=["Fine-Tuned Management"],
)
async def get_fine_tuned_model_endpoint(model_name: str):
    """Retrieves full details for a specific fine-tuned model."""
    try:
        model_dir = get_fine_tuned_model_dir(model_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fine-tuned model '{model_name}' not found",
        )

    artifacts = [p.name for p in sorted(model_dir.glob("*_model.pkl"))]
    prompt_path = model_dir / "system_prompt.txt"
    system_prompt = (
        prompt_path.read_text(encoding="utf-8").strip()
        if prompt_path.is_file()
        else None
    )
    all_files = [p.name for p in sorted(model_dir.iterdir()) if p.is_file()]

    return FineTunedModelDetail(
        model_name=model_name,
        has_system_prompt=system_prompt is not None,
        system_prompt=system_prompt,
        artifacts_ready=len(artifacts),
        total_expected_artifacts=3,
        artifacts_list=artifacts,
        path=str(model_dir.resolve()),
        files=all_files,
    )


@app.delete(
    "/v1/models/fine-tuned/{model_name}",
    summary="Delete a Fine-Tuned Model",
    tags=["Fine-Tuned Management"],
)
async def delete_fine_tuned_model_endpoint(model_name: str):
    """Deletes a fine-tuned model and its artifacts from disk."""
    try:
        model_dir = get_fine_tuned_model_dir(model_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fine-tuned model '{model_name}' not found",
        )

    shutil.rmtree(model_dir)
    return {
        "status": "deleted",
        "model_name": model_name,
        "message": f"Fine-tuned model '{model_name}' successfully removed",
    }


# ==========================================
# Fine-Tuning Execution Endpoints
# ==========================================


@app.post(
    "/v1/models/fine-tune",
    response_model=FineTuneResponse,
    summary="Create Fine-Tuned Model via Multipart Upload",
    tags=["Fine-Tuning"],
)
async def fine_tune_multipart_endpoint(
    model_name: str = Form(..., description="Unique name for the fine-tuned model"),
    system_prompt: Optional[str] = Form(None, description="System prompt text"),
    system_prompt_file: Optional[UploadFile] = File(
        None, description="System prompt as .txt file"
    ),
    safe_file: UploadFile = File(
        ..., description="Excel file (.xlsx) with safe user prompt examples"
    ),
    out_of_context_file: Optional[UploadFile] = File(
        None,
        description="Optional Excel file (.xlsx) with out-of-context examples",
    ),
    safe_column: str = Form("UserMessage", description="Column name for safe examples"),
    out_of_context_column: str = Form(
        "UserMessage", description="Column name for out-of-context examples"
    ),
    rehearsal_per_class: int = Form(800, description="Rehearsal count per class"),
    fine_tune_estimators: int = Form(80, description="Number of additional estimators"),
):
    """
    Executes fine-tuning for a custom domain model using uploaded Excel files.
    """
    clean_name = model_name.strip()
    if not clean_name or Path(clean_name).name != clean_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_name must be a simple alphanumeric name without directory separators",
        )

    prompt_text = (system_prompt or "").strip()
    if system_prompt_file and not prompt_text:
        content_bytes = await system_prompt_file.read()
        prompt_text = content_bytes.decode("utf-8").strip()

    if not prompt_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either system_prompt or system_prompt_file must be provided and non-empty",
        )

    start_time = perf_counter()
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        safe_path = temp_path / safe_file.filename
        safe_path.write_bytes(await safe_file.read())

        system_path = temp_path / "system_prompt.txt"
        system_path.write_text(prompt_text, encoding="utf-8")

        kwargs = {
            "safe_column": safe_column,
            "rehearsal_per_class": rehearsal_per_class,
            "fine_tune_estimators": fine_tune_estimators,
            "system_prompt_source_path": system_path,
        }

        if out_of_context_file:
            ooc_path = temp_path / out_of_context_file.filename
            ooc_path.write_bytes(await out_of_context_file.read())
            kwargs["out_of_context_excel_paths"] = [ooc_path]
            kwargs["out_of_context_column"] = out_of_context_column

        try:
            result = run_fine_tuning(
                system_name=clean_name,
                system_prompt=prompt_text,
                safe_excel_paths=[safe_path],
                **kwargs,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Fine-tuning process failed: {exc}",
            )

    latency = round((perf_counter() - start_time) * 1000, 3)
    return FineTuneResponse(
        system_name=result["system_name"],
        safe_examples=result["safe_examples"],
        out_of_context_examples=result["out_of_context_examples"],
        output_dir=result["output_dir"],
        tasks=result["tasks"],
        execution_time_ms=latency,
        message=f"Fine-tuned model '{clean_name}' successfully created.",
    )


@app.post(
    "/v1/models/fine-tune/json",
    response_model=FineTuneResponse,
    summary="Create Fine-Tuned Model via JSON Payload",
    tags=["Fine-Tuning"],
)
async def fine_tune_json_endpoint(request: FineTuneJsonRequest):
    """
    Executes fine-tuning directly with JSON arrays of safe strings.
    """
    clean_name = request.model_name.strip()
    if not clean_name or Path(clean_name).name != clean_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_name must be a simple alphanumeric name without directory separators",
        )

    if not request.safe_examples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="safe_examples list cannot be empty",
        )

    start_time = perf_counter()
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        safe_df = pd.DataFrame({"UserMessage": request.safe_examples})
        safe_path = temp_path / "safe_examples.xlsx"
        safe_df.to_excel(safe_path, index=False)

        system_path = temp_path / "system_prompt.txt"
        system_path.write_text(request.system_prompt.strip(), encoding="utf-8")

        kwargs = {
            "safe_column": "UserMessage",
            "rehearsal_per_class": request.rehearsal_per_class,
            "fine_tune_estimators": request.fine_tune_estimators,
            "system_prompt_source_path": system_path,
        }

        if request.out_of_context_examples:
            ooc_df = pd.DataFrame({"UserMessage": request.out_of_context_examples})
            ooc_path = temp_path / "out_of_context_examples.xlsx"
            ooc_df.to_excel(ooc_path, index=False)
            kwargs["out_of_context_excel_paths"] = [ooc_path]
            kwargs["out_of_context_column"] = "UserMessage"

        try:
            result = run_fine_tuning(
                system_name=clean_name,
                system_prompt=request.system_prompt.strip(),
                safe_excel_paths=[safe_path],
                **kwargs,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Fine-tuning process failed: {exc}",
            )

    latency = round((perf_counter() - start_time) * 1000, 3)
    return FineTuneResponse(
        system_name=result["system_name"],
        safe_examples=result["safe_examples"],
        out_of_context_examples=result["out_of_context_examples"],
        output_dir=result["output_dir"],
        tasks=result["tasks"],
        execution_time_ms=latency,
        message=f"Fine-tuned model '{clean_name}' successfully created.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
