import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from functools import lru_cache
from pathlib import Path

import joblib

from preprocessing import (
    OUT_OF_CONTEXT_FEATURE_DIMENSION,
    PATTERN_FEATURE_NAMES,
    RELATIONSHIP_METRICS_DIMENSION,
    RELATIONAL_FEATURE_DIMENSION,
    build_out_of_context_features,
    build_user_prompt_features,
    encode_prompts,
    extract_pattern_features,
    load_embedding_model,
    USER_PROMPT_FEATURE_DIMENSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_INJECTION_ARTIFACT = (
    PROJECT_ROOT / "artifact" / "prompt_injection_model.pkl"
)
DEFAULT_OUT_OF_CONTEXT_ARTIFACT = (
    PROJECT_ROOT / "artifact" / "out_of_context_model.pkl"
)
DEFAULT_TOXICITY_ARTIFACT = PROJECT_ROOT / "artifact" / "toxicity_model.pkl"


@lru_cache(maxsize=3)
def load_artifact(artifact_path: str, expected_task: str):
    artifact = joblib.load(artifact_path)
    required_keys = {
        "task",
        "classifier",
        "embedding_model_name",
        "feature_dimension",
        "include_pattern_features",
        "relationship_metrics_only",
        "user_prompt_only",
        "pattern_feature_names",
        "recommended_threshold",
    }
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"invalid model artifact; missing keys: {missing}")
    if artifact["task"] != expected_task:
        raise ValueError(
            f"expected {expected_task!r} artifact, got {artifact['task']!r}"
        )

    expected_dimension = (
        USER_PROMPT_FEATURE_DIMENSION
        if artifact["user_prompt_only"]
        else (
            OUT_OF_CONTEXT_FEATURE_DIMENSION
            if artifact.get("out_of_context_features", False)
            else (
                RELATIONSHIP_METRICS_DIMENSION
                if artifact["relationship_metrics_only"]
                else RELATIONAL_FEATURE_DIMENSION
            )
        )
    )
    if artifact["feature_dimension"] != expected_dimension:
        raise ValueError("artifact feature dimension does not match preprocessing")
    if expected_task in {"prompt_injection", "toxicity"} and not artifact[
        "user_prompt_only"
    ]:
        raise ValueError(f"{expected_task} artifact must be user-prompt-only")
    if expected_task == "out_of_context" and not artifact.get(
        "out_of_context_features", False
    ):
        raise ValueError(
            "out-of-context artifact must use the configured feature schema"
        )
    if artifact["include_pattern_features"]:
        if tuple(artifact["pattern_feature_names"]) != PATTERN_FEATURE_NAMES:
            raise ValueError("artifact pattern features do not match preprocessing")
    return artifact


@lru_cache(maxsize=1)
def load_model(model_name: str):
    return load_embedding_model(model_name)


def classify(artifact, features) -> dict[str, float | str]:
    probability = float(artifact["classifier"].predict_proba(features)[0, 1])
    threshold = float(artifact["recommended_threshold"])
    prediction = "detected" if probability >= threshold else "safe"
    return {
        "probability": probability,
        "prediction": prediction,
        "threshold": threshold,
    }


def predict(
    system_prompt: str,
    user_prompt: str,
    prompt_injection_artifact_path: Path = DEFAULT_PROMPT_INJECTION_ARTIFACT,
    out_of_context_artifact_path: Path = DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
    toxicity_artifact_path: Path = DEFAULT_TOXICITY_ARTIFACT,
) -> dict:
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")

    prompt_injection_artifact = load_artifact(
        str(prompt_injection_artifact_path.resolve()),
        "prompt_injection",
    )
    out_of_context_artifact = load_artifact(
        str(out_of_context_artifact_path.resolve()),
        "out_of_context",
    )
    toxicity_artifact = load_artifact(
        str(toxicity_artifact_path.resolve()),
        "toxicity",
    )
    artifacts = (
        prompt_injection_artifact,
        out_of_context_artifact,
        toxicity_artifact,
    )
    embedding_model_names = {
        artifact["embedding_model_name"] for artifact in artifacts
    }
    if len(embedding_model_names) != 1:
        raise ValueError("all artifacts must use the same embedding model")

    embedding_model = load_model(prompt_injection_artifact["embedding_model_name"])
    system_embeddings = encode_prompts(embedding_model, [system_prompt])
    user_embeddings = encode_prompts(embedding_model, [user_prompt])
    out_of_context_features = build_out_of_context_features(
        system_embeddings,
        user_embeddings,
        system_prompts=[system_prompt],
        user_prompts=[user_prompt],
    )
    prompt_injection_features = build_user_prompt_features(
        user_embeddings,
        extract_pattern_features([user_prompt]),
    )

    prediction_inputs = {
        "prompt_injection": (
            prompt_injection_artifact,
            prompt_injection_features,
        ),
        "out_of_context": (
            out_of_context_artifact,
            out_of_context_features,
        ),
        "toxicity": (
            toxicity_artifact,
            prompt_injection_features,
        ),
    }
    with ThreadPoolExecutor(max_workers=len(prediction_inputs)) as executor:
        futures = {
            task: executor.submit(classify, artifact, features)
            for task, (artifact, features) in prediction_inputs.items()
        }
        results = {task: future.result() for task, future in futures.items()}

    blocked = any(
        result["prediction"] == "detected"
        for result in results.values()
    )
    return {
        **results,
        "decision": "blocked" if blocked else "safe",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect prompt injection, out-of-context, and toxic requests."
    )
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--user-prompt", required=True)
    parser.add_argument(
        "--prompt-injection-artifact",
        type=Path,
        default=DEFAULT_PROMPT_INJECTION_ARTIFACT,
    )
    parser.add_argument(
        "--out-of-context-artifact",
        type=Path,
        default=DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
    )
    parser.add_argument(
        "--toxicity-artifact",
        type=Path,
        default=DEFAULT_TOXICITY_ARTIFACT,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = predict(
        arguments.system_prompt,
        arguments.user_prompt,
        arguments.prompt_injection_artifact,
        arguments.out_of_context_artifact,
        arguments.toxicity_artifact,
    )
    print(json.dumps(result))
