import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from predict import (
    DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
    DEFAULT_PROMPT_INJECTION_ARTIFACT,
    DEFAULT_TOXICITY_ARTIFACT,
    load_artifact,
    load_model,
)
from preprocessing import (
    preprocess_prompt_pairs,
    preprocess_user_prompts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifact" / "fine_tuned"


@dataclass(frozen=True)
class TaskConfig:
    name: str
    target_column: str
    data_path: Path
    artifact_path: Path
    user_prompt_only: bool
    include_pattern_features: bool
    relationship_metrics_only: bool
    out_of_context_features: bool


TASKS = {
    "prompt_injection": TaskConfig(
        name="prompt_injection",
        target_column="prompt_injection",
        data_path=DATA_DIR / "prompt_injection.parquet",
        artifact_path=DEFAULT_PROMPT_INJECTION_ARTIFACT,
        user_prompt_only=True,
        include_pattern_features=True,
        relationship_metrics_only=False,
        out_of_context_features=False,
    ),
    "out_of_context": TaskConfig(
        name="out_of_context",
        target_column="out_of_context",
        data_path=DATA_DIR / "out-of-context.parquet",
        artifact_path=DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
        user_prompt_only=False,
        include_pattern_features=False,
        relationship_metrics_only=False,
        out_of_context_features=True,
    ),
    "toxicity": TaskConfig(
        name="toxicity",
        target_column="toxicity",
        data_path=DATA_DIR / "toxicity.parquet",
        artifact_path=DEFAULT_TOXICITY_ARTIFACT,
        user_prompt_only=True,
        include_pattern_features=True,
        relationship_metrics_only=False,
        out_of_context_features=False,
    ),
}


def run_fine_tuning(
    system_name: str,
    system_prompt: str,
    safe_excel_paths: list[Path],
    safe_column: str = "safe_examples",
    out_of_context_excel_paths: list[Path] | None = None,
    out_of_context_column: str = "UserMessage",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    tasks: list[str] | None = None,
    rehearsal_per_class: int = 800,
    fine_tune_estimators: int = 80,
    calibration_holdout: float = 0.20,
    seed: int = 42,
    progress_callback=None,
    artifact_paths: dict[str, Path] | None = None,
    system_prompt_source_path: Path | None = None,
) -> dict:
    system_name = system_name.strip()
    system_prompt = system_prompt.strip()
    safe_column = safe_column.strip()
    out_of_context_column = out_of_context_column.strip()
    if (
        not system_name
        or system_name in {".", ".."}
        or Path(system_name).name != system_name
    ):
        raise ValueError("system_name must be a non-empty directory name")
    if not system_prompt:
        raise ValueError("system_prompt must be non-empty")
    if not safe_column:
        raise ValueError("safe_column must be non-empty")
    if not out_of_context_column:
        raise ValueError("out_of_context_column must be non-empty")
    if rehearsal_per_class <= 0 or fine_tune_estimators <= 0:
        raise ValueError("fine-tuning sample and estimator counts must be positive")
    if not 0.05 <= calibration_holdout <= 0.50:
        raise ValueError("calibration_holdout must be between 0.05 and 0.50")

    selected_tasks = list(TASKS) if not tasks or "all" in tasks else list(tasks)
    safe_examples = load_safe_examples(safe_excel_paths, safe_column)
    out_of_context_examples = (
        load_safe_examples(out_of_context_excel_paths, out_of_context_column)
        if out_of_context_excel_paths
        else []
    )
    if "out_of_context" in selected_tasks and not out_of_context_examples:
        selected_tasks.remove("out_of_context")
    if not selected_tasks:
        raise ValueError("at least one fine-tuning task must be selected")
    artifact_path_by_task = artifact_paths or {
        "prompt_injection": DEFAULT_PROMPT_INJECTION_ARTIFACT,
        "out_of_context": DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
        "toxicity": DEFAULT_TOXICITY_ARTIFACT,
    }
    embedding_model_names = {
        load_artifact(str(artifact_path_by_task[task].resolve()), task)[
            "embedding_model_name"
        ]
        for task in selected_tasks
    }
    if len(embedding_model_names) != 1:
        raise ValueError("all selected artifacts must use the same embedding model")
    embedding_model = load_model(next(iter(embedding_model_names)))

    output_dir = output_root / system_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if system_prompt_source_path is not None:
        shutil.copy2(system_prompt_source_path, output_dir / "system_prompt.txt")
    else:
        (output_dir / "system_prompt.txt").write_text(
            system_prompt,
            encoding="utf-8",
        )
    for safe_excel_path in safe_excel_paths:
        shutil.copy2(safe_excel_path, output_dir / safe_excel_path.name)
    for out_of_context_excel_path in out_of_context_excel_paths or []:
        shutil.copy2(out_of_context_excel_path, output_dir / out_of_context_excel_path.name)
    summaries = {}
    skipped_tasks = set(TASKS).difference(selected_tasks)
    for task_name in skipped_tasks:
        artifact_path = artifact_path_by_task[task_name]
        output_path = output_dir / artifact_path.name
        shutil.copy2(artifact_path, output_path)
        summaries[task_name] = {
            "output_path": str(output_path.resolve()),
            "fine_tuning": {"status": "skipped"},
        }
    for index, task_name in enumerate(selected_tasks, start=1):
        if progress_callback:
            progress_callback(index - 1, len(selected_tasks), task_name)
        updated_artifact = fine_tune_task(
            config=TASKS[task_name],
            artifact_path=artifact_path_by_task[task_name],
            embedding_model=embedding_model,
            safe_examples=safe_examples,
            out_of_context_examples=out_of_context_examples,
            system_prompt=system_prompt,
            rehearsal_per_class=rehearsal_per_class,
            fine_tune_estimators=fine_tune_estimators,
            calibration_holdout=calibration_holdout,
            random_state=seed,
        )
        output_path = output_dir / artifact_path_by_task[task_name].name
        joblib.dump(updated_artifact, output_path)
        summaries[task_name] = {
            "output_path": str(output_path.resolve()),
            "fine_tuning": updated_artifact["fine_tuning"],
        }
        if progress_callback:
            progress_callback(index, len(selected_tasks), task_name)

    return {
        "system_name": system_name,
        "safe_examples": len(safe_examples),
        "out_of_context_examples": len(out_of_context_examples),
        "output_dir": str(output_dir.resolve()),
        "tasks": summaries,
    }


def _find_column_name(dataframe: pd.DataFrame, expected_column: str) -> str | None:
    if expected_column in dataframe.columns:
        return expected_column
    expected_lower = expected_column.strip().lower()
    for column in dataframe.columns:
        if str(column).strip().lower() == expected_lower:
            return str(column)
    return None


def load_safe_examples(excel_paths: list[Path], safe_column: str) -> list[str]:
    safe_examples: list[str] = []
    for path in excel_paths:
        if not path.exists():
            raise FileNotFoundError(f"safe examples file not found: {path}")
        dataframe = pd.read_excel(path)
        safe_column_name = _find_column_name(dataframe, safe_column)
        if safe_column_name is None:
            raise ValueError(
                f"file {path} is missing required column: {safe_column}"
            )
        values = dataframe[safe_column_name].dropna().astype(str).str.strip()
        safe_examples.extend([value for value in values.tolist() if value])
    deduplicated = list(dict.fromkeys(safe_examples))
    if not deduplicated:
        raise ValueError("no non-empty safe examples were found")
    return deduplicated


def build_features(
    config: TaskConfig,
    embedding_model,
    user_prompts: list[str],
    system_prompt: str,
) -> np.ndarray:
    if config.user_prompt_only:
        return preprocess_user_prompts(
            user_prompts,
            embedding_model,
            batch_size=64,
            show_progress_bar=False,
        )
    system_prompts = [system_prompt] * len(user_prompts)
    return preprocess_prompt_pairs(
        system_prompts,
        user_prompts,
        embedding_model,
        batch_size=64,
        show_progress_bar=False,
        include_pattern_features=config.include_pattern_features,
        relationship_metrics_only=config.relationship_metrics_only,
        out_of_context_features=config.out_of_context_features,
    )


def sample_rehearsal_prompts(
    config: TaskConfig,
    per_class: int,
    random_state: int,
) -> tuple[list[str], np.ndarray]:
    if not config.data_path.exists():
        raise FileNotFoundError(f"dataset not found: {config.data_path}")
    dataframe = pd.read_parquet(config.data_path).copy()

    if config.target_column not in dataframe.columns:
        for alias in ("target", "label", "off_topic", "out_of_context"):
            if alias in dataframe.columns:
                dataframe[config.target_column] = pd.to_numeric(
                    dataframe[alias],
                    errors="coerce",
                ).fillna(0).astype(int)
                break

    if "split" not in dataframe.columns:
        if "system_prompt" in dataframe.columns:
            systems = (
                dataframe["system_prompt"]
                .fillna("")
                .astype(str)
                .drop_duplicates()
                .sample(frac=1.0, random_state=random_state)
                .tolist()
            )
            train_cut = int(len(systems) * 0.70)
            validation_cut = int(len(systems) * 0.85)
            train_systems = set(systems[:train_cut])
            validation_systems = set(systems[train_cut:validation_cut])

            dataframe["split"] = "test"
            dataframe.loc[dataframe["system_prompt"].isin(train_systems), "split"] = (
                "train"
            )
            dataframe.loc[
                dataframe["system_prompt"].isin(validation_systems),
                "split",
            ] = "validation"
        else:
            shuffled = dataframe.sample(frac=1.0, random_state=random_state)
            train_cut = int(len(shuffled) * 0.70)
            validation_cut = int(len(shuffled) * 0.85)
            shuffled = shuffled.reset_index(drop=True)
            shuffled["split"] = "test"
            shuffled.loc[: train_cut - 1, "split"] = "train"
            shuffled.loc[train_cut: validation_cut - 1, "split"] = "validation"
            dataframe = shuffled

    if "user_prompt" not in dataframe.columns:
        raise ValueError(f"dataset {config.data_path} is missing user_prompt column")
    if config.target_column not in dataframe.columns:
        raise ValueError(
            f"dataset {config.data_path} is missing target column: {config.target_column}"
        )

    train_data = dataframe[dataframe["split"].eq("train")].reset_index(drop=True)
    if set(train_data[config.target_column].unique()) != {0, 1}:
        raise ValueError(f"{config.name} train split must contain both classes")

    sampled_frames = []
    for klass in (0, 1):
        class_data = train_data[train_data[config.target_column].eq(klass)]
        sample_size = min(per_class, len(class_data))
        sampled_frames.append(
            class_data.sample(n=sample_size, random_state=random_state, replace=False)
        )

    sampled = pd.concat(sampled_frames, axis=0).sample(
        frac=1.0,
        random_state=random_state,
    )
    prompts = sampled["user_prompt"].astype(str).tolist()
    labels = sampled[config.target_column].to_numpy(dtype=np.int32)
    return prompts, labels


def fine_tune_task(
    config: TaskConfig,
    artifact_path: Path,
    embedding_model,
    safe_examples: list[str],
    out_of_context_examples: list[str],
    system_prompt: str,
    rehearsal_per_class: int,
    fine_tune_estimators: int,
    calibration_holdout: float,
    random_state: int,
) -> dict:
    artifact = load_artifact(str(artifact_path.resolve()), config.name)

    base_calibrated = artifact["classifier"]
    base_pipeline = base_calibrated.estimator.estimator
    base_xgb = base_pipeline.named_steps["classifier"]

    X_safe = build_features(config, embedding_model, safe_examples, system_prompt)
    y_safe = np.zeros(len(safe_examples), dtype=np.int32)

    X_out_of_context = np.empty((0, X_safe.shape[1]), dtype=np.float32)
    y_out_of_context = np.empty(0, dtype=np.int32)
    if config.name == "out_of_context":
        X_out_of_context = build_features(
            config,
            embedding_model,
            out_of_context_examples,
            system_prompt,
        )
        y_out_of_context = np.ones(len(out_of_context_examples), dtype=np.int32)

    rehearsal_prompts, y_rehearsal = sample_rehearsal_prompts(
        config,
        per_class=rehearsal_per_class,
        random_state=random_state,
    )
    X_rehearsal = build_features(
        config,
        embedding_model,
        rehearsal_prompts,
        system_prompt,
    )

    X_all = np.vstack([X_safe, X_out_of_context, X_rehearsal]).astype(np.float32)
    y_all = np.concatenate([y_safe, y_out_of_context, y_rehearsal]).astype(np.int32)

    X_train, X_calibration, y_train, y_calibration = train_test_split(
        X_all,
        y_all,
        test_size=calibration_holdout,
        random_state=random_state,
        stratify=y_all,
    )

    xgb_params = base_xgb.get_params(deep=True)
    xgb_params["n_estimators"] = fine_tune_estimators
    fine_tuned_xgb = XGBClassifier(**xgb_params)

    fine_tuned_pipeline = Pipeline(
        steps=[
            ("preprocessing", "passthrough"),
            ("classifier", fine_tuned_xgb),
        ]
    )
    fine_tuned_pipeline.fit(
        X_train,
        y_train,
        classifier__xgb_model=base_xgb.get_booster(),
    )

    fine_tuned_calibrated = CalibratedClassifierCV(
        FrozenEstimator(fine_tuned_pipeline),
        method="sigmoid",
    ).fit(X_calibration, y_calibration)

    base_safe_probabilities = base_calibrated.predict_proba(X_safe)[:, 1]
    tuned_safe_probabilities = fine_tuned_calibrated.predict_proba(X_safe)[:, 1]

    updated_artifact = dict(artifact)
    updated_artifact["classifier"] = fine_tuned_calibrated
    updated_artifact["artifact_version"] = max(int(artifact.get("artifact_version", 0)), 7)
    updated_artifact["fine_tuning"] = {
        "type": "system_safes",
        "safe_examples": int(len(safe_examples)),
        "rehearsal_per_class": int(rehearsal_per_class),
        "fine_tune_estimators": int(fine_tune_estimators),
        "calibration_holdout": float(calibration_holdout),
        "random_state": int(random_state),
        "safe_probability_mean_before": float(np.mean(base_safe_probabilities)),
        "safe_probability_mean_after": float(np.mean(tuned_safe_probabilities)),
        "safe_probability_p95_before": float(np.quantile(base_safe_probabilities, 0.95)),
        "safe_probability_p95_after": float(np.quantile(tuned_safe_probabilities, 0.95)),
    }
    return updated_artifact


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune existing detector model artifacts with system-specific "
            "safe examples from Excel files."
        )
    )
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument(
        "--safe-excel",
        type=Path,
        action="append",
        required=True,
        help="Excel file containing safe examples; can be repeated",
    )
    parser.add_argument(
        "--safe-column",
        default="safe_examples",
        help="Column name containing safe examples (default: safe_examples)",
    )
    parser.add_argument(
        "--out-of-context-excel",
        type=Path,
        action="append",
        help="Excel file containing out-of-context examples; can be repeated",
    )
    parser.add_argument(
        "--out-of-context-column",
        default="UserMessage",
        help="Column containing out-of-context examples (default: UserMessage)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", *TASKS.keys()],
        default=["all"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where fine-tuned artifacts will be saved",
    )
    parser.add_argument(
        "--rehearsal-per-class",
        type=int,
        default=800,
        help="How many original train samples to replay per class and task",
    )
    parser.add_argument(
        "--fine-tune-estimators",
        type=int,
        default=80,
        help="Additional trees to train on top of existing booster",
    )
    parser.add_argument(
        "--calibration-holdout",
        type=float,
        default=0.20,
        help="Fraction of fine-tuning data used for probability calibration",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    args = parser.parse_args()

    if not args.system_name.strip():
        parser.error("--system-name must be non-empty")
    if not args.system_prompt.strip():
        parser.error("--system-prompt must be non-empty")
    if not args.safe_column.strip():
        parser.error("--safe-column must be non-empty")
    if not args.out_of_context_column.strip():
        parser.error("--out-of-context-column must be non-empty")
    if args.rehearsal_per_class <= 0:
        parser.error("--rehearsal-per-class must be positive")
    if args.fine_tune_estimators <= 0:
        parser.error("--fine-tune-estimators must be positive")
    if not 0.05 <= args.calibration_holdout <= 0.50:
        parser.error("--calibration-holdout must be between 0.05 and 0.50")
    return args


def main():
    args = parse_args()
    print(json.dumps(run_fine_tuning(
        system_name=args.system_name,
        system_prompt=args.system_prompt,
        safe_excel_paths=args.safe_excel,
        safe_column=args.safe_column,
        out_of_context_excel_paths=args.out_of_context_excel,
        out_of_context_column=args.out_of_context_column,
        output_root=args.output_root,
        tasks=args.tasks,
        rehearsal_per_class=args.rehearsal_per_class,
        fine_tune_estimators=args.fine_tune_estimators,
        calibration_holdout=args.calibration_holdout,
        seed=args.seed,
        artifact_paths={
            "prompt_injection": args.prompt_injection_artifact,
            "out_of_context": args.out_of_context_artifact,
            "toxicity": args.toxicity_artifact,
        },
    ), indent=2))


if __name__ == "__main__":
    main()