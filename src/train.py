import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dataclasses import dataclass, replace
from multiprocessing import get_context
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import (
    DEFAULT_EMBEDDING_MODEL,
    OUT_OF_CONTEXT_FEATURE_DIMENSION,
    PATTERN_FEATURE_NAMES,
    PROMPT_INJECTION_FEATURE_DIMENSION,
    RELATIONSHIP_METRICS_DIMENSION,
    RELATIONAL_FEATURE_DIMENSION,
    USER_PROMPT_FEATURE_DIMENSION,
    load_embedding_model,
    preprocess_prompt_pairs,
    preprocess_user_prompts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifact"
SEED = 42


@dataclass(frozen=True)
class TaskConfig:
    name: str
    data_path: Path
    target_column: str
    artifact_path: Path
    include_pattern_features: bool
    relationship_metrics_only: bool
    out_of_context_features: bool
    user_prompt_only: bool
    calibration_group_column: str
    default_target_recall: float
    threshold_strategy: str
    xgb_parameters: dict[str, int | float]

    @property
    def feature_dimension(self) -> int:
        if self.user_prompt_only:
            return USER_PROMPT_FEATURE_DIMENSION
        if self.out_of_context_features:
            return OUT_OF_CONTEXT_FEATURE_DIMENSION
        if self.include_pattern_features:
            return PROMPT_INJECTION_FEATURE_DIMENSION
        if self.relationship_metrics_only:
            return RELATIONSHIP_METRICS_DIMENSION
        return RELATIONAL_FEATURE_DIMENSION


TASKS = {
    "prompt_injection": TaskConfig(
        name="prompt_injection",
        data_path=DATA_DIR / "prompt_injection.parquet",
        target_column="prompt_injection",
        artifact_path=ARTIFACT_DIR / "prompt_injection_model.pkl",
        include_pattern_features=True,
        relationship_metrics_only=False,
        out_of_context_features=False,
        user_prompt_only=True,
        calibration_group_column="user_prompt",
        default_target_recall=0.995,
        threshold_strategy="recall_floor",
        xgb_parameters={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
    ),
    "toxicity": TaskConfig(
        name="toxicity",
        data_path=DATA_DIR / "toxicity.parquet",
        target_column="toxicity",
        artifact_path=ARTIFACT_DIR / "toxicity_model.pkl",
        include_pattern_features=True,
        relationship_metrics_only=False,
        out_of_context_features=False,
        user_prompt_only=True,
        calibration_group_column="user_prompt",
        default_target_recall=0.995,
        threshold_strategy="recall_floor",
        xgb_parameters={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
    ),
    "out_of_context": TaskConfig(
        name="out_of_context",
        data_path=DATA_DIR / "out-of-context.parquet",
        target_column="out_of_context",
        artifact_path=ARTIFACT_DIR / "out_of_context_model.pkl",
        include_pattern_features=False,
        relationship_metrics_only=False,
        out_of_context_features=True,
        user_prompt_only=False,
        calibration_group_column="pair_id",
        default_target_recall=0.80,
        threshold_strategy="precision_recall_balance",
        xgb_parameters={
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
        },
    ),
}


def load_task_data(config: TaskConfig) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {config.data_path}; run src/build_datasets.py first"
        )
    dataframe = pd.read_parquet(config.data_path).copy()

    if config.target_column not in dataframe.columns:
        target_aliases = ["target", "label", "off_topic", "out_of_context"]
        for alias in target_aliases:
            if alias in dataframe.columns:
                dataframe[config.target_column] = pd.to_numeric(
                    dataframe[alias],
                    errors="coerce",
                ).fillna(0).astype(int)
                break

    if config.name == "out_of_context":
        if "split" not in dataframe.columns:
            if "system_prompt" not in dataframe.columns:
                raise ValueError(
                    "out_of_context dataset needs system_prompt to auto-generate split"
                )
            systems = (
                dataframe["system_prompt"]
                .fillna("")
                .astype(str)
                .drop_duplicates()
                .sample(frac=1.0, random_state=SEED)
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

        if "pair_id" not in dataframe.columns:
            dataframe["pair_id"] = np.arange(len(dataframe), dtype=np.int64)

    required_columns = {
        "user_prompt",
        config.target_column,
        "split",
        config.calibration_group_column,
    }
    if not config.user_prompt_only:
        required_columns.add("system_prompt")
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{config.name} dataset is missing columns: {missing}")
    if set(dataframe[config.target_column].unique()) != {0, 1}:
        raise ValueError(f"{config.target_column} must contain both binary classes")
    return dataframe


def split_task_data(dataframe: pd.DataFrame, config: TaskConfig):
    train_data = dataframe[dataframe["split"].eq("train")].reset_index(drop=True)
    validation_data = dataframe[dataframe["split"].eq("validation")].reset_index(
        drop=True
    )
    test_data = dataframe[dataframe["split"].eq("test")].reset_index(drop=True)

    calibration_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=SEED,
    )
    calibration_indices, threshold_indices = next(
        calibration_split.split(
            validation_data,
            y=validation_data[config.target_column],
            groups=validation_data[config.calibration_group_column],
        )
    )
    calibration_data = validation_data.iloc[calibration_indices].reset_index(drop=True)
    threshold_data = validation_data.iloc[threshold_indices].reset_index(drop=True)
    return train_data, calibration_data, threshold_data, test_data


def select_threshold(
    y_true,
    probabilities,
    target_recall,
    strategy: str,
):
    candidates = []
    for threshold in np.linspace(0.001, 0.999, 999):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
        recall = recall_score(y_true, predictions, zero_division=0)
        precision = precision_score(y_true, predictions, zero_division=0)
        false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
        f1 = f1_score(y_true, predictions, zero_division=0)
        candidates.append(
            (threshold, recall, precision, f1, false_positive_rate, fn)
        )

    if strategy == "precision_recall_balance":
        operating_points = [
            candidate
            for candidate in candidates
            if candidate[1] > 0.0 and candidate[2] > 0.0
        ]
        selected_pool = operating_points if operating_points else candidates
        selected = min(
            selected_pool,
            key=lambda item: (
                abs(item[2] - item[1]),
                -item[3],
                -item[0],
            ),
        )
        return selected[0], selected[1], selected[4], selected[5]

    eligible = [candidate for candidate in candidates if candidate[1] >= target_recall]
    if eligible:
        selected = min(eligible, key=lambda item: (item[4], -item[0]))
        return selected[0], selected[1], selected[4], selected[5]
    selected = max(candidates, key=lambda item: (item[1], -item[4]))
    return selected[0], selected[1], selected[4], selected[5]


def evaluate(y_true, probabilities, threshold) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def train_task(
    config: TaskConfig,
    embedding_model,
    batch_size: int,
    target_recall: float | None = None,
    xgb_jobs: int = -1,
) -> dict:
    dataframe = load_task_data(config)
    train_data, calibration_data, threshold_data, test_data = split_task_data(
        dataframe,
        config,
    )

    def transform(frame: pd.DataFrame) -> np.ndarray:
        if config.user_prompt_only:
            return preprocess_user_prompts(
                frame["user_prompt"].tolist(),
                embedding_model,
                batch_size=batch_size,
                show_progress_bar=True,
            )
        return preprocess_prompt_pairs(
            frame["system_prompt"].tolist(),
            frame["user_prompt"].tolist(),
            embedding_model,
            batch_size=batch_size,
            show_progress_bar=True,
            include_pattern_features=config.include_pattern_features,
            relationship_metrics_only=config.relationship_metrics_only,
            out_of_context_features=config.out_of_context_features,
        )

    print(f"[{config.name}] Generating training features...")
    X_train = transform(train_data)
    print(f"[{config.name}] Generating calibration features...")
    X_calibration = transform(calibration_data)
    print(f"[{config.name}] Generating threshold-selection features...")
    X_threshold = transform(threshold_data)
    print(f"[{config.name}] Generating test features...")
    X_test = transform(test_data)

    y_train = train_data[config.target_column].to_numpy()
    y_calibration = calibration_data[config.target_column].to_numpy()
    y_threshold = threshold_data[config.target_column].to_numpy()
    y_test = test_data[config.target_column].to_numpy()
    negative_count, positive_count = np.bincount(y_train)

    classifier = XGBClassifier(
        n_estimators=int(config.xgb_parameters["n_estimators"]),
        max_depth=int(config.xgb_parameters["max_depth"]),
        learning_rate=float(config.xgb_parameters["learning_rate"]),
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative_count / positive_count,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=xgb_jobs,
        random_state=SEED,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", "passthrough"),
            ("classifier", classifier),
        ]
    )
    print(f"[{config.name}] Training XGBoost...")
    pipeline.fit(X_train, y_train)

    print(f"[{config.name}] Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(pipeline),
        method="sigmoid",
    ).fit(X_calibration, y_calibration)

    requested_recall = target_recall or config.default_target_recall
    threshold_probabilities = calibrated_model.predict_proba(X_threshold)[:, 1]
    threshold, recall, false_positive_rate, false_negatives = select_threshold(
        y_threshold,
        threshold_probabilities,
        requested_recall,
        config.threshold_strategy,
    )
    test_probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, threshold)

    artifact = {
        "artifact_version": 6,
        "task": config.name,
        "classifier": calibrated_model,
        "calibration_method": "sigmoid",
        "embedding_model_name": DEFAULT_EMBEDDING_MODEL,
        "feature_dimension": config.feature_dimension,
        "include_pattern_features": config.include_pattern_features,
        "relationship_metrics_only": config.relationship_metrics_only,
        "out_of_context_features": config.out_of_context_features,
        "user_prompt_only": config.user_prompt_only,
        "pattern_feature_names": (
            PATTERN_FEATURE_NAMES if config.include_pattern_features else ()
        ),
        "recommended_threshold": float(threshold),
        "threshold_metrics": {
            "strategy": config.threshold_strategy,
            "target_recall": float(requested_recall),
            "validation_recall": float(recall),
            "validation_false_positive_rate": float(false_positive_rate),
            "validation_false_negatives": int(false_negatives),
        },
        "test_metrics": test_metrics,
    }
    config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.artifact_path)

    summary = {
        "task": config.name,
        "artifact_path": str(config.artifact_path.resolve()),
        "feature_dimension": config.feature_dimension,
        "recommended_threshold": float(threshold),
        **artifact["threshold_metrics"],
        "test_metrics": test_metrics,
    }
    print(json.dumps(summary, indent=2))
    return artifact


def train_task_in_worker(
    config: TaskConfig,
    batch_size: int,
    target_recall: float | None,
    xgb_jobs: int,
) -> str:
    embedding_model = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
    train_task(
        config,
        embedding_model,
        batch_size,
        target_recall,
        xgb_jobs,
    )
    return config.name


def parse_args():
    parser = argparse.ArgumentParser(description="Train one or all detection models.")
    parser.add_argument(
        "--task",
        choices=["all", *TASKS],
        default="all",
    )
    parser.add_argument("--target-recall", type=float)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="optional custom parquet path for the selected single task",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(len(TASKS), os.cpu_count() or 1),
        help="maximum concurrent tasks when --task all is used",
    )
    parser.add_argument(
        "--xgb-jobs",
        type=int,
        help="threads per XGBoost task; defaults to an even CPU split",
    )
    args = parser.parse_args()
    if args.target_recall is not None and not 0.0 < args.target_recall <= 1.0:
        parser.error("--target-recall must be in the interval (0, 1]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.xgb_jobs is not None and args.xgb_jobs <= 0:
        parser.error("--xgb-jobs must be positive")
    if args.data_path is not None and args.task == "all":
        parser.error("--data-path can only be used with a single --task")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    selected_tasks = list(
        TASKS.values() if arguments.task == "all" else [TASKS[arguments.task]]
    )
    if arguments.data_path is not None:
        selected_tasks = [replace(selected_tasks[0], data_path=arguments.data_path)]
    worker_count = min(arguments.workers, len(selected_tasks))
    xgb_jobs = arguments.xgb_jobs
    if xgb_jobs is None:
        xgb_jobs = (
            -1
            if worker_count == 1
            else max(1, (os.cpu_count() or 1) // worker_count)
        )

    if worker_count == 1:
        shared_embedding_model = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
        for task_config in selected_tasks:
            train_task(
                task_config,
                shared_embedding_model,
                arguments.batch_size,
                arguments.target_recall,
                xgb_jobs,
            )
    else:
        print(
            f"Training {len(selected_tasks)} tasks with {worker_count} workers "
            f"and {xgb_jobs} XGBoost threads per worker."
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(
                    train_task_in_worker,
                    task_config,
                    arguments.batch_size,
                    arguments.target_recall,
                    xgb_jobs,
                ): task_config.name
                for task_config in selected_tasks
            }
            for future in as_completed(futures):
                task_name = future.result()
                print(f"[{task_name}] Training completed.")
