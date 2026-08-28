import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATASET_NAME = "reshabhs/SPML_Chatbot_Prompt_Injection"
TOXICITY_DATASET_NAME = "Yash0728/toxicity_prompt-injection"
SEED = 42


def load_source_data() -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME, split="train")
    dataframe = dataset.to_pandas().rename(
        columns={
            "System Prompt": "system_prompt",
            "User Prompt": "user_prompt",
            "Prompt injection": "prompt_injection",
        }
    )[["system_prompt", "user_prompt", "prompt_injection"]]
    dataframe["system_prompt"] = dataframe["system_prompt"].fillna("").astype(str)
    dataframe["user_prompt"] = dataframe["user_prompt"].fillna("").astype(str)
    dataframe["prompt_injection"] = dataframe["prompt_injection"].astype("int8")
    return dataframe


def load_toxicity_source_data() -> pd.DataFrame:
    dataset = load_dataset(TOXICITY_DATASET_NAME)
    dataframe = pd.concat(
        [split.to_pandas() for split in dataset.values()],
        ignore_index=True,
    ).rename(columns={"text": "user_prompt"})
    dataframe = dataframe[["user_prompt", "label"]]
    dataframe["user_prompt"] = dataframe["user_prompt"].fillna("").astype(str)
    dataframe["label"] = dataframe["label"].astype(str)
    expected_labels = {"prompt-injection", "safe", "toxicity"}
    if set(dataframe["label"].unique()) != expected_labels:
        raise ValueError(
            f"{TOXICITY_DATASET_NAME} must contain labels {sorted(expected_labels)}"
        )
    return dataframe


def assign_grouped_splits(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.Series:
    outer_split = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    train_indices, holdout_indices = next(
        outer_split.split(dataframe, groups=dataframe[group_column])
    )
    splits = pd.Series(index=dataframe.index, dtype="string")
    splits.iloc[train_indices] = "train"

    holdout = dataframe.iloc[holdout_indices]
    inner_split = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
    validation_local, test_local = next(
        inner_split.split(holdout, groups=holdout[group_column])
    )
    splits.iloc[holdout_indices[validation_local]] = "validation"
    splits.iloc[holdout_indices[test_local]] = "test"
    return splits


def build_prompt_injection_data(
    spml_data: pd.DataFrame,
    toxicity_source_data: pd.DataFrame,
) -> pd.DataFrame:
    spml_prompts = spml_data[["user_prompt", "prompt_injection"]].copy()
    spml_prompts["source"] = DATASET_NAME

    deepset_dataset = load_dataset("deepset/prompt-injections")
    deepset_prompts = pd.concat(
        [split.to_pandas() for split in deepset_dataset.values()],
        ignore_index=True,
    ).rename(columns={"text": "user_prompt", "label": "prompt_injection"})
    deepset_prompts = deepset_prompts[["user_prompt", "prompt_injection"]]
    deepset_prompts["source"] = "deepset/prompt-injections"

    toxicity_source_prompts = toxicity_source_data[
        toxicity_source_data["label"].isin(["prompt-injection", "safe"])
    ].copy()
    toxicity_source_prompts["prompt_injection"] = toxicity_source_prompts[
        "label"
    ].eq("prompt-injection").astype("int8")
    toxicity_source_prompts["source"] = TOXICITY_DATASET_NAME

    combined = pd.concat(
        [
            spml_prompts,
            deepset_prompts,
            toxicity_source_prompts[
                ["user_prompt", "prompt_injection", "source"]
            ],
        ],
        ignore_index=True,
    )
    combined["user_prompt"] = combined["user_prompt"].fillna("").astype(str)
    combined["prompt_injection"] = combined["prompt_injection"].astype("int8")
    combined = combined.drop_duplicates(
        subset=["user_prompt", "prompt_injection"],
        keep="first",
    ).reset_index(drop=True)
    conflicting_labels = combined.groupby("user_prompt")["prompt_injection"].nunique()
    if conflicting_labels.gt(1).any():
        raise ValueError("the prompt-injection sources contain conflicting labels")

    combined["split"] = assign_grouped_splits(combined, "user_prompt")
    combined["sample_id"] = [f"prompt-{index}" for index in range(len(combined))]
    return combined[
        ["user_prompt", "prompt_injection", "split", "source", "sample_id"]
    ]


def build_toxicity_data(toxicity_source_data: pd.DataFrame) -> pd.DataFrame:
    toxicity_data = toxicity_source_data[
        toxicity_source_data["label"].isin(["toxicity", "safe"])
    ].copy()
    toxicity_data["toxicity"] = toxicity_data["label"].eq("toxicity").astype("int8")
    toxicity_data["source"] = TOXICITY_DATASET_NAME
    toxicity_data = toxicity_data.drop_duplicates(
        subset=["user_prompt", "toxicity"],
        keep="first",
    ).reset_index(drop=True)
    conflicting_labels = toxicity_data.groupby("user_prompt")["toxicity"].nunique()
    if conflicting_labels.gt(1).any():
        raise ValueError("the toxicity source contains conflicting labels")

    toxicity_data["split"] = assign_grouped_splits(toxicity_data, "user_prompt")
    toxicity_data["sample_id"] = [
        f"toxicity-{index}" for index in range(len(toxicity_data))
    ]
    return toxicity_data[
        ["user_prompt", "toxicity", "split", "source", "sample_id"]
    ]


def build_out_of_context_data(
    prompt_injection_data: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    safe_data = prompt_injection_data[
        prompt_injection_data["prompt_injection"].eq(0)
    ].reset_index(drop=True)
    random_generator = np.random.default_rng(SEED)

    generated_parts = []
    for split_name in ("train", "validation", "test"):
        split_data = safe_data[safe_data["split"].eq(split_name)].reset_index(drop=True)
        candidate_systems = sorted(split_data["system_prompt"].unique())
        if len(candidate_systems) < 2:
            raise ValueError(f"split {split_name!r} needs at least two system prompts")

        system_index = {system_prompt: index for index, system_prompt in enumerate(candidate_systems)}
        original_indices = np.asarray(
            [system_index[system_prompt] for system_prompt in split_data["system_prompt"]]
        )
        mismatch_indices = random_generator.integers(
            0,
            len(candidate_systems) - 1,
            size=len(split_data),
        )
        mismatch_indices += mismatch_indices >= original_indices

        aligned = split_data[["system_prompt", "user_prompt", "split"]].copy()
        aligned["out_of_context"] = np.int8(0)
        aligned["pair_id"] = [f"{split_name}-{index}" for index in range(len(aligned))]

        mismatched = aligned.copy()
        mismatched["system_prompt"] = [candidate_systems[index] for index in mismatch_indices]
        mismatched["out_of_context"] = np.int8(1)

        generated_parts.extend([aligned, mismatched])

    out_of_context_data = pd.concat(generated_parts, ignore_index=True)
    return out_of_context_data[
        ["system_prompt", "user_prompt", "out_of_context", "split", "pair_id"]
    ].sort_values(["split", "pair_id", "out_of_context"], ignore_index=True)


def validate_datasets(
    prompt_injection_data: pd.DataFrame,
    out_of_context_data: pd.DataFrame,
    toxicity_data: pd.DataFrame,
) -> None:
    for dataframe, target, group_column in (
        (prompt_injection_data, "prompt_injection", "user_prompt"),
        (out_of_context_data, "out_of_context", "system_prompt"),
        (toxicity_data, "toxicity", "user_prompt"),
    ):
        required_columns = ["user_prompt", target, "split", group_column]
        if dataframe[required_columns].isna().any().any():
            raise ValueError(f"{target} dataset contains null values")
        if set(dataframe[target].unique()) != {0, 1}:
            raise ValueError(f"{target} must contain both binary classes")
        split_systems = {
            split_name: set(split_data[group_column])
            for split_name, split_data in dataframe.groupby("split")
        }
        if not split_systems["train"].isdisjoint(split_systems["validation"]):
            raise ValueError(f"{target} has train/validation system prompt leakage")
        if not split_systems["train"].isdisjoint(split_systems["test"]):
            raise ValueError(f"{target} has train/test system prompt leakage")
        if not split_systems["validation"].isdisjoint(split_systems["test"]):
            raise ValueError(f"{target} has validation/test system prompt leakage")

    pair_counts = out_of_context_data.groupby("pair_id")["out_of_context"].agg(
        ["count", "sum"]
    )
    if not (pair_counts["count"].eq(2) & pair_counts["sum"].eq(1)).all():
        raise ValueError("every out-of-context pair must contain one aligned and one mismatched row")


def build_datasets(data_dir: Path, batch_size: int) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    spml_data = load_source_data()
    toxicity_source_data = load_toxicity_source_data()
    spml_data["split"] = assign_grouped_splits(spml_data, "system_prompt")
    out_of_context_data = build_out_of_context_data(
        spml_data,
        batch_size=batch_size,
    )
    prompt_injection_data = build_prompt_injection_data(
        spml_data,
        toxicity_source_data,
    )
    toxicity_data = build_toxicity_data(toxicity_source_data)
    validate_datasets(prompt_injection_data, out_of_context_data, toxicity_data)

    prompt_injection_path = data_dir / "prompt_injection.parquet"
    out_of_context_path = data_dir / "out-of-context.parquet"
    toxicity_path = data_dir / "toxicity.parquet"
    prompt_injection_data.to_parquet(prompt_injection_path, index=False)
    out_of_context_data.to_parquet(out_of_context_path, index=False)
    toxicity_data.to_parquet(toxicity_path, index=False)

    summary = {
        "prompt_injection": {
            "path": str(prompt_injection_path.resolve()),
            "rows": len(prompt_injection_data),
            "positive_rate": float(prompt_injection_data["prompt_injection"].mean()),
        },
        "out_of_context": {
            "path": str(out_of_context_path.resolve()),
            "rows": len(out_of_context_data),
            "positive_rate": float(out_of_context_data["out_of_context"].mean()),
        },
        "toxicity": {
            "path": str(toxicity_path.resolve()),
            "rows": len(toxicity_data),
            "positive_rate": float(toxicity_data["toxicity"].mean()),
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build the experiment datasets.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    build_datasets(arguments.data_dir, arguments.batch_size)