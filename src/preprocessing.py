from collections.abc import Sequence
import re

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIMENSION = 384
PATTERN_DEFINITIONS = (
    (
        "has_exfiltration_verb",
        r"\b(?:reveal|show|display|print|repeat|expose|leak|output|provide|tell)\b",
    ),
    (
        "has_instruction_term",
        r"\b(?:instruction|directive|rule|policy|guideline|constraint|restriction)s?\b",
    ),
    ("has_you_are", r"\byou\s+are\b"),
    (
        "has_constraint_language",
        r"\b(?:must|never|do\s+not|must\s+not|should\s+not|refrain\s+from|under\s+no\s+circumstances)\b",
    ),
    ("has_hash_delimiter", r"#{3,}"),
    (
        "has_override_verb",
        r"\b(?:ignore|disregard|forget|override|bypass|discard|supersede)\b",
    ),
    ("has_ignore", r"\bignore\b"),
    (
        "has_override_instruction",
        r"\b(?:ignore|disregard|forget|override|bypass|discard|supersede)\b.{0,100}"
        r"\b(?:instruction|directive|rule|policy|guideline|constraint|restriction|prompt)s?\b",
    ),
    (
        "has_role_switch",
        r"\b(?:act\s+as|pretend|simulate|roleplay|you\s+are\s+now|new\s+role|from\s+now\s+on)\b",
    ),
)
PATTERN_FEATURE_NAMES = tuple(name for name, _ in PATTERN_DEFINITIONS)
COMPILED_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    for _, pattern in PATTERN_DEFINITIONS
)
RELATIONAL_FEATURE_DIMENSION = EMBEDDING_DIMENSION * 4 + 4
RELATIONSHIP_METRICS_DIMENSION = 4
OUT_OF_CONTEXT_FEATURE_DIMENSION = EMBEDDING_DIMENSION * 2 + 3
USER_PROMPT_FEATURE_DIMENSION = EMBEDDING_DIMENSION + len(PATTERN_DEFINITIONS)
PROMPT_INJECTION_FEATURE_DIMENSION = (
    RELATIONAL_FEATURE_DIMENSION + len(PATTERN_DEFINITIONS)
)
FEATURE_DIMENSION = PROMPT_INJECTION_FEATURE_DIMENSION


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_prompts(
    model: SentenceTransformer,
    prompts: Sequence[str],
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> np.ndarray:
    if not prompts:
        raise ValueError("prompts must contain at least one item")
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("every prompt must be a string")

    unique_prompts = list(dict.fromkeys(prompts))
    unique_embeddings = model.encode(
        unique_prompts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    ).astype(np.float32)
    embedding_by_prompt = dict(zip(unique_prompts, unique_embeddings, strict=True))
    return np.vstack([embedding_by_prompt[prompt] for prompt in prompts])


def jaccard_distance(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", left.lower()))
    right_tokens = set(re.findall(r"\w+", right.lower()))
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return 1.0 - len(left_tokens & right_tokens) / len(union)


def build_features(
    system_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    pattern_features: np.ndarray | None = None,
) -> np.ndarray:
    if system_embeddings.shape != user_embeddings.shape:
        raise ValueError("system and user embeddings must have the same shape")
    if system_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if pattern_features is not None:
        expected_pattern_shape = (system_embeddings.shape[0], len(PATTERN_DEFINITIONS))
        if pattern_features.shape != expected_pattern_shape:
            raise ValueError(
                f"expected pattern feature shape {expected_pattern_shape}, "
                f"got {pattern_features.shape}"
            )

    difference = system_embeddings - user_embeddings
    absolute_difference = np.abs(difference)
    elementwise_product = system_embeddings * user_embeddings
    dot_product = np.sum(elementwise_product, axis=1, keepdims=True)
    system_norm = np.linalg.norm(system_embeddings, axis=1, keepdims=True)
    user_norm = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    cosine_similarity = dot_product / np.maximum(system_norm * user_norm, 1e-12)
    euclidean_distance = np.linalg.norm(difference, axis=1, keepdims=True)
    manhattan_distance = np.sum(absolute_difference, axis=1, keepdims=True)

    feature_blocks = [
        system_embeddings,
        user_embeddings,
        absolute_difference,
        elementwise_product,
        cosine_similarity,
        euclidean_distance,
        manhattan_distance,
        dot_product,
    ]
    if pattern_features is not None:
        feature_blocks.append(pattern_features)
    features = np.hstack(feature_blocks).astype(np.float32)

    expected_dimension = (
        PROMPT_INJECTION_FEATURE_DIMENSION
        if pattern_features is not None
        else RELATIONAL_FEATURE_DIMENSION
    )
    if features.shape[1] != expected_dimension:
        raise ValueError(
            f"expected {expected_dimension} features, got {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    return features


def build_relationship_metrics(
    system_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
) -> np.ndarray:
    if system_embeddings.shape != user_embeddings.shape:
        raise ValueError("system and user embeddings must have the same shape")
    if system_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")

    difference = system_embeddings - user_embeddings
    dot_product = np.sum(system_embeddings * user_embeddings, axis=1, keepdims=True)
    system_norm = np.linalg.norm(system_embeddings, axis=1, keepdims=True)
    user_norm = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    features = np.hstack(
        [
            dot_product / np.maximum(system_norm * user_norm, 1e-12),
            np.linalg.norm(difference, axis=1, keepdims=True),
            np.sum(np.abs(difference), axis=1, keepdims=True),
            dot_product,
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    return features


def build_out_of_context_features(
    system_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    system_prompts: Sequence[str] | None = None,
    user_prompts: Sequence[str] | None = None,
) -> np.ndarray:
    if system_embeddings.shape != user_embeddings.shape:
        raise ValueError("system and user embeddings must have the same shape")
    if system_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")

    difference = system_embeddings - user_embeddings
    absolute_difference = np.abs(difference)
    dot_product = np.sum(system_embeddings * user_embeddings, axis=1, keepdims=True)
    system_norm = np.linalg.norm(system_embeddings, axis=1, keepdims=True)
    user_norm = np.linalg.norm(user_embeddings, axis=1, keepdims=True)

    if (system_prompts is None) != (user_prompts is None):
        raise ValueError(
            "system_prompts and user_prompts must be both provided or both omitted"
        )
    if system_prompts is not None and user_prompts is not None:
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have the same length")
        if len(system_prompts) != system_embeddings.shape[0]:
            raise ValueError("prompt list lengths must match embedding rows")
        jaccard_distances = np.asarray(
            [
                jaccard_distance(system_prompt, user_prompt)
                for system_prompt, user_prompt in zip(
                    system_prompts,
                    user_prompts,
                    strict=True,
                )
            ],
            dtype=np.float32,
        ).reshape(-1, 1)
    else:
        # Fallback for compatibility when raw prompts are not available.
        jaccard_distances = np.zeros((system_embeddings.shape[0], 1), dtype=np.float32)

    features = np.hstack(
        [
            user_embeddings,
            absolute_difference,
            dot_product / np.maximum(system_norm * user_norm, 1e-12),
            jaccard_distances,
            dot_product,
        ]
    ).astype(np.float32)
    if features.shape[1] != OUT_OF_CONTEXT_FEATURE_DIMENSION:
        raise ValueError(
            f"expected {OUT_OF_CONTEXT_FEATURE_DIMENSION} features, "
            f"got {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    return features


def extract_pattern_features(user_prompts: Sequence[str]) -> np.ndarray:
    if any(not isinstance(prompt, str) for prompt in user_prompts):
        raise ValueError("every user prompt must be a string")

    return np.asarray(
        [
            [float(bool(pattern.search(prompt))) for pattern in COMPILED_PATTERNS]
            for prompt in user_prompts
        ],
        dtype=np.float32,
    ).reshape(len(user_prompts), len(PATTERN_DEFINITIONS))


def build_user_prompt_features(
    user_embeddings: np.ndarray,
    pattern_features: np.ndarray,
) -> np.ndarray:
    if user_embeddings.ndim != 2:
        raise ValueError("user embeddings must be two-dimensional")
    expected_pattern_shape = (user_embeddings.shape[0], len(PATTERN_DEFINITIONS))
    if pattern_features.shape != expected_pattern_shape:
        raise ValueError(
            f"expected pattern feature shape {expected_pattern_shape}, "
            f"got {pattern_features.shape}"
        )
    features = np.hstack([user_embeddings, pattern_features]).astype(np.float32)
    if features.shape[1] != USER_PROMPT_FEATURE_DIMENSION:
        raise ValueError(
            f"expected {USER_PROMPT_FEATURE_DIMENSION} features, "
            f"got {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    return features


def preprocess_user_prompts(
    user_prompts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> np.ndarray:
    user_embeddings = encode_prompts(
        model,
        user_prompts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return build_user_prompt_features(
        user_embeddings,
        extract_pattern_features(user_prompts),
    )


def preprocess_prompt_pairs(
    system_prompts: Sequence[str],
    user_prompts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress_bar: bool = False,
    include_pattern_features: bool = True,
    relationship_metrics_only: bool = False,
    out_of_context_features: bool = False,
) -> np.ndarray:
    if len(system_prompts) != len(user_prompts):
        raise ValueError("system_prompts and user_prompts must have the same length")

    system_embeddings = encode_prompts(
        model,
        system_prompts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    user_embeddings = encode_prompts(
        model,
        user_prompts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    if out_of_context_features:
        if include_pattern_features or relationship_metrics_only:
            raise ValueError(
                "out_of_context_features cannot be combined with other feature modes"
            )
        return build_out_of_context_features(
            system_embeddings,
            user_embeddings,
            system_prompts=system_prompts,
            user_prompts=user_prompts,
        )
    if relationship_metrics_only:
        if include_pattern_features:
            raise ValueError(
                "pattern features cannot be combined with relationship_metrics_only"
            )
        return build_relationship_metrics(system_embeddings, user_embeddings)

    pattern_features = (
        extract_pattern_features(user_prompts) if include_pattern_features else None
    )
    return build_features(system_embeddings, user_embeddings, pattern_features)