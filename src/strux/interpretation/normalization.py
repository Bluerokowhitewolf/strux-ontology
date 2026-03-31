import ast
import math
import re
import unicodedata
from difflib import SequenceMatcher


def normalize_label(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_semantic_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_semantic_text(value: object) -> str:
    return normalize_semantic_text(value).replace(" ", "")


def slugify(value: object) -> str:
    normalized = normalize_label(value)
    return normalized.replace(" ", "-") or "unknown"


def coerce_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float):
        return default if math.isnan(value) else value
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def normalize_score(value: object, scale_max: float) -> float:
    return clamp(coerce_float(value) / scale_max)


def split_list_like(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                items = parsed
            else:
                items = re.split(r"[,\n;]+", text)
        else:
            items = re.split(r"[,\n;]+", text)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item).split()).strip()
        if not text:
            continue
        normalized = normalize_label(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(text)
    return cleaned


def soft_match_score(left: str, right: str) -> float:
    left_norm = normalize_label(left)
    right_norm = normalize_label(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    return clamp(max(overlap, ratio))


def weighted_overlap(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return 0.0 if denominator == 0 else numerator / denominator
