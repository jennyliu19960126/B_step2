"""Input provenance, population sizing and auditable final candidate selection."""
import ast
import math
from numbers import Integral


def population_target(feature_count, multiplier):
    if isinstance(multiplier, bool) or not isinstance(multiplier, Integral) or multiplier <= 0:
        raise ValueError("population_multiplier must be a positive integer")
    if feature_count <= 0:
        raise ValueError("GP input must contain features")
    return int(feature_count * multiplier)


def feature_origins(metadata):
    names = metadata["feature_names"]
    origins = metadata.get("feature_origins")
    if origins is None:
        mapping = metadata.get("source_mapping", [])
        if len(mapping) != len(names):
            raise ValueError("Input metadata requires aligned feature_origins or source_mapping")
        origins = []
        for row in mapping:
            origin = row.get("origin")
            if origin is None:
                group = row.get("source_group", "")
                if group in ("base", "gate1"):
                    origin = "base"
                elif group == "step1" or group.startswith(("x20_", "x40_", "x60_")):
                    origin = "step1"
            origins.append(origin)
    if len(origins) != len(names) or any(x not in ("base", "step1") for x in origins):
        raise ValueError("Every GP input needs explicit base/step1 provenance")
    return list(origins)


def dependency_kind(stack, origins):
    if isinstance(stack, str):
        stack = ast.literal_eval(stack)
    if not isinstance(stack, (list, tuple)):
        raise ValueError("Expected a prefix formulation stack")
    indices = [int(x) for x in stack if isinstance(x, Integral) and not isinstance(x, bool)]
    if any(i < 0 or i >= len(origins) for i in indices):
        raise ValueError("Expression feature index outside input metadata")
    if not indices:
        return "constant_only"
    return "contains_step1" if any(origins[i] == "step1" for i in indices) else "base_only"


def passes_raw_ic(program, threshold):
    value = getattr(program, "base_fitness_", None)
    return value is not None and math.isfinite(value) and (threshold is None or value >= threshold)


def select_final_candidates(frame, origins, metric, threshold, max_generation=None):
    result = frame.copy()
    if max_generation is not None:
        result = result[result["generation"] <= max_generation].copy()
    result = result.drop_duplicates("formulation", keep="first").copy()
    result["dependency_kind"] = result["formulation_stack"].map(lambda s: dependency_kind(s, origins))
    eligible = result[metric].abs() > threshold
    selected = result[eligible & result["dependency_kind"].eq("contains_step1")].copy()
    counts = {"unique_candidates": len(result),
              "base_only": int(result["dependency_kind"].eq("base_only").sum()),
              "constant_only": int(result["dependency_kind"].eq("constant_only").sum()),
              "final_count": len(selected)}
    return selected, counts
