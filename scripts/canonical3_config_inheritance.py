from pathlib import Path
from copy import deepcopy
import yaml


def deep_merge(parent, child):
    """Return parent with child overrides, recursively for mappings."""
    if isinstance(parent, dict) and isinstance(child, dict):
        out = deepcopy(parent)
        for key, value in child.items():
            out[key] = deep_merge(out[key], value) if key in out else deepcopy(value)
        return out
    return deepcopy(child)


def canonical3_template(root: Path, label: str):
    """Resolve v2 templates without dropping parent generation fields."""
    v2 = root / "configs/group12_15_v2/smoke"
    if label == "12":
        return yaml.safe_load((v2 / "group12_v2_case01.yaml").read_text())
    if label == "13":
        return yaml.safe_load((v2 / "group13_v2_case01.yaml").read_text())
    if label.startswith("14."):
        parent = yaml.safe_load((v2 / "group12_v2_case01.yaml").read_text())
        child = yaml.safe_load((v2 / "group14_v2_case01.yaml").read_text())
        return deep_merge(parent, child)
    if label.startswith("15."):
        parent = yaml.safe_load((v2 / "group13_v2_case01.yaml").read_text())
        child = yaml.safe_load((v2 / "group15_v2_case01.yaml").read_text())
        return deep_merge(parent, child)
    return None
