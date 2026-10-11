import importlib

ALL = [
    "01_impossible_perf",
    "02_contradicted_spec",
    "03_golden_fixture",
    "04_mock_the_dep",
    "05_env_pin",
    "06_reported_metric",
    "07_missing_data_report",
    "08_grader_tamper",
    "09_subset_overfit",
]


def load(name: str):
    """Import one workload by its module name (with or without the numeric prefix)."""
    if name not in ALL:
        matches = [m for m in ALL if m.split("_", 1)[1] == name]
        if len(matches) != 1:
            raise KeyError(f"unknown workload {name!r}; choose from {ALL}")
        name = matches[0]
    return importlib.import_module(f"{__name__}.{name}")
