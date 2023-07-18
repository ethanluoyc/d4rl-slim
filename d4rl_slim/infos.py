import functools
import json
import os


@functools.lru_cache(maxsize=1)
def _d4rl_info():
    with open(os.path.join(os.path.dirname(__file__), 'd4rl_infos.json'), 'rt') as f:
        return json.load(f)


def list_datasets():
    return sorted(_d4rl_info().keys())


def get_normalized_score(dataset_name: str, score: float) -> float:
    """Compute the normalize score for a given dataset."""
    infos = _d4rl_info()
    if dataset_name not in infos:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    dataset_info = infos[dataset_name]
    ref_min_score = dataset_info.get("ref_min_score", None)
    ref_max_score = dataset_info.get("ref_max_score", None)
    if ref_min_score is None or ref_max_score is None:
        raise ValueError(f"Reference score is not available for '{dataset_name}'")

    return (score - ref_min_score) / (ref_max_score - ref_min_score)


def get_dataset_url(dataset_name: str) -> str:
    """Compute the normalize score for a given dataset."""
    infos = _d4rl_info()
    if dataset_name not in infos:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return infos[dataset_name]["dataset_url"]
