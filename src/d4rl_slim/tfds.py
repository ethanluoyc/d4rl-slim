# ruff: noqa: E501
import functools

from d4rl_slim.infos import list_datasets


@functools.lru_cache
def _d4rl_tfds_mapping():
    D4RL_TFDS_MAP = {}

    AVAILABLE_D4RL_DATASETS = list_datasets()
    for env in ["halfcheetah", "hopper", "walker2d", "ant"]:
        for dset in [
            "random",
            "medium",
            "expert",
            "medium-replay",
            "full-replay",
            "medium-expert",
        ]:
            for version in ["v0", "v1", "v2"]:
                if version == "v0" and dset == "full-replay":
                    continue
                if version == "v0" and dset == "medium-replay":
                    d4rl_name = f"{env}-mixed-{version}"
                else:
                    d4rl_name = f"{env}-{dset}-{version}"
                tfds_name = f"d4rl_mujoco_{env}/{version}-{dset}"
                assert d4rl_name in AVAILABLE_D4RL_DATASETS, d4rl_name
                D4RL_TFDS_MAP[d4rl_name] = tfds_name

    # Antmaze
    for dset in [
        "large-diverse",
        "large-play",
        "medium-play",
        "medium-diverse",
        "umaze",
        "umaze-diverse",
    ]:
        for version in ["v0", "v2"]:
            # V2 added recently in nightly only
            # https://github.com/tensorflow/datasets/pull/5008
            d4rl_name = f"antmaze-{dset}-{version}"
            tfds_name = f"d4rl_antmaze/{dset}-{version}"
            assert d4rl_name in AVAILABLE_D4RL_DATASETS, d4rl_name
            D4RL_TFDS_MAP[d4rl_name] = tfds_name

    # Adroit
    for env in ["door", "hammer", "pen", "relocate"]:
        for dset in ["human", "cloned", "expert"]:
            for version in ["v0", "v1"]:
                d4rl_name = f"{env}-{dset}-{version}"
                tfds_name = f"d4rl_adroit_{env}/{version}-{dset}"
                assert d4rl_name in AVAILABLE_D4RL_DATASETS, d4rl_name
                D4RL_TFDS_MAP[d4rl_name] = tfds_name

    return D4RL_TFDS_MAP


def get_tfds_name(d4rl_name: str) -> str:
    """Return the corresponding TFDS name for a given D4RL dataset name."""
    mapping = _d4rl_tfds_mapping()
    tfds_name = mapping.get(d4rl_name, None)
    if tfds_name is None:
        raise ValueError(f"Dataset {d4rl_name} not available in TFDS")
