import gymnasium

from d4rl_slim import infos


def get_environment(dataset_name: str, **env_kwargs):
    """Load a Gymnasium environment compatible with a given dataset."""

    if dataset_name not in infos.list_datasets():
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # mujoco environments are delegated to gymnasium mujoco v4 environments.
    if dataset_name.startswith(("ant-", "halfcheetah-", "hopper-", "walker2d-")):
        env_id, kwargs = {
            "halfcheetah": ("HalfCheetah-v4", {}),
            "hopper": ("Hopper-v4", {}),
            "walker2d": ("Walker2d-v4", {}),
            # v4 ant defaults to not using contact forces.
            "ant": ("Ant-v4", {"use_contact_forces": True}),
        }[dataset_name.split("-")[0]]
        env = gymnasium.make(env_id, **kwargs, **env_kwargs)
        # D4RL uses a NormalizedBox which is only used for normalizing actions
        env = gymnasium.wrappers.RescaleAction(env, -1.0, 1.0)
    else:
        # Fall back to loading from d4rl_slim namespace.
        slim_env_name = f"d4rl_slim/{dataset_name}"
        env = gymnasium.make(slim_env_name, **env_kwargs)

    return env
