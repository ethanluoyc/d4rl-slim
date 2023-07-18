from gymnasium.envs.registration import _register


def register(id, **kwargs):
    namespaced_id = f"d4rl_slim/{id}"
    return _register(namespaced_id, **kwargs)
