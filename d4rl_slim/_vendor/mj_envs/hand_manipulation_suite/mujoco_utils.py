import mujoco

def actuator_name2id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def site_name2id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

def body_name2id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

def joint_name2id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

def sensor_name2id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

