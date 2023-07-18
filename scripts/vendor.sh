#!/usr/bin/env bash

MJRL_PATH=third_party/mjrl

MJRL_SHA256=`git --git-dir=$MJRL_PATH/.git rev-parse HEAD`
rm -rf d4rl_slim/_vendor/mjrl
mkdir -p d4rl_slim/_vendor/mjrl
cp $MJRL_PATH/mjrl/envs/mujoco_env.py d4rl_slim/_vendor/mjrl/mujoco_env.py
echo $MJRL_SHA256 > d4rl_slim/_vendor/mjrl/GIT_COMMIT

MJENV_PATH=third_party/mj_envs

MJRL_SHA256=`git --git-dir=$MJENV_PATH/.git rev-parse HEAD`
rm -rf d4rl_slim/_vendor/mj_envs
mkdir -p d4rl_slim/_vendor/mj_envs
cp -r $MJENV_PATH/mj_envs d4rl_slim/_vendor/
cp -r $MJENV_PATH/dependencies/Adroit/resources d4rl_slim/_vendor/mj_envs/hand_manipulation_suite/assets
echo $MJENV_SHA256 > d4rl_slim/_vendor/mj_envs/GIT_COMMIT
find d4rl_slim/_vendor/mj_envs -name "*.py" -type f -exec sed -i -e 's/from mjrl.envs/from d4rl_slim._vendor.mjrl/g' {} \;
find d4rl_slim/_vendor/mj_envs -name "*.py" -type f -exec sed -i -e 's/from mj_envs/from d4rl_slim._vendor.mj_envs/g' {} \;
sed -i -e 's/import  mj_envs.hand_manipulation_suite/import d4rl_slim._vendor.mj_envs.hand_manipulation_suite/g' d4rl_slim/_vendor/mj_envs/__init__.py
sed -i -e 's/..\/..\/..\/dependencies\/Adroit\///g' d4rl_slim/_vendor/mj_envs/hand_manipulation_suite/assets/DAPG_assets.xml
