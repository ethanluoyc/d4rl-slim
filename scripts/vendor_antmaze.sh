rm -r d4rl_slim/_vendor/d4rl
pip install --no-deps -t d4rl_slim/_vendor https://github.com/Farama-Foundation/D4RL/archive/71a9549f2091accff93eeff68f1f3ab2c0e0a288.tar.gz
rm -r d4rl_slim/_vendor/*.egg-info
rm -r d4rl_slim/_vendor/d4rl/{carla,flow,gym_*,hand_manipulation_suite,kitchen,pointmaze,pointmaze_bullet}
rm -r d4rl_slim/_vendor/d4rl/__init__.py
rm -r d4rl_slim/_vendor/d4rl/ope.py
rm -r d4rl_slim/_vendor/d4rl/utils
rm -r d4rl_slim/_vendor/d4rl/locomotion/{common.py,generate_dataset.py,swimmer.py,point.py}
rm -r d4rl_slim/_vendor/d4rl/locomotion/assets/point.xml
rm -r d4rl_slim/_vendor/d4rl/locomotion/__init__.py

