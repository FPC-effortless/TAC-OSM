# Pinned publisher field contract

Source: BodenAI/RW-RL-HIL-Dataset, CC-BY-4.0, pinned revision
9290f5a96dc91ea58256e81d9ce2dd9621890b51.
https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset/blob/9290f5a96dc91ea58256e81d9ce2dd9621890b51/README.md

The publisher describes synchronized14-dimensional `observations.state.qpos`
as robot state and `action.qpos` as robot action. The action stream contains
policy and human authority; original autonomous masks remain compulsory.
The15FPSsaved timestamp is retained-trajectory time; observed capture elapsed
is distinct. These facts do not identify torque, intervention effects or
future action availability. ME090 predicts later recorded action.qpos as an
auxiliary reference target; it never supplies true future references to the
model. No autonomous action selection or real policy value is claimed.
