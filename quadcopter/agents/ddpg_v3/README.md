# DDPG v3
- add 50% Dropout to all Dense layers for regularization

# DDPG_v2
For v2, taking inspiration from the DDPG paper, I made several changes:
- Use batch normalization after all layers
- Concatenated the action and state pathways rather than adding them
- Added a dense layer after the concatenation
- Set the learning rates as in DDPG paper:
  - 1e-3 for critic
  - 1e-4 for actor
- Used a smaller value of Tau (1e-3)

# DDPG_v1
- Abandoned

# DDPG_v0
- Edited to work with Env instead of Task
