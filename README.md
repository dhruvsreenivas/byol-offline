# BYOL-Offline
Bootstrap your own latent (BYOL) [[1]](https://arxiv.org/abs/2006.07733) [[2]](https://arxiv.org/abs/2206.08332) and other methods commonly used in exploration applied in offline reinforcement learning.

## Updates
- Fixed JAX Dreamer setup -- VAE now trains fine.
- BYOL loss looks fine to me as well, but need to test this more to make sure that the uncertainty quant actually makes sense (i.e. implement online with PPO and see if it works on M-Revenge or something).
- Better implementations of DDPG and SAC that are more conducive to model-based RL.
