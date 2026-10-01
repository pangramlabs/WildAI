"""Model, data mixtures and training loop for the WildAI models (adapted from nanochat; see NOTICE).

- `model`: the GPT (`GPT`, `GPTConfig`); `checkpoint.load_model` loads trained checkpoints.
- `recipe`: the paper's training recipe as typed defaults.
- `pools`: tokenize raw document pools (`python -m wildai.training.pools`).
- `mixture`: which documents a run trains on and in what order; `registry` maps paper model names to runs.
- `train`: the training CLI (`python -m wildai.training.train`).
"""
