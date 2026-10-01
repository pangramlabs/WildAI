"""Evaluation of WildAI checkpoints.

- `bpb`: bits per byte on C4, FW22, FW26, FW26-H, FW26-AI and Cosmopedia (`python -m wildai.evaluation.bpb`).
- `paloma`: document-bounded macro BPB over Paloma's 16 sources (`python -m wildai.evaluation.paloma`).
- `downstream`: CORE and 5-shot MMLU (`python -m wildai.evaluation.downstream`).
- `generate`, `phrases`, `labels`: sampled continuations, their AI-typical phrase rate and the share a detector labels AI.
"""
