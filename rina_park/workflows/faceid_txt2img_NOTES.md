# Identity-lock readiness (commercial platform)

Do not use InsightFace-provided AntelopeV2/Buffalo pretrained weights for
Instagram or Patreon output. The InsightFace code license and pretrained
weight terms are different; the provided weights are non-commercial research
only.

Platform identity lock stays disabled until the registry contains a
commercially approved CLIP-based adapter or self-trained LoRA with exact
source, revision, SHA-256, license, attribution and local path. The exact
component must pass `factory.readiness.identity_lock_readiness` and the fixed
12-scene benchmark at 12/12.

`devenira_rina_park` remains a metadata label, not a trigger token, until that
gate passes. Readiness checks do not download weights.
