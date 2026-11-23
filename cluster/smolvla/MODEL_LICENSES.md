# SmolVLA and pi0.5 licensing

Checked 2026-11-22 against the [teacher model card](https://huggingface.co/lerobot/pi05_libero_finetuned_v044), [Gemma Terms of Use](https://gemma.ai/resources/terms/) and [Prohibited Use Policy](https://gemma.ai/resources/prohibited-use/).

## What this contribution publishes

The public contribution contains our code, documentation and evaluation measurements. It contains no teacher/student weights or training episodes and provides no hosted inference service. The project's Apache-2.0 source license does not relicense external checkpoints.

Our teacher was `lerobot/pi05_libero_finetuned_v044@7152f13e34c6c115d808aea0e5d08aca74eac14b`, whose card specifies Gemma terms. Its successful action demonstrations trained the teacher-v2, combined-v3 and subsequent continuation students. We treat these students as Model Derivatives under Section 1.1(e), which includes imitation through synthetic outputs. The released SmolVLA checkpoint's separate license remains recorded in THIRD_PARTY_LICENSES.md.

## If a model is shared later

Gemma obligations apply to use as well as distribution. Sharing with teammates or offering hosted access can count as distribution. Hugging Face consent enables access; it does not replace downstream compliance.

Before distributing these student weights or a hosted derivative, supply the applicable agreement, carry forward enforceable use restrictions, identify modifications, and preserve applicable upstream license/copyright notices. Non-hosted distributions also require a `Notice` file containing:

> Gemma is provided under and subject to the Gemma Terms of Use found at gemma.ai/resources/terms

Review all applicable model/backbone terms at release time. Attribution alone is insufficient. Do not label the teacher-trained weights as solely Apache-2.0 or imply Google endorsement.

Generated outputs themselves are not Model Derivatives, and Google claims no rights in them under Section 3.3. Other data/asset rights still apply. This note records the project's handling; it is not a legal certification of a future release.
