# Prior-work check: candidates as tokens inside the MLLM (2026-11-05)

Read from the papers themselves (alphaXiv full text), not from memory.

| Work | Candidates | How candidates enter the LLM | Do candidate tokens see the query? | Selection | Overlap with ours |
|---|---|---|---|---|---|
| GROUNDHOG (CVPR'24, 2402.16846) | class-agnostic masks, Mask2Former+ (parts included) | each mask -> mask-pooled CLIP+DINOv2 feature -> "visual entity token", the LLM's visual input | no: entity tokens are the visual input before the text | MLP on [grounding-query ; entity-token output] | candidates as LLM tokens; per-mask scoring |
| Groma (ECCV'24, 2404.13013) | box proposals (DDETR) | ROIAlign region tokens in the input, referenced by proxy tokens | region tokens precede the query | LLM writes the proxy token | proposals as LLM tokens (boxes) |
| PSALM (ECCV'24, 2403.14598) | learned mask queries (Mask2Former decoder) | N mask tokens appended after the prompt; ablation: after the prompt beats before it | yes | condition embedding classifies the decoded masks | query-attending tokens after the prompt (but queries, not proposals) |
| RAS (2506.05342, UIUC/Adobe) | SAM automatic masks (ORES), Co-DETR (RES/GRES) | mask-pooled features of a 4-encoder ensemble + sinusoidal position map -> mask tokens; context first, then all candidate mask tokens fed to the LLM | yes: candidate tokens come after the prompt | binary classifier on each candidate token's output state (non-autoregressive, one pass) | closest: query-attending candidate mask tokens inside the LLM, one-pass selection, per-candidate classification; also reports oracle candidate quality |
| STAMP / STAMPlus (CVPR'26, 2608.02791) | none: one [MASK] token per image patch | patch-aligned [MASK] tokens after the dialogue, hybrid (bidirectional) attention | yes | per-patch FG/BG classifier | in-sequence tokens after the answer, custom attention mask |
| PANORAMA (2609.19143) | SAM 3 proposals conditioned on the [SEG] concept (200 queries) | not in the LLM: SAM 3's own scorer compares concept vs proposal queries | proposals are phrase-conditioned | dot product, threshold 0.5 | query-conditioned proposal pool; selection from SAM 3 proposals |
| SetCon (NeurIPS'26, 2605.20110) | SAM 3 decoder conditioned on concept-span states | not in the LLM | n/a | DETR-style set decoding | SAM 3 + Qwen3-VL |
| DRAgent (2608.22885) | Grounding DINO boxes | Set-of-Mark overlays, yes/no verification per candidate | yes (visual prompt) | yes-logit ranking | MLLM as discriminator over candidates |

## What remains specific to us

- Candidate pool = grid masks + SAM 3 text instances of the query phrases (rank-tagged), one containment tree.
  PANORAMA conditions proposals on the query; GROUNDHOG/RAS use a fixed per-image pool. The union, measured
  (46% / 44% / 62% coverage on PACO parts), is ours; GROUNDHOG Table 1 and RAS Table 6 report proposal / oracle
  quality, so the *kind* of analysis has precedent, the part-level numbers do not.
- Mask-token construction: pooled from the LLM's own image-token input embeddings (no external encoders), attention
  restricted to the mask's dilated region, rotary position at the mask centroid. RAS pools external encoders and
  lets candidate tokens attend freely. Untested whether these details matter (pool_flat compares only against the
  pooled-feature selector, not against RAS-style tokens).
- Part focus: object->part chains, tree targets, part benchmarks (PARTONOMY, InstructPart, PartImageNet, PACO parts),
  none of which GROUNDHOG, RAS or PANORAMA report.
- Tree attention / family loss / anchoring: ours, measured null on test.

## Must cite and discuss

GROUNDHOG, Groma, PSALM, RAS, STAMP, PANORAMA, SetCon, SAMTok, OP-HRG, PARTONOMY.
RAS is the paper a reviewer will put next to ours.
