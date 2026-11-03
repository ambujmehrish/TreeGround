# D44: query-aware tree with object-anchored selection

Status: spec for discussion before implementation. Nothing here is implemented or run.

## Problem (from measurements)

The tree-attention models (D41/D43) select from the grid tree: SAM 3 automatic masks (32x32 points, dedup 0.9,
cap 256). On PACO val parts, a mask with IoU >= 0.5 to the target exists in that tree for 46% of queries
(box_coverage_zero, dev). No selector, attention pattern or loss can pass that ceiling, which is why full, tree
and shuffled-tree attention tie on validation and why the test numbers trail SAM 3 text on InstructPart and
PartImageNet.

SAM 3 text (same model, same image encoding, one cheap decoder call per phrase) reaches 44% with its top
instance and 55% with its top three. Grid + text top-3 for the part phrase and its object phrase: 62%. Qwen
boxes: 30-34%, below the grid tree; they are dropped for good.

The gap between SAM 3 text's top-1 (44%) and top-3 (55%) is a choice between masks of the same phrase at
different levels or instances. That is the decision a part-whole tree is for.

## Method: two changes to the current method, nothing else

### 1. Query-aware tree

Candidates per image = grid nodes (as now) + SAM 3 text instances, top-3 per phrase, for the phrases of the
image's queries (the part phrase and, for part queries, the object phrase), duplicates at IoU >= 0.95 dropped
(s3_pool.augment, D30). One tree over all candidates by the existing containment rule (T.build_tree). A text
instance therefore sits under the grid node that contains it, or above the grid nodes it contains.

Node token inputs (TA.NodeEmbed) gain 5 label-free values next to the 9 geometry values: source flag (grid /
text), one-hot rank among the phrase's instances (1, 2, 3), and the SAM 3 instance score (0 for grid nodes).
Rank matters: rank 1 is right less than half the time.

Nothing else in the token construction changes: coverage-weighted image-token embedding, dilated region,
centroid mRoPE position, ScaledSelector heads at layers 12 / 24 / 36.

### 2. Object-anchored part selection

A part query is answered as a chain: two selection tokens in the same forward pass, object then part (the
chain queries of PACO and Pascal-Part already train this; CHAIN_TEMPLATE "the {part} of the {object}").
- the object token selects the object node (top-1 over the tree);
- the part token scores only nodes in the selected object's subtree (hard), or every node with a bias
  lambda on the subtree (soft; lambda chosen on val). Hard is the existing tree-restricted chain decoding of
  s3_eval (part_tree_iou); soft is new.
Fallback: if the selected object's subtree is empty, the part token scores every node (flat).

The part token attends to the object token and, through C's soft tree bias, to the tree relations of every
candidate, so the chosen object is visible to it as a selected ancestor. B's family loss trains the part
positive to beat its own parent and children, which is the level decision above.

Nothing hierarchical multiplies probabilities down the path (D42's failure): one object decision, one part
decision inside it.

### Unchanged

Backbone Qwen3-VL-4B-Instruct + LoRA (same targets, r, lr), trained from scratch, focal selection loss, LM
loss, text-only items, batch 32 (0.6 / 0.2 / 0.2 PACO / Pascal / gRefCOCO), 500 warm-up, readout layers
12 / 24 / 36, D43 soft tree bias (C) and family loss (B) as in commit 5bb4054b.

## Data

Training reads the D30 pool caches (s3_pool: paco_train = the 10k worklist images, pascal_part_train,
grefcoco_train). They hold, per image: grid nodes, text instances with phrase / rank / score, the tree over
the pool, and the targets (chain paths by max-IoU path, single-step targets, no-target items). No cache job.

The grid-only control is the existing ta_full run, trained on the 10k worklist + the 34k PACO rest images.
It has more training images than any D44 row, so a D44 gain cannot come from data.

Validation: the fixed PACO val subset of ta_train (selector success if covered), per layer; used only to pick
variant, layer, checkpoint and lambda.

## Inference

Per image: SAM 3 image encoding once; grid nodes as now; per query, text instances for its phrase(s); one tree
per query over grid + that query's instances (at test the pool is per query, not per image, so the other
queries of an image add nothing). Part queries on target-only benchmarks (PACO test, PARTONOMY, OP-HRG) run as
chains with the object name from the query; gRefCOCO expressions have no object / part split and run as single
selections over grid + the expression's instances, with the trained no-target rule.

Cost over the current method: one SAM 3 text decoder call per phrase (shared image encoding) and up to 6
extra node tokens per query.

## Rows (val picks the variant; the table is test only, published 4B rows alongside)

| Row | Tree | Selector | Part decision | Tests |
|---|---|---|---|---|
| SAM 3 text (published + our run) | - | top-1 | - | pool source alone |
| ta_full (D41, running on test) | grid | full attn | single | pool source alone |
| D44-flat | grid + text | full attn | single | pool helps, hierarchy unused |
| D44-C | grid + text | soft tree bias | single | tree as attention |
| D44-C-anchor (hard / soft) | grid + text | soft tree bias | chain, subtree | tree as decision |
| D44-C-B-anchor | grid + text | soft tree bias + family | chain, subtree | level discrimination |
| shuffled-tree control of the best row | shuffled | same | same | the structure, not parameters |

Trainings: D44-flat, D44-C, D44-C-B, and the shuffled control of whichever of C / C-B val picks: 4 runs,
4 GPUs each, AIFAC_S07_041 only. Anchoring is a decoding choice (hard / soft / none) on the same
checkpoint, so it costs no training. Steps as D41 (12k) unless the val curve is flat earlier.

Go / no-go on val before any test run: D44-C-anchor must beat D44-flat on part success; otherwise hierarchy
is still not working and we stop before spending test compute.

## Timeline (CVPR, 15 days from Nov 2)

- Nov 3-4: implement (pool source for ta_train, node features, chain decoding with subtree restriction in
  ta_eval / ta_test_eval), tests, code review, 20-step check run on AIFAC.
- Nov 5-7: 4 trainings in parallel on AIFAC.
- Nov 8: val choice; go / no-go.
- Nov 8-10: test evals on RHYME (PARTONOMY, InstructPart, PartImageNet, gRefCOCO testA/B, PACO test).
- Nov 3-15: paper in parallel.

## Open decisions

1. Pool per query at test vs per image as in the cache (proposed: per query at test; training keeps the
   per-image cache because that is what exists).
2. Soft anchoring bias: one scalar lambda chosen on val, or a learned scalar trained with the chain loss
   (proposed: val-chosen scalar; nothing new to train, and the hard version is the ablation).
3. The 34k PACO rest images: not in the D30 pool. Proposed: train D44 without them; add a text-augmented rest
   cache only if a run finishes early and val says more data helps.
