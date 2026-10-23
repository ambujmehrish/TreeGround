import argparse
import csv
import json
import math
from pathlib import Path

# Paper tables from the published results/metrics/<stage>.json files: one CSV (fractions) and one booktabs LaTeX
# tabular (percentages) per table. A stage that has not been published yet, or a block that was not computed,
# is shown as a dash.

DASH = "–"
OURS = (("v2", "V2 (frozen, 10k img.)"), ("s1", "S1"), ("s2_eval", "S2"), ("s2_from_s1_eval", "S2 (from S1)"))
CHAIN_OURS = (("v4", OURS[0][1]), ("v4_s1", "S1")) + OURS[2:]
BASELINES = (("b_sam3_text_merge", "SAM 3 (text)"), ("b_qwen_box_sam_merge", "Qwen3-VL box + SAM 3"),
             ("b_sa2va_merge", "Sa2VA"))
SPLITS = (("all", "all"), ("two_level", "two-level"), ("single_level", "single-level"))
DATASETS = (("pascal_part", "Pascal-Part"), ("ade20k", "ADE20K"), ("lvis_hier", "LVIS hier."),
            ("openimages_hier", "OpenImages hier."))
BINS = ("small", "medium", "large")


def get(m, *path):
    for k in path:
        if not isinstance(m, dict) or k not in m:
            return None
        m = m[k]
    return m


def num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def stat(x):
    if isinstance(x, dict):
        v = num(x.get("mean", x.get("value")))
        ci = x.get("ci95")
        lo, hi = (num(ci[0]), num(ci[1])) if isinstance(ci, (list, tuple)) and len(ci) == 2 else (None, None)
        return None if v is None else (v, lo, hi)
    v = num(x)
    return None if v is None else (v, None, None)


def load(root: Path, stage: str):
    p = root / f"{stage}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m if isinstance(m, dict) else None


def esc(s: str) -> str:
    s = s.replace("\\", "\0")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s.replace("\0", r"\textbackslash{}")


def csv_cells(kind: str, v) -> list[str]:
    if kind == "text":
        return [DASH if v is None else str(v)]
    if kind == "int":
        return [DASH if num(v) is None else str(int(v))]
    s = stat(v)
    cells = [DASH if s is None else f"{s[0]:.3f}"]
    if kind == "ci":
        cells += [DASH if s is None or s[i] is None else f"{s[i]:.3f}" for i in (1, 2)]
    return cells


def tex_cell(kind: str, v) -> str:
    if kind == "text":
        return "--" if v is None else esc(str(v))
    if kind == "int":
        return "--" if num(v) is None else str(int(v))
    s = stat(v)
    if s is None:
        return "--"
    out = f"{100 * s[0]:.1f}"
    if kind == "ci" and s[1] is not None and s[2] is not None:
        out += f" \\scriptsize{{[{100 * s[1]:.1f}, {100 * s[2]:.1f}]}}"
    return out


def write(out: Path, name: str, cols: list[tuple], rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for key, _, kind in cols for c in ([key, f"{key}_lo", f"{key}_hi"] if kind == "ci" else [key])])
        for r in rows:
            w.writerow([c for key, _, kind in cols for c in csv_cells(kind, r.get(key))])
    lines = [r"\begin{tabular}{" + "".join("l" if k == "text" else "r" for _, _, k in cols) + "}", r"\toprule",
             " & ".join(h for _, h, _ in cols) + r" \\", r"\midrule"]
    group = None
    for r in rows:
        if group is not None and r.get("_group") != group:
            lines.append(r"\midrule")
        group = r.get("_group")
        lines.append(" & ".join(tex_cell(kind, r.get(key)) for key, _, kind in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")


def retrieval(root: Path) -> tuple[list, list]:
    cols = [("method", "Method", "text"), ("queries", "Queries", "int"),
            ("gIoU", "gIoU", "ci"), ("cIoU", "cIoU", "ci"), ("gIoU_object", "gIoU obj.", "ci"),
            ("gIoU_part", "gIoU part", "ci"), ("success_0.5", r"Succ.@0.5", "ci"),
            ("n_acc", "N-acc", "frac"), ("no_target", "No-target", "int"),
            ("selector_success_covered", r"Sel. succ. $\mid$ cov.", "ci")]
    rows = []
    for stage, label in OURS:
        m = load(root, stage)
        m = get(m, "single_step") if stage.startswith("s2") else m
        rows.append({"_group": "ours", "method": label, "queries": get(m, "queries", "present"),
                     "gIoU": get(m, "iii_end_to_end_gIoU", "present"),
                     "cIoU": get(m, "iii_end_to_end_gIoU", "cIoU_present"),
                     "gIoU_object": get(m, "iii_end_to_end_gIoU", "by_kind", "object"),
                     "gIoU_part": get(m, "iii_end_to_end_gIoU", "by_kind", "part"),
                     "success_0.5": get(m, "iii_end_to_end_gIoU", "success@0.5_present"),
                     "n_acc": get(m, "iii_end_to_end_gIoU", "no_target_abstention_rate"),
                     "no_target": get(m, "queries", "no_target"),
                     "selector_success_covered": get(m, "ii_selector_success", "all")})
    variants = get(load(root, "e2e_variants"), "stages") or {}
    for stage, label in OURS:
        v = variants.get(stage)
        if not isinstance(v, dict):
            continue
        nr = v["no_rejection_present"]
        rows.append({"_group": "ours (no rejection, D15)", "method": f"{label}, no rejection",
                     "queries": v["present_queries"], "gIoU": nr.get("gIoU"), "cIoU": nr.get("cIoU"),
                     "gIoU_object": get(nr, "by_kind", "object"), "gIoU_part": get(nr, "by_kind", "part"),
                     "success_0.5": nr.get("success@0.5"), "no_target": v["no_target_queries"]})
    for stage, label in BASELINES:
        m = load(root, stage)
        rows.append({"_group": "baselines", "method": label, "queries": get(m, "present", "n"),
                     "gIoU": get(m, "present", "gIoU"), "cIoU": get(m, "present", "cIoU"),
                     "gIoU_object": get(m, "by_kind", "object", "gIoU"),
                     "gIoU_part": get(m, "by_kind", "part", "gIoU"), "success_0.5": get(m, "present", "success@0.5"),
                     "n_acc": get(m, "no_target", "abstention_rate"), "no_target": get(m, "no_target", "n")})
    return cols, rows


def chain_rows(label: str, search: str, blk, trees, group: str) -> list[dict]:
    rows = []
    for split, split_label in SPLITS:
        s = get(blk, split)
        for tree in trees:
            rows.append({"_group": group, "method": label, "search": search, "split": split_label, "tree": tree,
                         "chains": get(s, "chains"), "common": get(s, "common_eligible_subset", "size"),
                         "success_common": get(s, tree, "chain_success_common"),
                         "end_to_end": get(s, tree, "chain_success_end_to_end"),
                         "feasible": get(s, tree, "feasible_path_coverage")})
    return rows


CHAIN_COLS = [("method", "Method", "text"), ("search", "Search", "text"), ("split", "Split", "text"),
              ("tree", "Tree", "text"), ("chains", "Chains", "int"), ("common", "Common", "int"),
              ("success_common", "Chain succ. (common)", "ci"), ("end_to_end", "End-to-end", "ci"),
              ("feasible", "Feasible", "frac")]


def chains(root: Path) -> tuple[list, list]:
    rows = []
    rand = ("flat", "true", "random")
    for stage, label in CHAIN_OURS:
        m = load(root, stage)
        m = get(m, "chains") if stage.startswith("s2") else m
        strict = {"all": get(m, "natural_search"), "two_level": get(m, "chain_split_spec_strict", "two_level"),
                  "single_level": get(m, "chain_split_spec_strict", "single_level")}
        rows += chain_rows(label, "spec", strict, rand, stage)
        for key, search, trees in (("natural_search_d5", "D5 inclusive", rand),
                                   ("natural_search_d6_box_tree", "D6 box tree", ("flat", "true", "box")),
                                   ("natural_no_rejection_d12", "D12 no rejection", rand)):
            if isinstance(get(m, key), dict):
                rows += chain_rows(label, search, m[key], trees, f"{stage}:{key}")
    for stage, label in BASELINES:
        m = load(root, stage)
        for split, split_label in SPLITS:
            s = get(m, "chains", split)
            rows.append({"_group": stage, "method": label, "search": "independent", "split": split_label,
                         "chains": get(s, "chains"), "end_to_end": get(s, "chain_success_end_to_end")})
    return CHAIN_COLS, rows


def transfer(root: Path) -> tuple[list, list]:
    cols = [("dataset", "Dataset", "text"), ("selector", "Selector", "text"), ("search", "Search", "text"),
            ("part_recall", "Part recall", "frac"), ("chain_feasible", "Chain feasible", "frac"),
            ("chains", "Chains", "int"), ("common", "Common", "int"), ("flat", "Flat", "ci"), ("true", "True", "ci"),
            ("random", "Random", "ci"), ("box", "Box", "ci")]
    rows = []
    for ds, ds_label in DATASETS:
        for tag, sel in (("", "V2"), ("_s1", "S1")):
            m = load(root, f"x_transfer_{ds}{tag}_merge")
            cov = {"part_recall": get(m, "coverage_uncapped", "part_recall"),
                   "chain_feasible": get(m, "coverage_uncapped", "chain_feasible_mask_tree")}
            for key, search in (("natural_strict", "spec"), ("natural_d5", "D5 inclusive"),
                                ("natural_d6", "D6 box tree"), ("natural_no_rejection", "D12 no rejection")):
                blk = get(m, key)
                if key != "natural_strict" and not isinstance(blk, dict):
                    continue
                s = get(blk, "all")
                rows.append({"_group": ds, "dataset": ds_label, "selector": sel, "search": search, **cov,
                             "chains": get(s, "chains"), "common": get(s, "common_eligible_subset", "size"),
                             **{t: get(s, t, "chain_success_common") for t in ("flat", "true", "random", "box")}})
    return cols, rows


def bins(b, inner: bool) -> dict:
    return {k: get(b, k, "recall") if inner else get(b, k) for k in BINS}


def coverage(root: Path) -> tuple[list, list]:
    cols = [("source", "Source", "text"), ("setting", "Setting", "text"), ("n", "N", "int"),
            ("recall", "Recall", "frac"), *[(b, b.capitalize(), "frac") for b in BINS],
            ("feasible", "Feasible path", "frac"), ("best_match", "Best-match path", "frac"),
            ("both_found", "Both found", "frac"), ("feasible_given_both", r"Feasible $\mid$ both", "frac")]
    rows = []
    v1 = load(root, "v1")
    taus = sorted(get(v1, "by_tau") or {}) or ["0.9"]
    for tau in taus:
        p = get(v1, "by_tau", tau, "paco")
        rows.append({"_group": "v1", "source": "PACO parts", "setting": f"final tree, dedup {tau}",
                     "n": get(p, "part_annotations"), "recall": get(p, "part_recall"),
                     **bins(get(p, "by_area_bin"), True), "feasible": get(p, "feasible_path_coverage"),
                     "best_match": get(p, "best_match_path_coverage")})
    pre = get(v1, "diagnostic_pre_dedup")
    rows.append({"_group": "v1", "source": "PACO parts", "setting": "before dedup", "recall": get(pre, "part_recall"),
                 **bins(get(pre, "by_area_bin"), True)})
    for tau in taus:
        p = get(v1, "by_tau", tau, "partonomy")
        if isinstance(p, dict):
            rows.append({"_group": "v1", "source": "PARTONOMY parts", "setting": f"final tree, dedup {tau}",
                         "n": get(p, "part_annotations"), "recall": get(p, "part_recall")})
    d4 = load(root, "d4_node_cap")
    k, chosen = get(d4, "k"), get(d4, "chosen_order")
    for order in ("uncapped", "stability", "sam_iou"):
        p = get(d4, "paco_val", order)
        setting = "no node cap" if order == "uncapped" else \
            f"cap {k if k is not None else DASH} by {order}" + (" (chosen)" if order == chosen else "")
        rows.append({"_group": "d4", "source": "PACO parts", "setting": setting, "n": get(d4, "paco_val", "parts"),
                     "recall": get(p, "part_recall"), **bins(get(p, "by_area_bin"), False)})
    ceil = load(root, "v1_ceiling")
    objs = get(ceil, "objects") or {}
    for key in sorted(k for k in objs if k != "pre_dedup") + ["pre_dedup"]:
        o = objs.get(key)
        rows.append({"_group": "objects", "source": "PACO objects",
                     "setting": "before dedup" if key == "pre_dedup" else f"final tree, dedup {key}",
                     "n": get(o, "n"), "recall": get(o, "recall"), **bins(get(o, "by_area_bin"), True)})
    oracle = get(ceil, "oracle")
    for prompt in sorted(oracle) if isinstance(oracle, dict) else ["part/point+parent_box"]:
        for variant in ("any_of_3_masks", "top_scored_mask", "stability_and_area_filtered"):
            o = get(oracle, prompt, variant)
            rows.append({"_group": "oracle", "source": f"oracle {prompt}", "setting": variant.replace("_", " "),
                         "n": get(o, "n"), "recall": get(o, "recall"), **bins(get(o, "by_area_bin"), True)})
    cs = load(root, "v1_chain_split")
    tag = get(cs, "proposal_setting")
    for tau in sorted(get(cs, "by_tau") or {}) or ["0.9"]:
        c = get(cs, "by_tau", tau)
        rows.append({"_group": "chain_split", "source": "PACO chains (diagnostic)",
                     "setting": f"{tag or DASH}, dedup {tau}", "n": get(c, "chains"), "recall": get(c, "part_found"),
                     "feasible": get(c, "tree_feasible"), "both_found": get(c, "both_found"),
                     "feasible_given_both": get(c, "tree_feasible_given_both_found")})
    return cols, rows


TABLES = {"retrieval_paco": retrieval, "chains_paco": chains, "transfer": transfer, "coverage": coverage}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    for name, build in TABLES.items():
        cols, rows = build(Path(args.metrics))
        write(Path(args.out), name, cols, rows)
        print(f"{name}: {len(rows)} rows -> {args.out}/{name}.{{csv,tex}}")


if __name__ == "__main__":
    main()
