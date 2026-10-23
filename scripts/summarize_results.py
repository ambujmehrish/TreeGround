import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.run import load_config, runs_root  # noqa: E402

# Prints the V2 dev report, the PACO V4 chain search and every transfer merge in one compact view.


def latest(root: Path, stage: str) -> Path | None:
    runs = sorted(glob.glob(str(root / stage / "2*" / "metrics.json")))
    return Path(runs[-1]) if runs else None


def ci(x) -> str:
    if not isinstance(x, dict) or "mean" not in x:
        return "n/a"
    lo, hi = (x.get("ci95") or [float("nan"), float("nan")])
    return f"{x['mean']:.3f} [{lo:.3f}, {hi:.3f}] n={x.get('n')}"


def search(title: str, summary: dict | None, names) -> None:
    if not isinstance(summary, dict):
        print(f"   {title}: {summary}")
        return
    sub = summary.get("common_eligible_subset", {})
    print(f"   {title}: chains {summary.get('chains')} common eligible {sub.get('size')} ({sub.get('fraction')})")
    for n in names:
        b = summary.get(n) or {}
        print(f"      {n:6s} chain success (common) {ci(b.get('chain_success_common'))}   step2 (common) "
              f"{ci(b.get('step2_success_common'))}   end-to-end {ci(b.get('chain_success_end_to_end'))}   "
              f"feasible {b.get('feasible_path_coverage')}   best-id step2 "
              f"{b.get('diagnostic_best_proposal_identification_step2')}")
    if "diagnostic_best_proposal_identification_step1" in summary:
        print(f"      step 1 picks the best proposal: {summary['diagnostic_best_proposal_identification_step1']}")


def blocks(m: dict, strict, d5, d6, norej=None) -> None:
    for title, blk, names in (("spec", strict, ("flat", "true", "random")), ("D5", d5, ("flat", "true", "random")),
                              ("D6 box", d6, ("flat", "true", "box")),
                              ("D12 no rejection", norej, ("flat", "true", "random"))):
        if not isinstance(blk, dict):
            print(f"   {title}: {blk}")
            continue
        for part in ("all", "two_level", "single_level"):
            if part in blk:
                search(f"{title}/{part}", blk[part], names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/x_transfer.yaml")
    cfg = load_config(ap.parse_args().config)
    root = runs_root(cfg)
    for stage in ("v2", "s1"):
        v2 = latest(root, stage)
        print(f"== PACO {stage.upper()}", v2)
        if not v2:
            continue
        m = json.loads(v2.read_text())
        print(json.dumps({k: v for k, v in m.items() if k not in ("per_category", "curve", "calibration_curve")},
                         indent=1, default=str)[:4000])
    for stage in ("v4", "v4_s1"):
        v4 = latest(root, stage)
        print(f"== PACO {stage.upper()}", v4)
        if not v4:
            continue
        m = json.loads(v4.read_text())
        split = m.get("chain_split_spec_strict", {})
        blocks(m, {"all": m.get("natural_search"), "two_level": split.get("two_level"),
                   "single_level": split.get("single_level")}, m.get("natural_search_d5"),
               m.get("natural_search_d6_box_tree"), m.get("natural_no_rejection_d12"))
    for ds, tag in itertools.product(("pascal_part", "ade20k", "lvis_hier", "openimages_hier"),
                                     ("", "_s1", "_s2", "_s2_from_s1")):
        p = latest(root, f"x_transfer_{ds}{tag}_merge")
        print(f"== transfer {ds}{tag}", p)
        if p:
            m = json.loads(p.read_text())
            print("   coverage", json.dumps(m.get("coverage_uncapped"), default=str))
            blocks(m, m.get("natural_strict"), m.get("natural_d5"), m.get("natural_d6"), m.get("natural_no_rejection"))


if __name__ == "__main__":
    main()
