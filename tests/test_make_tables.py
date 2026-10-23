import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_tables", ROOT / "scripts" / "make_tables.py")
MT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MT)


def ci(m, lo, hi, n=10):
    return {"mean": m, "ci95": [lo, hi], "n": n}


def single_step(g):
    return {"iii_end_to_end_gIoU": {"present": ci(g, g - 0.01, g + 0.01), "no_target_abstention_rate": 0.9},
            "ii_selector_success": {"all": ci(0.5, 0.45, 0.55)}, "queries": {"present": 100, "no_target": 50}}


def summary(base, trees=("flat", "true", "random")):
    return {"chains": 40, "common_eligible_subset": {"size": 12, "fraction": 0.3},
            **{t: {"chain_success_common": ci(base + i / 10, base, base + 0.2),
                   "chain_success_end_to_end": ci(base / 2, base / 3, base), "feasible_path_coverage": 0.6}
               for i, t in enumerate(trees)}}


def search(base):
    return {"natural_search": summary(base),
            "chain_split_spec_strict": {"two_level": summary(base / 2), "single_level": None},
            "natural_search_d5": "not computed: deviation D5 is not approved in deviations.jsonl",
            "natural_search_d6_box_tree": {"all": summary(base, ("flat", "true", "box")), "two_level": None,
                                           "single_level": None},
            "natural_no_rejection_d12": "not computed: deviation D12 is not approved in deviations.jsonl"}


def baseline(g):
    blk = {"n": 100, "gIoU": ci(g, g - 0.02, g + 0.02), "cIoU": {"value": 0.4, "ci95": [0.35, 0.45]},
           "success@0.5": ci(0.3, 0.25, 0.35)}
    return {"present": blk, "by_kind": {"object": {**blk, "gIoU": ci(0.6, 0.5, 0.7)}, "part": blk},
            "no_target": {"n": 50, "abstention_rate": 0.2},
            "chains": {k: {"chains": 40, "chain_success_end_to_end": ci(0.1, 0.05, 0.15)}
                       for k in ("all", "two_level", "single_level")}}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def setup(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    stages = {"v2": single_step(0.273), "s2_eval": {"single_step": single_step(0.3), "chains": search(0.2)},
              "v4": search(0.3), "b_sa2va_merge": baseline(0.35),
              "x_transfer_ade20k_merge": {"coverage_uncapped": {"part_recall": 0.64, "chain_feasible_mask_tree": 0.5},
                                          "natural_strict": {"all": summary(0.05)}, "natural_d5": None,
                                          "natural_no_rejection": "not reported: deviation D12 is not approved"},
              "v1": {"by_tau": {"0.9": {"paco": {"part_annotations": 20, "part_recall": 0.4,
                                                 "by_area_bin": {"small": {"n": 5, "recall": 0.25}},
                                                 "feasible_path_coverage": 0.5}, "sceneparser": "missing"}}}}
    for k, v in stages.items():
        (m / f"{k}.json").write_text(json.dumps(v))
    out = tmp_path / "tables"
    for name, build in MT.TABLES.items():
        MT.write(out, name, *build(m))
    return out


def test_retrieval_rows_use_the_same_definitions_and_dash_missing(tmp_path):
    out = setup(tmp_path)
    rows = {r["method"]: r for r in read(out / "retrieval_paco.csv")}
    assert len(rows) == 7
    v2, sa2va, s2 = rows["V2 (frozen, 10k img.)"], rows["Sa2VA"], rows["S2"]
    assert (v2["gIoU"], v2["gIoU_lo"], v2["n_acc"], v2["queries"]) == ("0.273", "0.263", "0.900", "100")
    assert v2["cIoU"] == v2["success_0.5"] == "–" and v2["selector_success_covered"] == "0.500"
    assert s2["gIoU"] == "0.300"
    assert (sa2va["cIoU"], sa2va["cIoU_hi"], sa2va["gIoU_object"], sa2va["n_acc"]) == ("0.400", "0.450", "0.600", "0.200")
    assert sa2va["selector_success_covered"] == "–"
    assert all(v == "–" for k, v in rows["S1"].items() if k != "method")
    tex = (out / "retrieval_paco.tex").read_text()
    assert r"27.3 \scriptsize{[26.3, 28.3]}" in tex and r"\toprule" in tex and r"\bottomrule" in tex
    assert "S1 & -- & --" in tex


def test_chain_table_skips_uncomputed_blocks_and_keeps_missing_stages(tmp_path):
    rows = read(setup(tmp_path) / "chains_paco.csv")
    v2 = [r for r in rows if r["method"].startswith("V2")]
    assert {r["search"] for r in v2} == {"spec", "D6 box tree"}
    flat = next(r for r in v2 if r["search"] == "spec" and r["split"] == "all" and r["tree"] == "flat")
    assert (flat["success_common"], flat["end_to_end"], flat["feasible"], flat["common"]) == \
        ("0.300", "0.150", "0.600", "12")
    single = [r for r in v2 if r["search"] == "spec" and r["split"] == "single-level"]
    assert len(single) == 3 and all(r["success_common"] == "–" for r in single)
    assert {r["tree"] for r in v2 if r["search"] == "D6 box tree"} == {"flat", "true", "box"}
    assert len([r for r in rows if r["method"] == "S1"]) == 9
    assert any(r["method"] == "S2" and r["success_common"] == "0.200" for r in rows)
    sa2va = [r for r in rows if r["method"] == "Sa2VA"]
    assert len(sa2va) == 3 and sa2va[0]["end_to_end"] == "0.100" and sa2va[0]["success_common"] == "–"


def test_transfer_and_coverage(tmp_path):
    out = setup(tmp_path)
    rows = read(out / "transfer.csv")
    ade = [r for r in rows if r["dataset"] == "ADE20K"]
    assert [(r["selector"], r["search"]) for r in ade] == [("V2", "spec"), ("S1", "spec")]
    assert (ade[0]["part_recall"], ade[0]["random"], ade[0]["box"]) == ("0.640", "0.250", "–")
    assert all(r["flat"] == "–" for r in rows if r["dataset"] != "ADE20K")
    cov = read(out / "coverage.csv")
    assert (cov[0]["recall"], cov[0]["small"], cov[0]["medium"], cov[0]["feasible"]) == ("0.400", "0.250", "–", "0.500")
    assert all(r["recall"] == "–" for r in cov if r["source"].startswith("oracle"))
    assert "40.0" in (out / "coverage.tex").read_text()


def test_tex_escaping():
    assert MT.esc("a_b & 5%") == r"a\_b \& 5\%"
    assert MT.tex_cell("ci", {"value": 0.5, "ci95": [float("nan"), 0.6]}) == "50.0"
    assert MT.csv_cells("ci", "not computed") == ["–", "–", "–"]
