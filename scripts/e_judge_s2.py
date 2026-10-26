import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import e_edit_s2 as ES2  # noqa: E402
import e_judge as EJ  # noqa: E402
from treeground.run import ProtocolError, canonical_hash, read_json, resolve_run  # noqa: E402

# Deviation D29: the D18 judge (Qwen2.5-VL-72B, VIEScore, both variants, unchanged prompts and settings) on the D23
# edit conditions (S2-family selector masks, flat and tree) of the config's S2 stage, read from the D23 merge's own
# image directory; the D14 conditions of the same samples are already judged (judge3) and are not re-judged.

_load = EJ.load_edit_script


def stage(cfg: dict) -> str:
    return ES2.ES.X2.S2T.s2_stage(cfg)


def load_edit_script():
    m = _load()
    m.conditions = lambda cfg, editor: list(cfg["edit_s2"]["conditions"][editor])
    m.stage_tag = lambda cfg: f"_{stage(cfg)}"
    return m


def edit_cache(cfg: dict, editor: str, dataset: str, smoke: bool, E):
    items = ES2.E.evaluated(ES2.E.samples(cfg, dataset), smoke, cfg["edit"]["smoke_samples"])
    ident = ES2.identity(cfg, editor, dataset, items, smoke)
    d = ES2.out_dir(cfg, editor, dataset, ident, smoke)
    merge_run = resolve_run(cfg, f"e_{editor}_{dataset}_{stage(cfg)}_merge", cfg["inputs"].get("edit_merge", "latest"),
                            smoke)
    if read_json(merge_run / "metrics.json")["images_dir"].get("d23") != str(d / "images"):
        raise ProtocolError(f"{merge_run} was merged from another D23 edit cache than {d}")
    return d, canonical_hash(ident), items, merge_run


_required = EJ.required
EJ.load_edit_script, EJ.edit_cache = load_edit_script, edit_cache
EJ.required = lambda cfg: _required(cfg) + ["D23", "D29"] + (["D21"] if cfg.get("s2", {}).get("stage_suffix") else [])

if __name__ == "__main__":
    EJ.main()
