import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground.run import ProtocolError, load_config, read_json, require_approved, resolve_run, start_run, write_json  # noqa: E402

# Deviation D28: blinded pairwise human study of region-grounded edits. A fixed-seed uniform sample per edit set (no
# selection by quality) is exported with the source image, the instruction and the edited images of each compared
# condition as anonymous JPEGs; trials.json (what raters see: image ids and a random left/right order) is kept
# apart from key.json (which condition each image id is).


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def save_jpg(arr_or_img, path: Path, long_side: int, quality: int) -> None:
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    im = im.convert("RGB")
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(path, "JPEG", quality=quality)


def anon(name: str, sid: str, cond: str) -> str:
    return hashlib.sha256(f"{name}:{sid}:{cond}".encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    require_approved(cfg, ["D23", "D28"])
    st = cfg["study"]
    E = _load("e_edit")
    stage = _load("s2_train").s2_stage(cfg)
    editor, name = st["editor"], st["name"]
    conds = sorted({c for pair in st["pairs"] for c in pair})
    out = ROOT / "results" / "user_study" / name
    (out / "img").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(st["seed"])
    trials, key, sources = [], {}, {}
    for ds, n in st["datasets"].items():
        items = {s.sample_id: s for s in E.evaluated(E.samples(cfg, ds), False, 0)}
        mrun = resolve_run(cfg, f"e_{editor}_{ds}_{stage}_merge", cfg["inputs"].get("edit_merge", "latest"), False)
        dirs = read_json(mrun / "metrics.json")["images_dir"]
        new = set(cfg["edit_s2"]["conditions"][editor])
        ids = sorted(items)
        pick = sorted(ids[i] for i in rng.choice(len(ids), size=min(n, len(ids)), replace=False))
        for sid in pick:
            s = items[sid]
            image, _ = s.load()
            src = anon(name, sid, "source")
            save_jpg(image, out / "img" / f"{src}.jpg", st["long_side"], st["quality"])
            ims = {}
            for c in conds:
                f = Path(dirs["d23" if c in new else "d14"]) / E.safe_name(sid) / f"{c}.png"
                if not f.exists():
                    raise ProtocolError(f"missing edit {f}")
                ims[c] = anon(name, sid, c)
                with Image.open(f) as im:
                    save_jpg(im, out / "img" / f"{ims[c]}.jpg", st["long_side"], st["quality"])
                key[ims[c]] = {"dataset": ds, "sample_id": sid, "condition": c}
            key[src] = {"dataset": ds, "sample_id": sid, "condition": "source"}
            sources[sid] = {"dataset": ds, "instruction": s.instruction, "source": src}
            for a, b in st["pairs"]:
                left, right = (a, b) if rng.random() < 0.5 else (b, a)
                trials.append({"sample": sid, "dataset": ds, "instruction": s.instruction, "source": src,
                               "left": ims[left], "right": ims[right]})
    order = rng.permutation(len(trials))
    trials = [dict(trials[i], id=f"t{k:04d}") for k, i in enumerate(order)]
    write_json(out / "trials.json", {"name": name, "question": st["question"], "options": st["options"],
                                     "trials": trials})
    write_json(out / "key.json", {"name": name, "editor": editor, "s2_stage": stage, "pairs": st["pairs"],
                                  "seed": st["seed"], "images": key})
    run = start_run(f"e_study_export_{name}", cfg, False, {"out": str(out)})
    write_json(run / "metrics.json", {"deviation": "D28", "trials": len(trials), "samples": len(sources),
                                      "images": len(key), "out": str(out)})
    print(f"{len(trials)} trials, {len(sources)} samples -> {out}")


if __name__ == "__main__":
    main()
