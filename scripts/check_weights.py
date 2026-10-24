import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.run import load_config, start_run, write_json  # noqa: E402

# Loads every pinned model exactly as the pipeline does (same classes, revisions, dtypes) with the loading report,
# and fails if any weight was missing from the checkpoint, had the wrong shape, or stayed on the meta device.


def report(name: str, model, info: dict) -> dict:
    meta = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.device.type == "meta"]
    out = {k: sorted(map(str, info.get(k, []))) for k in ("missing_keys", "mismatched_keys", "unexpected_keys")}
    out["meta_tensors"] = meta
    out["parameters"] = int(sum(p.numel() for p in model.parameters()))
    out["ok"] = not (out["missing_keys"] or out["mismatched_keys"] or meta)
    print(f"{name}: {'OK' if out['ok'] else 'PROBLEM'} params {out['parameters']:,} missing {len(out['missing_keys'])} "
          f"mismatched {len(out['mismatched_keys'])} unexpected {len(out['unexpected_keys'])} meta {len(meta)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    from transformers import Qwen3VLForConditionalGeneration, Sam3TrackerModel
    m, c = cfg["models"], cfg.get("hf_home")
    checks = {}
    for name, cls, key, dtype in (("qwen3vl", Qwen3VLForConditionalGeneration, "qwen", cfg["features"]["dtype"]),
                                  ("sam3_tracker", Sam3TrackerModel, "sam3", cfg["proposals"]["dtype"])):
        model, info = cls.from_pretrained(m[key]["repo"], revision=m[key]["revision"], dtype=getattr(torch, dtype),
                                          cache_dir=c, output_loading_info=True)
        checks[name] = report(name, model, info)
        del model
    run = start_run("check_weights", cfg, False, {})
    write_json(run / "metrics.json", {"models": checks, "passed": all(v["ok"] for v in checks.values())})
    print(run)
    sys.exit(0 if all(v["ok"] for v in checks.values()) else 1)


if __name__ == "__main__":
    main()
