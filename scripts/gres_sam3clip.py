import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gres_eval as G  # noqa: E402

# D33 for SAM 3 text: gRefCOCO expressions longer than SAM 3's 32-token CLIP text encoder keep their leading words
# (as D27, grefcoco_style_eval.fit_text); results live under their own stage 'sam3_text_clip'.

METHOD, STAGE = "sam3_text", "sam3_text_clip"
_stage, _Baseline = G.stage_name, G.Baseline


class Baseline(_Baseline):
    def __init__(self, cfg: dict, method: str):
        super().__init__(cfg, method)
        import b_baselines as B
        tok = self.pred.proc.tokenizer
        limit = min(tok.model_max_length, 32)
        inner = self.pred

        def clip(p: str) -> str:
            words = p.split()
            while len(words) > 1 and len(tok(B.concept(" ".join(words))).input_ids) > limit:
                words.pop()
            return " ".join(words)

        self.pred = lambda image, phrases: inner(image, [clip(p) for p in phrases])


def identity(cfg: dict, method: str, split: str) -> dict:
    ident = _identity(cfg, method, split)
    ident["clip"] = (ROOT / "scripts" / "gres_sam3clip.py").read_text()
    return ident


_identity = G.identity
G.Baseline, G.identity = Baseline, identity
G.stage_name = lambda cfg, method: STAGE if method == METHOD else _stage(cfg, method)

if __name__ == "__main__":
    if f"--method={METHOD}" not in sys.argv and METHOD not in sys.argv:
        raise SystemExit(f"gres_sam3clip.py is only for --method {METHOD}")
    G.main()
