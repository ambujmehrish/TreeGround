import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gres_sam3clip  # noqa: E402,F401  (stage name 'sam3_text_clip' for SAM 3 text, as on gRefCOCO)
import gseval_eval as GS  # noqa: E402

# D45 for SAM 3 text: GSEval expressions longer than SAM 3's 32-token CLIP text encoder keep their leading words
# (the D27 / gRefCOCO rule of gres_sam3clip); results live under the stage 'sam3_text_clip'.

METHOD = gres_sam3clip.METHOD
_Baseline, _identity = GS.Baseline, GS.identity


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


def identity(cfg: dict, method: str) -> dict:
    ident = _identity(cfg, method)
    ident["clip"] = (ROOT / "scripts" / "gseval_sam3clip.py").read_text()
    return ident


GS.Baseline, GS.identity = Baseline, identity

if __name__ == "__main__":
    if f"--method={METHOD}" not in sys.argv and METHOD not in sys.argv:
        raise SystemExit(f"gseval_sam3clip.py is only for --method {METHOD}")
    GS.main()
