#!/usr/bin/env bash
# Download datasets into $TG_DATA on a LOGIN node.  bash scripts/fetch_data.sh [item ...]
# items: paco grefcoco partonomy sceneparser coco_val2017 coco_train2017 coco_train2014 (default: all)
# EgoObjects images (SceneParser) are licence-gated: place them at $TG_DATA/egoobjects/datasets/EgoObjects/images
set -euo pipefail
: "${TG_DATA:?source tg_env.sh first}"
if [ -z "${TMPDIR:-}" ] || [ ! -d "$TMPDIR" ] || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="$TG_ROOT/tmp"
fi
mkdir -p "$TMPDIR"
ITEMS=${*:-paco grefcoco partonomy sceneparser coco_val2017 coco_train2017 coco_train2014}
PACO_SHA256=02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80

hf_get() {  # repo dest files...
    python -c '
import sys
from huggingface_hub import hf_hub_download
repo, dest, *files = sys.argv[1:]
for f in files:
    print("  ->", hf_hub_download(repo, f, repo_type="dataset", local_dir=dest), flush=True)
' "$@"
}

untar_zst() {  # archive dest
    python -c '
import sys, tarfile, zstandard
src, dest = sys.argv[1:]
with open(src, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
    with tarfile.open(fileobj=r, mode="r|") as t:
        t.extractall(dest, filter="data")
' "$@"
}

coco() {  # split
    local dir="$TG_DATA/coco"
    mkdir -p "$dir"
    if [ -f "$dir/$1.done" ]; then echo "  coco $1 already present"; return; fi
    curl -fL --retry 4 -C - -o "$dir/$1.zip" "http://images.cocodataset.org/zips/$1.zip"
    echo "  unzipping $1 (no output for a few minutes)"
    unzip -q -o "$dir/$1.zip" -d "$dir" && rm "$dir/$1.zip" && touch "$dir/$1.done"
    echo "  $1: $(ls "$dir/$1" | wc -l) images"
}

for item in $ITEMS; do
    echo "== $item"
    case $item in
        paco)
            d="$TG_DATA/paco/annotations"; mkdir -p "$d"
            if [ ! -f "$d/paco_lvis_v1_val.json" ]; then
                curl -fL --retry 4 -o "$d/paco_lvis_v1.zip" https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip
                echo "$PACO_SHA256  $d/paco_lvis_v1.zip" | sha256sum -c
                unzip -q -o -j "$d/paco_lvis_v1.zip" -d "$d" && rm "$d/paco_lvis_v1.zip"
            fi
            ls "$d";;
        grefcoco) hf_get FudanCVL/gRefCOCO "$TG_DATA/grefcoco" "grefs(unc).json" instances.json;;
        partonomy)
            d="$TG_DATA/partonomy-core"
            hf_get partonomy/partonomy-core "$d" test-qa_pairs.json images.tar.zst
            [ -d "$d/images" ] || { echo "  extracting images"; untar_zst "$d/images.tar.zst" "$d"; }
            echo "  partonomy: $(ls "$d/images" | wc -l) images";;
        sceneparser)
            hf_get SceneParser/SceneParser-bench "$TG_DATA/sceneparser" train.jsonl val.jsonl
            [ -d "$TG_DATA/egoobjects/datasets/EgoObjects/images" ] || \
                echo "  NOTE: EgoObjects images missing at $TG_DATA/egoobjects/datasets/EgoObjects/images; SceneParser will be reported 'missing'";;
        coco_val2017) coco val2017;;
        coco_train2017) coco train2017;;
        coco_train2014) coco train2014;;
        *) echo "unknown item $item" >&2; exit 1;;
    esac
done
echo "fetch_data done"
