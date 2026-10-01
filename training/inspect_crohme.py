"""Inspect the extracted CROHME23 dataset and write data/processed/crohme_info.json.

Read-only: never modifies data/raw/crohme/. Counts files per split/modality,
invents the symbol vocabulary from SymLG ground truth, samples InkML truth
(LaTeX) annotations and image shapes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROHME = ROOT / "data" / "raw" / "crohme" / "TC11_CROHME23"
OUT = ROOT / "data" / "processed" / "crohme_info.json"

TRUTH_RE = re.compile(r'<annotation type="truth">\s*(.*?)\s*</annotation>', re.S)


def parse_lg_objects(path: Path) -> list[str]:
    labels = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("O,"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                labels.append(parts[2])
    return labels


def main() -> None:
    info: dict = {"version": "CROHME2023 (TC11, ICDAR 2023)",
                  "modalities": {}, "splits": {}, "counts": {}}
    total = {"png": 0, "inkml": 0, "lg": 0}
    for mod, ext in (("IMG", ".png"), ("INKML", ".inkml"), ("SymLG", ".lg")):
        mdir = CROHME / mod
        files = sorted(mdir.rglob(f"*{ext}")) if mdir.exists() else []
        per_split: dict[str, int] = Counter()
        per_sub: dict[str, int] = Counter()
        for f in files:
            rel = f.relative_to(mdir).parts
            per_split[rel[0]] += 1
            per_sub["/".join(rel[:2])] += 1
        info["modalities"][mod] = {"files": len(files), "by_split": dict(per_split),
                                   "by_subset": dict(per_sub)}
        total[ext.lstrip(".")] = len(files)
    info["counts"] = total

    # expression IDs: IMG train file stems (offline renders, our use case)
    train_imgs = sorted((CROHME / "IMG" / "train").rglob("*.png"))
    info["splits"] = {
        "train_expressions_with_png": len(train_imgs),
        "expression_id": "PNG/InkML/SymLG share the same file stem "
                         "(e.g. 001-equation000) within aligned subsets",
    }

    # symbol vocabulary from a SymLG sample (train, capped for speed)
    symlg_train = sorted((CROHME / "SymLG" / "train").rglob("*.lg"))[:3000]
    vocab: Counter = Counter()
    n_objects = 0
    for lg in symlg_train:
        labs = parse_lg_objects(lg)
        vocab.update(labs)
        n_objects += len(labs)
    info["symbol_annotation"] = {
        "format": ".lg label graph: 'O,<id>,<label>,<w>,...' objects + "
                  "'R,<from>,<to>,<relation>,<w>' spatial relations "
                  "(Right, Sup, Sub, Inside, ...)",
        "lg_files_scanned": len(symlg_train),
        "objects_scanned": n_objects,
        "distinct_labels_scanned": len(vocab),
        "most_common_labels": vocab.most_common(25),
    }

    # InkML truth (LaTeX) availability
    inkml_train = sorted((CROHME / "INKML" / "train").rglob("*.inkml"))[:500]
    with_truth = 0
    examples: list[str] = []
    for ik in inkml_train:
        txt = ik.read_text(encoding="utf-8", errors="replace")
        m = TRUTH_RE.search(txt)
        if m:
            with_truth += 1
            if len(examples) < 5:
                examples.append(m.group(1))
    info["annotations"] = {
        "inkml": "strokes as <trace> X Y sequences + <annotation type='truth'> "
                 "LaTeX + Content-MathML + writer/UI metadata",
        "inkml_scanned": len(inkml_train),
        "inkml_with_latex_truth": with_truth,
        "latex_examples": examples,
        "image_format": "PNG renders (scanned paper or InkML renderings)",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[crohme] {json.dumps(info['counts'])}")
    print(f"[crohme] distinct symbol labels (sample): {len(vocab)}")
    print(f"[crohme] info -> {OUT}")


if __name__ == "__main__":
    main()
