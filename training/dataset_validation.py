"""Validate HASYv2 + CROHME datasets (read-only). Writes validation_report.json.

HASYv2: manifest coverage, image decodability, label consistency, distribution.
CROHME: on-disk vs archive counts, PNG decodability, SymLG parseability,
  InkML XML validity (test/val fully + train sample), cross-modal stem
  alignment on the test sets. Corrupted files are reported, never ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import xml.dom.minidom as minidom
import zipfile
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TRUTH_RE = re.compile(r'<annotation type="truth">\s*(.*?)\s*</annotation>', re.S)


def val_hasy(report: dict) -> None:
    r = report["hasyv2"] = {}
    man_path = PROCESSED / "hasyv2_manifest.csv"
    rows = list(csv.DictReader(man_path.open(encoding="utf-8")))
    r["manifest_rows"] = len(rows)
    bad, shapes, dist = [], Counter(), Counter()
    for row in rows:
        p = RAW / "hasyv2" / row["path"]
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            bad.append(row["path"])
            continue
        shapes[f"{im.shape[1]}x{im.shape[0]}"] += 1
        dist[row["class"]] += 1
    r["unreadable"] = bad
    r["shapes"] = dict(shapes)
    r["class_distribution"] = dict(sorted(dist.items()))
    # label consistency: manifest symbol_id must be in class_mapping.json
    mapping = json.loads((PROCESSED / "class_mapping.json").read_text(encoding="utf-8"))
    mism = [row["path"] for row in rows
            if mapping.get(row["symbol_id"], {}).get("class") != row["class"]]
    r["label_mismatches"] = mism
    r["ok"] = not bad and not mism
    print(f"[val:hasy] rows={len(rows)} bad={len(bad)} mismatch={len(mism)} "
          f"shapes={dict(shapes)}")


def png_ok(p: Path) -> bool:
    try:
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED) is not None
    except Exception:
        return False


def val_crohme(report: dict, quick: bool) -> None:
    c = report["crohme"] = {}
    base = RAW / "crohme" / "TC11_CROHME23"
    # 1. on-disk vs archive counts
    with zipfile.ZipFile(RAW / "CROHME23.zip") as z:
        archived = Counter(Path(n).suffix.lower() or "<dir>"
                           for n in z.namelist() if not n.endswith("/"))
    on_disk = Counter(p.suffix.lower() for p in base.rglob("*") if p.is_file())
    c["archived_counts"] = {k: archived[k] for k in (".png", ".inkml", ".lg")}
    c["ondisk_counts"] = {k: on_disk.get(k, 0) for k in (".png", ".inkml", ".lg")}
    c["extraction_gaps"] = {k: archived[k] - on_disk.get(k, 0)
                            for k in (".png", ".inkml", ".lg")}

    # 2. PNG decodability (all)
    pngs = sorted(base.rglob("*.png"))
    bad_png = [str(p.relative_to(base)) for p in pngs if not png_ok(p)]
    c["png_total"] = len(pngs)
    c["png_unreadable"] = bad_png
    print(f"[val:crohme] png={len(pngs)} bad={len(bad_png)}")

    # 3. SymLG parseability (all unless quick)
    lgs = sorted(base.rglob("*.lg"))
    if quick:
        rng = random.Random(0)
        lgs = rng.sample(lgs, min(3000, len(lgs)))
    bad_lg, no_obj = [], 0
    for lg in lgs:
        try:
            n_obj = sum(1 for ln in lg.read_text(encoding="utf-8",
                                                 errors="strict").splitlines()
                        if ln.startswith("O,"))
            if n_obj == 0:
                no_obj += 1
        except Exception:
            bad_lg.append(str(lg.relative_to(base)))
    c["lg_checked"] = len(lgs)
    c["lg_unparseable"] = bad_lg[:50]
    c["lg_n_unparseable"] = len(bad_lg)
    c["lg_zero_objects"] = no_obj
    print(f"[val:crohme] lg checked={len(lgs)} bad={len(bad_lg)} empty={no_obj}")

    # 4. InkML validity: all test/val + train sample; truth presence
    inkml_testval = [p for p in base.rglob("*.inkml")
                     if "/test/" in p.as_posix() or "/val/" in p.as_posix()]
    train_pool = [p for p in base.rglob("*.inkml") if "/train/" in p.as_posix()]
    sample = random.Random(1).sample(train_pool, min(1000, len(train_pool)))
    bad_xml, no_truth = [], 0
    for ik in inkml_testval + sample:
        try:
            txt = ik.read_text(encoding="utf-8", errors="strict")
            minidom.parseString(txt)
            if not TRUTH_RE.search(txt):
                no_truth += 1
        except Exception:
            bad_xml.append(str(ik.relative_to(base)))
    c["inkml_testval"] = len(inkml_testval)
    c["inkml_train_sample"] = len(sample)
    c["inkml_invalid_xml"] = bad_xml[:50]
    c["inkml_n_invalid"] = len(bad_xml)
    c["inkml_no_latex_truth"] = no_truth
    print(f"[val:crohme] inkml testval={len(inkml_testval)} train_sample={len(sample)} "
          f"bad_xml={len(bad_xml)} no_truth={no_truth}")

    # 5. cross-modal stem alignment on both test sets
    align = {}
    for subset in ("CROHME2019_test", "CROHME2023_test"):
        png_stems = {p.stem for p in (base / "IMG" / "test" / subset).glob("*.png")}
        lg_stems = {p.stem for p in (base / "SymLG" / "test" / subset).glob("*.lg")}
        ik_dir = base / "INKML" / "test" / subset
        ik_stems = {p.stem for p in ik_dir.glob("*.inkml")} if ik_dir.exists() else set()
        align[subset] = {
            "png": len(png_stems), "lg": len(lg_stems), "inkml": len(ik_stems),
            "png_missing_lg": sorted(png_stems - lg_stems)[:10],
            "n_png_missing_lg": len(png_stems - lg_stems),
            "png_missing_inkml": sorted(png_stems - ik_stems)[:10],
            "n_png_missing_inkml": len(png_stems - ik_stems),
        }
    c["test_alignment"] = align
    c["ok"] = (not bad_png and not bad_lg and not bad_xml
               and all(v == 0 for v in c["extraction_gaps"].values()))
    print(f"[val:crohme] alignment={json.dumps({k: {kk: vv for kk, vv in v.items() if isinstance(vv, int)} for k, v in align.items()})}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    report: dict = {"hasyv2": {}, "crohme": {}}
    val_hasy(report)
    val_crohme(report, args.quick)
    out = PROCESSED / "validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[val] ok={report['hasyv2']['ok'] and report['crohme']['ok']} -> {out}")


if __name__ == "__main__":
    main()
