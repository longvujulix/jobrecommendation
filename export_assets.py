"""Export lightweight assets and sample files for GUI Project 2 (ITViec Hub).

Usage:
    python export_assets.py
    python export_assets.py --source /path/to/Project2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / "Project 2"


def sha256_of(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export(source_dir: Path) -> None:
    print(f"[EXPORT] Đang trích xuất assets từ nguồn: {source_dir}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {source_dir}")

    assets_dir = ROOT / "assets"
    models_dir = assets_dir / "models"
    images_dir = assets_dir / "images"
    samples_dir = ROOT / "samples"

    models_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}

    # 1. Copy trained models
    src_models = source_dir / "outputs" / "models"
    for name in ["recommender.joblib", "classifier.joblib"]:
        src_file = src_models / name
        dst_file = models_dir / name
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            manifest[f"models/{name}"] = sha256_of(dst_file)
            print(f"  ✓ Đã copy mô hình {name} ({dst_file.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  ⚠ Không tìm thấy {src_file}")

    # 2. Copy JSON outputs
    src_out = source_dir / "outputs"
    for name in ["classification.json", "recommendation.json", "eda.json"]:
        src_file = src_out / name
        dst_file = assets_dir / name
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            manifest[name] = sha256_of(dst_file)
            print(f"  ✓ Đã copy file số liệu {name}")
        else:
            print(f"  ⚠ Không tìm thấy {src_file}")

    # 3. Copy image plots
    if src_out.exists():
        for p in src_out.glob("*.png"):
            dst_file = images_dir / p.name
            shutil.copy2(p, dst_file)
            manifest[f"images/{p.name}"] = sha256_of(dst_file)
            print(f"  ✓ Đã copy biểu đồ {p.name}")

    # 4. Export companies parquet
    comp_excel = source_dir / "data" / "Overview_Companies.xlsx"
    if comp_excel.exists():
        df_comp = pd.read_excel(comp_excel).drop_duplicates("id").sort_values("id").reset_index(drop=True)
        parquet_dst = assets_dir / "companies.parquet"
        df_comp.to_parquet(parquet_dst, index=False)
        manifest["companies.parquet"] = sha256_of(parquet_dst)
        print(f"  ✓ Đã lưu companies.parquet ({len(df_comp)} công ty, {parquet_dst.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  ⚠ Không tìm thấy {comp_excel}")

    # 5. Export reviews parquet and sample review sets
    rev_excel = source_dir / "data" / "Reviews.xlsx"
    if rev_excel.exists():
        df_rev = pd.read_excel(rev_excel)
        parquet_rev_dst = assets_dir / "reviews.parquet"
        df_rev.to_parquet(parquet_rev_dst, index=False)
        manifest["reviews.parquet"] = sha256_of(parquet_rev_dst)
        print(f"  ✓ Đã lưu reviews.parquet ({len(df_rev)} reviews, {parquet_rev_dst.stat().st_size / 1024:.1f} KB)")

        sample_cols = [
            "id", "Company Name", "Title", "What I liked", "Suggestions for improvement",
            "Rating", "Salary & benefits", "Training & learning", "Management cares about me",
            "Culture & fun", "Office & workspace", "Recommend?"
        ]
        sample_with = df_rev.sample(n=25, random_state=42)[sample_cols]
        sample_with_dst = samples_dir / "mau_review_co_nhan.csv"
        sample_with.to_csv(sample_with_dst, index=False, encoding="utf-8-sig")

        sample_without = sample_with.drop(columns=["Recommend?"])
        sample_without_dst = samples_dir / "mau_review_khong_nhan.csv"
        sample_without.to_csv(sample_without_dst, index=False, encoding="utf-8-sig")
        print("  ✓ Đã xuất các file review mẫu trong samples/")

    # Save manifest
    (assets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[EXPORT HOÀN TẤT] Tổng cộng {len(manifest)} artifacts đã được đóng gói độc lập.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export assets for GUI Project 2")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Đường dẫn tới thư mục Project 2")
    args = parser.parse_args()
    export(args.source)
