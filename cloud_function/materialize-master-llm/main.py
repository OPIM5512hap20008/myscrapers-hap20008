# main.py
# Build a single, ever-growing CSV from all structured JSONL files (LLM outputs).
# Reads:  gs://<bucket>/<STRUCTURED_PREFIX>/run_id=*/jsonl_llm/*.jsonl
# Writes: gs://<bucket>/<STRUCTURED_PREFIX>/datasets/listings_master_llm.csv (atomic publish)

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME        = os.getenv("GCS_BUCKET")                      # REQUIRED
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured") # e.g., "structured"

storage_client = storage.Client()

# Accept BOTH runID formats
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

# -------------------- CSV SCHEMA --------------------
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",

    "price", "year", "make", "model", "mileage",
    "source_txt",

    # Derived / rule-based
    "is_old_car",
    "high_mileage_flag",

    "transmission",
    "fuel",

    # LLM extracted location + attributes
    "color",
    "city",
    "state",
    "zip_code",

    # LLM metadata (traceability)
    "llm_provider",
    "llm_model",
    "llm_ts"
]

# -------------------- HELPERS --------------------
def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass

    run_ids = []
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                run_ids.append(rid)

    return sorted(run_ids)


def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    """Yield dict records from .jsonl under .../run_id=<run_id>/jsonl_llm/"""
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/"

    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue

        try:
            data = blob.download_as_text()
        except Exception:
            continue

        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                rec.setdefault("run_id", run_id)
                yield rec
            except Exception:
                continue


def _run_id_to_dt(rid: str) -> datetime:
    try:
        if RUN_ID_ISO_RE.match(rid):
            return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if RUN_ID_PLAIN_RE.match(rid):
            return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _open_gcs_text_writer(bucket: str, key: str):
    b = storage_client.bucket(bucket)
    blob = b.blob(key)
    return blob.open("w")


def _derive_flags(row: Dict) -> Dict:
    """Compute derived features if missing."""
    current_year = datetime.now().year

    year = row.get("year")
    mileage = row.get("mileage")

    # is_old_car
    try:
        if year is not None:
            row["is_old_car"] = (current_year - int(year)) >= 10
        else:
            row["is_old_car"] = None
    except Exception:
        row["is_old_car"] = None

    # high_mileage_flag
    try:
        if mileage is not None:
            row["high_mileage_flag"] = int(mileage) >= 120000
        else:
            row["high_mileage_flag"] = None
    except Exception:
        row["high_mileage_flag"] = None

    return row


def _write_csv(records: Iterable[Dict], dest_key: str) -> int:
    n = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        w = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()

        for rec in records:
            row = {c: rec.get(c, None) for c in CSV_COLUMNS}

            # Ensure numeric fields are properly cast where possible
            for field in ["year", "mileage"]:
                if row.get(field) is not None:
                    try:
                        row[field] = int(row[field])
                    except Exception:
                        pass

            row = _derive_flags(row)

            w.writerow(row)
            n += 1

    return n


# -------------------- MAIN MATERIALIZE --------------------
def materialize_http(request: Request):
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)

        if not run_ids:
            return jsonify({
                "ok": False,
                "error": f"no runs found under {STRUCTURED_PREFIX}/"
            }), 200

        latest_by_post: Dict[str, Dict] = {}

        for rid in run_ids:
            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid):
                pid = rec.get("post_id")
                if not pid:
                    continue

                prev = latest_by_post.get(pid)

                rec_dt = _run_id_to_dt(rec.get("run_id", rid))
                prev_dt = _run_id_to_dt(prev.get("run_id")) if prev else None

                if prev is None or rec_dt > prev_dt:
                    latest_by_post[pid] = rec

        # Write output
        base = f"{STRUCTURED_PREFIX}/datasets"
        final_key = f"{base}/listings_master_llm.csv"

        rows_written = _write_csv(latest_by_post.values(), final_key)

        return jsonify({
            "ok": True,
            "runs_scanned": len(run_ids),
            "unique_listings": len(latest_by_post),
            "rows_written": rows_written,
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"{type(e).__name__}: {e}"
        }), 500
