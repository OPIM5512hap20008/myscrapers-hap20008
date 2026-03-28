# main.py
# Purpose: LLM extractor that reads per-listing JSONL records,
# fetches original TXT, uses Vertex AI (Gemini) to extract structured fields,
# and writes "<post_id>_llm.jsonl" to GCS.

import os
import re
import json
import logging
import traceback
from datetime import datetime, timezone

from flask import Request, jsonify
from google.cloud import storage

# Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# -------------------- ENV --------------------
PROJECT_ID  = os.getenv("PROJECT_ID", "")
REGION      = os.getenv("REGION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET", "")

STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Controls whether existing outputs are overwritten
OVERWRITE = os.getenv("OVERWRITE", "false").lower() == "true"

storage_client = storage.Client()
_MODEL = None

RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

# -------------------- MODEL --------------------
def get_model():
    global _MODEL
    if _MODEL is None:
        if not PROJECT_ID:
            raise RuntimeError("Missing PROJECT_ID")

        vertexai.init(project=PROJECT_ID, location=REGION)
        _MODEL = GenerativeModel(LLM_MODEL)
        logging.info(f"Initialized Vertex model: {LLM_MODEL}")
    return _MODEL


# -------------------- STORAGE HELPERS --------------------
def list_run_ids():
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=f"{STRUCTURED_PREFIX}/", delimiter="/")

    for _ in blobs:
        pass

    runs = []
    for prefix in getattr(blobs, "prefixes", []):
        tail = prefix.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            run_id = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(run_id) or RUN_ID_PLAIN_RE.match(run_id):
                runs.append(run_id)

    return sorted(runs)


def list_inputs(run_id):
    prefix = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/"
    bucket = storage_client.bucket(BUCKET_NAME)

    return [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".jsonl")]


def download_text(blob_name):
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(blob_name).download_as_text()


def upload_jsonl(blob_name, record):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")


def blob_exists(blob_name):
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(blob_name).exists()


# -------------------- LLM EXTRACTION --------------------
def extract_fields(text: str) -> dict:
    model = get_model()

    schema = {
        "type": "object",
        "properties": {
            "price": {"type": "integer", "nullable": True},
            "year": {"type": "integer", "nullable": True},
            "make": {"type": "string", "nullable": True},
            "model": {"type": "string", "nullable": True},
            "mileage": {"type": "integer", "nullable": True},
            "transmission": {"type": "string", "nullable": True},
            "fuel": {"type": "string", "nullable": True},

            # NEW FIELDS
            "color": {"type": "string", "nullable": True},
            "city": {"type": "string", "nullable": True},
            "state": {"type": "string", "nullable": True},
            "zip_code": {"type": "string", "nullable": True}
        }
    }

    prompt = f"""
You are an information extraction system.

Extract structured fields from the vehicle listing text below.

STRICT RULES:
- Return ONLY valid JSON
- Do NOT infer or guess missing values
- Use null if a field is not explicitly present
- Do NOT hallucinate location information
- Do NOT add explanations or extra text

FIELDS:
- price (USD integer)
- year (integer)
- make (string)
- model (string)
- mileage (integer)
- transmission (automatic/manual)
- fuel (gas/diesel/electric/hybrid)
- color (vehicle color if explicitly stated)
- city (listing location city if explicitly stated)
- state (state abbreviation or full name if explicitly stated)
- zip_code (if explicitly stated)

TEXT:
{text}
"""

    config = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        response_mime_type="application/json",
        response_schema=schema,
    )

    response = model.generate_content(prompt, generation_config=config)

    try:
        data = json.loads(response.text)
    except Exception:
        logging.error(f"Failed to parse LLM response: {response.text}")
        raise

    # Normalize numeric fields
    def safe_int(x):
        try:
            return int(str(x).replace(",", "").strip()) if x is not None else None
        except:
            return None

    data["price"] = safe_int(data.get("price"))
    data["year"] = safe_int(data.get("year"))
    data["mileage"] = safe_int(data.get("mileage"))

    # Normalize strings
    def norm(x):
        return str(x).strip() if x else None

    data["make"] = norm(data.get("make"))
    data["model"] = norm(data.get("model"))

    return data


# -------------------- ENTRYPOINT --------------------
def llm_extract_http(request: Request):
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME or not PROJECT_ID:
        return jsonify({"ok": False, "error": "missing env vars"}), 500

    body = request.get_json(silent=True) or {}
    run_id = body.get("run_id")

    if not run_id:
        runs = list_run_ids()
        if not runs:
            return jsonify({"ok": False, "error": "no runs found"}), 200
        run_id = runs[-1]

    logging.info(f"Using run_id: {run_id}")

    inputs = list_inputs(run_id)

    processed = written = skipped = errors = 0

    for in_key in inputs:
        processed += 1

        try:
            raw_line = download_text(in_key).strip()
            base = json.loads(raw_line)

            post_id = base["post_id"]
            source_txt = base["source_txt"]

            out_key = in_key.rsplit("/", 2)[0] + f"/jsonl_llm/{post_id}_llm.jsonl"

            # Overwrite control
            if blob_exists(out_key) and not OVERWRITE:
                skipped += 1
                continue

            raw_text = download_text(source_txt)

            parsed = extract_fields(raw_text)

            logging.info(f"LLM output for {post_id}: {parsed}")

            record = {
                "post_id": post_id,
                "run_id": base.get("run_id", run_id),
                "scraped_at": base.get("scraped_at"),
                "source_txt": source_txt,

                "price": parsed.get("price"),
                "year": parsed.get("year"),
                "make": parsed.get("make"),
                "model": parsed.get("model"),
                "mileage": parsed.get("mileage"),
                "transmission": parsed.get("transmission"),
                "fuel": parsed.get("fuel"),

                # NEW FIELDS
                "color": parsed.get("color"),
                "city": parsed.get("city"),
                "state": parsed.get("state"),
                "zip_code": parsed.get("zip_code"),

                "llm_model": LLM_MODEL,
                "llm_ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            upload_jsonl(out_key, record)
            written += 1

        except Exception as e:
            errors += 1
            logging.error(f"Error processing {in_key}: {e}")
            logging.error(traceback.format_exc())

    return jsonify({
        "ok": True,
        "run_id": run_id,
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "errors": errors
    }), 200
