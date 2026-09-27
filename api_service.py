# ============================================================
#  api_service.py
#
#  Lightweight FastAPI wrapper around the existing 3-stage
#  inference pipeline (inference.py / stage1-3 modules).
#
#  Fixes vs. the raw CLI (inference.py) when used as a service:
#    1. Models are loaded ONCE at startup and cached in memory,
#       not reloaded from disk on every request (inference.py's
#       load_stage_models() does a fresh torch.load() every call —
#       fine for a one-off CLI run, expensive per HTTP request).
#    2. Batch endpoint processes multiple studies with per-study
#       fault isolation: one failing study returns an error for
#       itself but does not abort the rest of the batch.
#    3. Basic structured logging + timing per request, useful for
#       anyone operating this as a real service.
#
#  This intentionally reuses the real pipeline functions
#  (find_series_id, run_for_series_key) from inference.py rather
#  than reimplementing inference logic — the wrapper only adds
#  the service layer around them.
#
#  Run:
#      uvicorn api_service:app --host 0.0.0.0 --port 8000
#
#  Endpoints:
#      GET  /health
#      GET  /predict/{study_id}?series_key=sag_t2
#      POST /predict/batch        body: {"study_ids": [123, 456, ...]}
# ============================================================

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config_and_utils import CFG, SAGITTAL_SERIES, load_dataframes
from inference import load_stage_models, find_series_id, run_for_series_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("lumbar_api")


# ── In-memory model + data cache, populated once at startup ───
# Loading a checkpoint from disk takes real time (torch.load +
# building the nn.Module). Doing this on every HTTP request would
# add that cost to every single call. Loading once here means a
# request only pays for the actual forward pass.
STATE = {
    "models": {},     # series_key -> (s1, s2, s3) or None if unavailable
    "descs_df": None,  # series-description lookup table, loaded once
}


def _load_all_available_models():
    """
    Attempt to load Stage 1/2/3 checkpoints for every known series key.
    A series key with missing checkpoints is recorded as None so
    requests for it fail fast with a clear message, rather than the
    server crashing at startup because sag_t1 (say) was never trained.
    """
    for series_key in SAGITTAL_SERIES:
        s1, s2, s3 = load_stage_models(series_key)
        if s1 is None or s2 is None or s3 is None:
            log.warning(
                "Series '%s': one or more checkpoints missing "
                "(s1=%s s2=%s s3=%s) — this branch will be unavailable.",
                series_key, s1 is not None, s2 is not None, s3 is not None,
            )
            STATE["models"][series_key] = None
        else:
            log.info("Series '%s': all checkpoints loaded.", series_key)
            STATE["models"][series_key] = (s1, s2, s3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading models and dataframes …")
    t0 = time.time()
    _load_all_available_models()
    _, _, descs_df = load_dataframes()
    STATE["descs_df"] = descs_df
    log.info("Startup complete in %.1fs", time.time() - t0)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Lumbar Spinal Stenosis Grading API",
    description="Serves the 3-stage slice-selection -> keypoint-detection "
                "-> severity-classification pipeline.",
    version="1.0",
    lifespan=lifespan,
)


# ── Request/response models ────────────────────────────────────
class BatchRequest(BaseModel):
    study_ids: list[int] = Field(..., min_length=1, max_length=200)
    series_key: str = Field(default="sag_t2")


class StudyResult(BaseModel):
    study_id: int
    status: str                 # "ok" | "error"
    series_id: Optional[int] = None
    levels: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


# ── Core single-study logic, shared by both endpoints ──────────
def _predict_one(study_id: int, series_key: str) -> StudyResult:
    t0 = time.time()

    if series_key not in SAGITTAL_SERIES:
        return StudyResult(
            study_id=study_id, status="error",
            error=f"Unknown series_key '{series_key}'. "
                  f"Valid options: {list(SAGITTAL_SERIES.keys())}",
        )

    models = STATE["models"].get(series_key)
    if models is None:
        return StudyResult(
            study_id=study_id, status="error",
            error=f"No trained checkpoints available for '{series_key}'. "
                  f"Train it first with train.py.",
        )

    series_id = find_series_id(STATE["descs_df"], study_id, series_key)
    if series_id is None:
        return StudyResult(
            study_id=study_id, status="error",
            error=f"No '{series_key}' series found for this study_id.",
        )

    try:
        # Pass the models we already loaded at startup, so this call does
        # NOT re-read checkpoint files from disk (see the preloaded_models
        # parameter added to inference.run_for_series_key). CLI usage of
        # inference.py is unaffected — it never passes this argument, so
        # main() still behaves exactly as before.
        levels = run_for_series_key(study_id, series_id, series_key,
                                     preloaded_models=models)
        if not levels:
            return StudyResult(
                study_id=study_id, status="error",
                error="Pipeline ran but produced no per-level predictions "
                      "(check that the study has all expected levels).",
            )
        return StudyResult(
            study_id=study_id, status="ok",
            series_id=series_id, levels=levels,
            latency_ms=round((time.time() - t0) * 1000, 1),
        )
    except FileNotFoundError as e:
        # Most likely cause: this study_id isn't in the local DICOM
        # directory (e.g. a test-set study not present on this machine).
        return StudyResult(
            study_id=study_id, status="error",
            error=f"Data not found: {e}",
        )
    except Exception as e:
        # Catch-all so one bad study can never take down a batch request
        # or crash the process. Logged with full detail server-side;
        # the client just sees a clean error message.
        log.exception("Unexpected error processing study_id=%s", study_id)
        return StudyResult(
            study_id=study_id, status="error",
            error=f"Internal error: {type(e).__name__}: {e}",
        )


# ── Endpoints ───────────────────────────────────────────────────
@app.get("/health")
def health():
    available = [k for k, v in STATE["models"].items() if v is not None]
    return {
        "status": "ok",
        "device": CFG.device,
        "series_available": available,
    }


@app.get("/predict/{study_id}", response_model=StudyResult)
def predict_single(
        study_id: int,
        series_key: str = Query(default="sag_t2"),
):
    """Run the full 3-stage pipeline on a single study."""
    return _predict_one(study_id, series_key)


@app.post("/predict/batch", response_model=list[StudyResult])
def predict_batch(req: BatchRequest):
    """
    Run the pipeline over a list of study_ids. Each study is processed
    independently — one failure never aborts the rest of the batch,
    matching the fault-tolerance behavior expected of a batch/queue
    system rather than a single all-or-nothing request.
    """
    log.info("Batch request: %d studies, series_key=%s",
              len(req.study_ids), req.series_key)
    results = []
    for study_id in req.study_ids:
        result = _predict_one(study_id, req.series_key)
        results.append(result)
        if result.status == "error":
            log.warning("study_id=%s failed: %s", study_id, result.error)
    ok_count = sum(1 for r in results if r.status == "ok")
    log.info("Batch complete: %d/%d succeeded", ok_count, len(results))
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
