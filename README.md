# Wellar AI Readiness Server

FastAPI service that analyzes image, audio, and video inputs and returns a non-medical readiness result.

## Flow

`Flutter/Mobile -> Directus wellness_scans / scan_media -> AI Server /process -> Directus scan_results / employee_baselines / alerts`

Expected runtime flow:

1. Mobile or backend creates a `wellness_scans` item in Directus.
2. Media assets are linked in `scan_media`.
3. Mobile calls `POST /process` with only `scan_id`.
4. The AI server reads `wellness_scans` by `scan_id`, accepts only `media_ready`, marks the scan `processing`, returns `202`, then continues in the background.
5. Background processing resolves `scan_media` from Directus, downloads Directus assets by file ID, validates media quality, runs scoring, writes exactly one `scan_results` row, and marks `wellness_scans` as `completed` or `failed`.

## Environment

- `DIRECTUS_URL`
- `DIRECTUS_TOKEN`
- `DIRECTUS_TIMEOUT`
- `PORT`
- `ML_MODEL_PATH`
- `MAX_DOWNLOAD_BYTES`
- `REQUIRE_VIDEO`
- `REQUIRE_AUDIO`
- `REQUIRE_FACE`
- `REQUIRE_IMAGE`
- `REQUIRE_PHRASE_MATCH`
- `PHRASE_MATCH_THRESHOLD`
- `MIN_VIDEO_SECONDS`
- `MIN_AUDIO_SECONDS`
- `MIN_FACE_VISIBLE_RATIO`
- `MIN_VIDEO_QUALITY`
- `MIN_AUDIO_QUALITY`
- `MIN_IMAGE_QUALITY`

`ML_MODEL_PATH` defaults to `models/latest.pt`. If the configured model file is present and loadable, ML prediction is blended into readiness scoring. If the model or transcription stack is unavailable, scans fail cleanly instead of crashing the server.

## Endpoints

- `GET /health`
- `GET /debug/scan/{scan_id}`
- `GET /baseline/status?member_id=...&business_profile_id=...`
- `POST /process`
- `POST /baseline`

## Example: /process

```json
{
  "scan_id": "wellness-scan-id"
}
```

The backend ignores old request-body overrides such as `media`, `task`, `previous_confidence`, `member_id`, `business_profile_id`, `department_id`, and client media URLs. `/process` trusts Directus `wellness_scans` and `scan_media` only.

## First real test

Run:

```powershell
.\test_process.ps1 -ScanId "<real-wellness-scan-id>" -ServerUrl "http://127.0.0.1:8000"
```

This calls `GET /debug/scan/{scan_id}` first, then `POST /process`.

## Example success response

```json
{
  "status": "completed",
  "retake_required": false,
  "readiness_score": 78,
  "risk_level": "stable",
  "confidence": 0.82,
  "camera_confidence": 0.81,
  "voice_confidence": 0.77,
  "task_performance_score": 87,
  "baseline_used": true,
  "confidence_drift": 0.04,
  "explanation": "Signals are within the expected readiness range.",
  "suggested_action": "No action needed.",
  "ai_model_version": "1.2.0"
}
```

## Example quality failure response

```json
{
  "status": "failed",
  "retake_required": true,
  "failure_reason": "low_quality_visual",
  "explanation": "Scan quality was too weak for a reliable readiness result.",
  "suggested_action": "Please retake the scan in better lighting with clear face, voice, and reaction input.",
  "ai_model_version": "1.2.0"
}
```

## Baseline behavior

- First 2 valid scans create a provisional baseline.
- Personalized baseline becomes active after 7 valid scans.
- From scan 8 onward, readiness scoring can compare current signals against the employee's own baseline.
- Baseline data is stored in Directus `employee_baselines`.

## Training

Training still expects a JSONL manifest at `data/manifest.jsonl`.

```bash
python -m ml.train --manifest data/manifest.jsonl --out models/latest.pt
```

## Directus expectations

- `wellness_scans.status` must use only: `pending`, `media_ready`, `processing`, `completed`, `failed`
- `scan_media.scan_id` must point to the target `wellness_scans` row
- `scan_media.video_file`, `scan_media.audio_file`, and `scan_media.thumbnail` should store Directus file IDs, not client URLs
- `scan_results.scan_id` must be unique at the database level

Optional Directus fields are supported when present and skipped when absent:

- `wellness_scans.failure_message`
- `wellness_scans.user_message`
- `wellness_scans.expected_phrase`
- `wellness_scans.processing_attempts`
- `wellness_scans.processing_started_at`
- `wellness_scans.ai_model_version`
- `scan_results.internal_analysis`
- `scan_results.analysis_metadata`
- `scan_results.spoken_transcript`
- `scan_results.expected_phrase`
- `scan_results.phrase_match_score`
- `scan_results.validation_warnings`
- `scan_results.audio_quality_score`
- `scan_results.video_quality_score`
- `scan_results.image_quality_score`

## Deployment SQL

Apply [2026_06_05_scan_results_unique_scan_id.sql](</d:/flutter/last/ai-server/sql/2026_06_05_scan_results_unique_scan_id.sql>) before deployment. It:

- removes duplicate `scan_results` rows, keeping the newest row per `scan_id`
- adds the `scan_results_scan_id_unique` unique constraint on `scan_results.scan_id`

If you are applying this through Directus:

1. Open the Directus project connected to the deployment database.
2. Open the SQL runner or run the SQL directly against Postgres.
3. Execute `sql/2026_06_05_scan_results_unique_scan_id.sql`.
4. Verify `scan_results` no longer has duplicate `scan_id` values.
5. Verify the `scan_results_scan_id_unique` constraint exists.

Apply [2026_06_06_scan_results_ai_model_version_length.sql](</d:/flutter/last/ai-server/sql/2026_06_06_scan_results_ai_model_version_length.sql>) before deployment. It:

- updates `scan_results.ai_model_version` to `varchar(100)` so it can store `Conntinuity Intelligence Engine v1.2`

## scan_requests / notifications

Previous backend behavior updated `scan_requests` on successful AI completion for `manager_request` and `bulk_request` scans. That path has been restored after successful result writeback so manager/bulk request records can still close without relying on Directus Flow.

Notifications were not restored. In the old code they were already effectively disabled because no notification target user was resolved. The current mobile result flow in this repository reads scan state from `wellness_scans` and `scan_results`; there is no evidence here that mobile depends on notification creation. Future notification delivery should be handled by a dedicated backend or business-event flow after result persistence, not inline inside `/process`.
