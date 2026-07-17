# Wellar AI Readiness Server

FastAPI service that validates image, audio, and video evidence and produces a non-medical readiness result.

## Runtime flow

`Flutter/Mobile → Directus wellness_scans / scan_media → AI Server /process → Directus scan_results / employee_baselines / alerts`

1. Mobile or backend creates a `wellness_scans` row.
2. Directus file IDs are linked through `scan_media`.
3. An authenticated user calls `POST /process` with only `scan_id`.
4. The server verifies the Directus user, scan ownership, active membership, scan status, and required media.
5. The server marks the scan `processing`, returns HTTP `202`, and schedules same-process background analysis.
6. Background processing downloads Directus assets, validates media, computes the result, upserts one `scan_results` row, then marks the scan `completed` or `failed`.
7. A qualified completed scan may update `employee_baselines` only after the core result and scan-status writeback succeed.

`scan_results.scan_id` must have a database-level unique constraint. The in-process duplicate guard does not replace that constraint and does not provide distributed locking across multiple server replicas.

## Required environment

- `DIRECTUS_URL`
- `DIRECTUS_TOKEN`
- `DIRECTUS_TIMEOUT`
- `PORT`

## Model and validation environment

- `ML_MODEL_PATH` — defaults to `models/latest.pt`
- `REQUIRE_LOCAL_MODEL` — defaults to `false`
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

## Runtime controls

- `AI_SERVER_ENV`
- `DEBUG_SCAN_ENDPOINT_ENABLED`
- `FAST_SCAN_MODE`
- `FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS`
- `MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS`
- `AUDIO_ANALYSIS_WORKER_TIMEOUT_SECONDS`
- `AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS`
- `AUDIO_FFMPEG_CONVERSION_TIMEOUT_SECONDS`
- `LOG_LEVEL`

The local ML model is optional unless `REQUIRE_LOCAL_MODEL=true`. Phrase transcription is optional unless phrase matching is required for the active validation policy.

## Endpoints

- `GET /health`
- `GET /baseline/status?member_id=...&business_profile_id=...`
- `POST /process`
- `POST /baseline`

`GET /debug/scan/{scan_id}` is registered only when both conditions are true:

- `AI_SERVER_ENV` is `dev`, `development`, `local`, or `test`
- `DEBUG_SCAN_ENDPOINT_ENABLED=true`

It is not a production endpoint.

## Process a scan

`POST /process` requires a Directus user bearer token:

```http
Authorization: Bearer <directus-user-access-token>
Content-Type: application/json
```

Request body:

```json
{
  "scan_id": "wellness-scan-id"
}
```

The server ignores legacy request-body overrides such as client media URLs, task values, member IDs, business-profile IDs, department IDs, previous confidence, and request IDs. The canonical data source is Directus.

A newly accepted scan returns HTTP `202`:

```json
{
  "ok": true,
  "scan_id": "wellness-scan-id",
  "status": "accepted"
}
```

Idempotent responses may include:

- `already_processing` — HTTP `202`
- `already_completed` — HTTP `200`
- `already_failed` — HTTP `200`
- `recovered_completed` — HTTP `200` when a result exists but the scan status still needs recovery

The final readiness payload is persisted in `scan_results`; the mobile application reads the scan and result from Directus.

## Real deployment test

Use a real `wellness_scans` ID and a Directus access token for the user who owns the scan and has an active membership:

```powershell
.\test_process.ps1 `
  -ScanId "<real-wellness-scan-id>" `
  -AccessToken "<directus-user-access-token>" `
  -ServerUrl "https://<your-ai-server-domain>"
```

For a non-production deployment where the debug endpoint is explicitly enabled:

```powershell
.\test_process.ps1 `
  -ScanId "<real-wellness-scan-id>" `
  -AccessToken "<directus-user-access-token>" `
  -ServerUrl "https://<your-ai-server-domain>" `
  -IncludeDebugContext
```

Do not store access tokens in Git. `DIRECTUS_ACCESS_TOKEN` may be set temporarily in the local PowerShell session instead of passing `-AccessToken`.

## Failure behavior

Required missing, unreadable, timed-out, or low-quality media fail closed and do not create a scored result. Example failure data written to the scan may include:

```json
{
  "status": "failed",
  "failure_reason": "low_quality_media",
  "failure_message": "The scan quality is too low. Please try again with better lighting, a steady camera, and clear audio."
}
```

An audio decode timeout is preserved as diagnostic evidence and maps to `audio_validation_timeout`, rather than being silently reduced to an ordinary missing-audio condition.

## Baseline behavior

Default calibration thresholds are defined in `config.py`:

- Scan 1: collecting
- Scan 2: provisional
- Scan 3: active and eligible for personalized scoring when all other quality gates pass
- Scan 5: high-confidence baseline threshold
- Stored feature history is bounded by `BASELINE_MAX_STORED_SAMPLES`

Only qualified, stable, reliable scans update the baseline. Baseline mutation occurs after successful core result persistence and scan completion writeback.

## Training

Training expects a JSONL manifest at `data/manifest.jsonl`:

```bash
python -m ml.train --manifest data/manifest.jsonl --out models/latest.pt
```

The training pipeline requires all configured classes, at least two samples per class, finite 21-feature vectors, and a stratified train/validation split.

## Directus expectations

- `wellness_scans.status`: `pending`, `media_ready`, `processing`, `completed`, or `failed`
- `scan_media.scan_id` points to the target `wellness_scans` row
- `scan_media.video_file`, `scan_media.audio_file`, and `scan_media.thumbnail` contain Directus file IDs
- `scan_results.scan_id` is unique at the database level

Optional schema fields are detected and omitted when unavailable.

## Deployment SQL

Apply these migrations before production deployment:

- `sql/2026_06_05_scan_results_unique_scan_id.sql`
- `sql/2026_06_06_scan_results_ai_model_version_length.sql`
- `sql/2026_07_01_phase2_baseline_foundation.sql`

The first migration removes duplicate result rows before adding the unique `scan_results.scan_id` constraint. The second expands `scan_results.ai_model_version` for `cie_v1_2`. The baseline migration establishes the baseline foundation and must not null existing member relationships.

## Easypanel deployment

The repository Docker image is the deployment source of truth. Configure secrets in Easypanel, deploy the container, verify `GET /health`, then run one authenticated real scan and confirm:

1. `wellness_scans`: `media_ready → processing → completed` or an explicit failure state
2. exactly one `scan_results` row for the scan
3. no raw Directus response bodies or tokens in logs
4. baseline mutation only for an eligible completed scan