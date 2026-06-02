# Wellar AI Readiness Server

FastAPI service that analyzes image, audio, and video inputs and returns a non-medical readiness result.

## Flow

`Flutter/Mobile -> Directus wellness_scans / scan_media -> AI Server /process -> Directus scan_results / employee_baselines / alerts`

Expected runtime flow:

1. Mobile or backend creates a `wellness_scans` item in Directus.
2. Media assets are linked in `scan_media`.
3. Mobile calls `POST /process` with `scan_id` and optional overrides.
4. The AI server resolves scan context from Directus, downloads assets, runs quality gating and readiness scoring, updates personalized baseline, writes `scan_results`, updates `wellness_scans`, updates `business_profile_members`, and optionally creates `alerts` / `notifications`.

## Environment

- `DIRECTUS_URL`
- `DIRECTUS_TOKEN`
- `PORT`

If `models/latest.pt` exists, ML prediction is blended into readiness scoring.

## Endpoints

- `GET /health`
- `GET /debug/scan/{scan_id}`
- `GET /baseline/status?member_id=...&business_profile_id=...`
- `POST /process`
- `POST /baseline`

## Example: /process

```json
{
  "scan_id": "wellness-scan-id",
  "request_id": "scan-request-id-optional",
  "member_id": "business-profile-member-id",
  "business_profile_id": "business-profile-id",
  "department_id": "department-id",
  "previous_confidence": 0.71,
  "media": {
    "image": "directus-image-asset-id",
    "audio": "directus-audio-asset-id",
    "video": "directus-video-asset-id"
  },
  "task": {
    "reaction_time": 0.72,
    "errors": 1,
    "attempts": 5
  }
}
```

If `media` is omitted, the service will try to resolve assets from `scan_media` using `scan_id`.

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
