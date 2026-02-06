# AI Scan Server

FastAPI service that analyzes image, audio, and video inputs and returns a scored result.

## Endpoints

- `GET /health` -> service status and model version
- `POST /process` -> analyze media and return a result
- `POST /baseline` -> update baseline for a subject (optional)

## Training (ML)

Training expects a JSONL manifest at `data/manifest.jsonl`. Each line is one sample:

```json
{
  "scan_id": "scan-001",
  "subject_id": "user-123",
  "label": "Stable",
  "media": {
    "image": "C:/data/face.jpg",
    "audio": "C:/data/voice.wav",
    "video": "C:/data/clip.mp4"
  },
  "task": {
    "reaction_time": 0.7,
    "errors": 1
  }
}
```

Train and save a model:

```bash
python -m ml.train --manifest data/manifest.jsonl --out models/latest.pt
```

If `models/latest.pt` exists, `/process` will automatically blend ML output with heuristics.

## Example: /process

```json
{
  "scan_id": "scan-001",
  "subject_id": "user-123",
  "media": {
    "image": "C:/data/face.jpg",
    "audio": "C:/data/voice.wav",
    "video": "C:/data/clip.mp4"
  },
  "task": {
    "reaction_time": 0.7,
    "errors": 1
  },
  "previous_confidence": 0.62
}
```

## Example: /baseline

```json
{
  "subject_id": "user-123",
  "media": {
    "image": "C:/data/face.jpg",
    "audio": "C:/data/voice.wav",
    "video": "C:/data/clip.mp4"
  }
}
```
