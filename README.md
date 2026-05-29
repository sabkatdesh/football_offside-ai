# OFFSIDE AI ⚽

> Football offside detection using YOLOv8, ByteTrack and FastAPI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is this?

**Offside AI** is a web application that automatically detects offside situations in football video clips. Upload a short match clip — the system detects all players, the ball, and the referee, applies offside logic based on their real-world pitch positions, and returns a clear **OFFSIDE / ONSIDE verdict** along with a fully annotated output video and an involvement log.

No sign-in required. Single-page upload-and-view flow.

---

## Demo

| Upload | Processing | Result |
|---|---|---|
| Drop your football clip | Pipeline runs detection + offside logic | Verdict + annotated video + log |

---

## How It Works

```
Upload Video
     │
     ▼
┌─────────────┐
│  Validator  │  ← Quick pre-flight: is this actually a football video?
└─────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│                  Main Pipeline                   │
│                                                  │
│  1. Pitch Detection  → finds pitch keypoints     │
│  2. Homography       → maps frame → pitch coords │
│  3. Player Detection → detects all players       │
│  4. Team Classifier  → assigns players to teams  │
│  5. ByteTrack        → tracks players across     │
│                         frames with tracker IDs  │
│  6. Offside Logic    → finds 2nd last defender   │
│                         computes offside line    │
│                         checks attackers vs line │
│  7. Possession Check → detects ball transfers    │
│  8. Radar View       → 2D bird's-eye pitch map   │
└──────────────────────────────────────────────────┘
     │
     ▼
Annotated Video + Verdict + Involvement Log
```

---

## Features

- **Automatic team classification** — distinguishes two teams by jersey color using clustering
- **Goalkeeper team resolution** — assigns goalkeepers to the correct team by proximity
- **Offside line detection** — finds the 2nd last defender position per frame
- **Ball possession tracking** — detects when an offside player receives the ball
- **2D radar view** — bird's-eye pitch map shown side-by-side with the original video
- **Involvement log** — lists every frame where an offside player touched the ball
- **Pre-flight validation** — rejects non-football or too-short videos before processing
- **Clean web UI** — dark theme, drag & drop upload, verdict banner, log table

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Multi-object Tracking | ByteTrack (via Supervision) |
| Pitch Keypoint Detection | Custom YOLO model |
| Team Classification | K-means clustering on jersey crops |
| Homography / View Transform | Roboflow Sports library |
| Backend API | FastAPI |
| Video Processing | OpenCV + Supervision |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project Structure

```
offside-ai/
│
├── main.py                       # FastAPI app — upload, validate, run pipeline
├── pipeline_3.py                 # Full detection + offside pipeline
├── offside_2.py                  # Offside logic (line, check, involvement)
├── validator.py                  # Pre-flight video validation
├── ball_tracking.py              # Standalone ball path tracker (utility)
├── pitch_detection.py            # Standalone pitch annotator (utility)
├── player_detection.py           # Standalone player annotator (utility)
│
├── index.html                    # Single-page frontend
│
├── football-pitch-detection.pt   # YOLO pitch keypoint model
├── football-player-detection.pt  # YOLO player detection model
│
└── requirements.txt              # Python dependencies
```

---

## Offside Logic Explained

### Step 1 — Find attacking direction
The system locates the goalkeeper closest to either goal line. The team of that goalkeeper is the **defending team**. The opposite team is **attacking**.

### Step 2 — Compute the offside line
All defending players (including the goalkeeper) are mapped to pitch coordinates via homography. They are sorted by distance to the defending goal. The **2nd closest** player to the goal defines the offside line x-coordinate.

### Step 3 — Check attackers
An attacker is flagged offside if:
- Their pitch x-coordinate is **closer to the defending goal than the offside line**
- AND **closer to the defending goal than the ball**

### Step 4 — Involvement detection
A flagged player becomes an **offside involvement** when ball possession transfers to them — i.e. the closest player to the ball changes from someone else to an offside-flagged player.

---

## API

### `POST /analyze`

Upload a video file for analysis.

**Request:** `multipart/form-data` with field `video`

**Response:**
```json
{
  "verdict": "OFFSIDE",
  "output_video_url": "/outputs/abc123_output.mp4",
  "total_events": 2,
  "involvement_log": [
    {
      "frame": 142,
      "tracker_id": 7,
      "from_player": 3
    },
    {
      "frame": 198,
      "tracker_id": 7,
      "from_player": null
    }
  ]
}
```

| Field | Description |
|---|---|
| `verdict` | `"OFFSIDE"` or `"ONSIDE"` |
| `output_video_url` | URL to the annotated output video |
| `total_events` | Number of offside involvement events |
| `involvement_log` | Per-event details: frame, player tracker ID, who passed |
| `from_player` | Tracker ID of the passer, or `null` if ball was loose |

---

## Installation & Running Locally

### Prerequisites
- Python 3.10+
- CUDA-capable GPU recommended (runs on CPU but slow)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/sabkatdesh/football_offside-ai.git
cd football_offside-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the models (see Models section below)

# 5. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 6. Open in browser
# http://localhost:8000
```

---

## Requirements

```
fastapi
uvicorn
python-multipart
numpy
opencv-python-headless
torch
torchvision
ultralytics
supervision
git+https://github.com/roboflow/sports.git
tqdm
```

---

## Models

Two custom YOLO models are required. They are hosted on HuggingFace and must be placed in the root of the project directory before running.

| Model | Purpose | Download |
|---|---|---|
| `football-pitch-detection.pt` | Detects 32 pitch keypoints for homography | [🤗 Sabkat/football-pitch-detection](https://huggingface.co/Sabkat/football-pitch-detection) |
| `football-player-detection.pt` | Detects ball, players, goalkeepers, referees | [🤗 Sabkat/football-player-detection](https://huggingface.co/Sabkat/football-player-detection) |

### Download instructions

You can download the `.pt` files directly from the HuggingFace model pages linked above, or use the HuggingFace CLI:

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Sabkat/football-pitch-detection', filename='football-pitch-detection.pt', local_dir='.')
hf_hub_download(repo_id='Sabkat/football-player-detection', filename='football-player-detection.pt', local_dir='.')
"
```

Place both `.pt` files in the root project directory alongside `main.py`.

---

## Validation Rules

Before running the full pipeline, the validator checks:

| Check | Threshold |
|---|---|
| Minimum video length | 30 frames |
| Pitch detected in sampled frames | At least 2 out of 5 sampled frames |
| Players detected in sampled frames | At least 10 player crops total |

If any check fails, the API returns a `422` with a human-readable reason shown in the UI.

---

## Frontend

Single HTML file with no framework dependencies:

- Drag & drop or click-to-browse upload
- Live input video preview before analysis
- Indeterminate progress bar during processing
- **OFFSIDE** (red) / **ONSIDE** (green) verdict banner
- Side-by-side input and output video panels
- Involvement log table with frame numbers and player IDs
- Reset button to analyze another clip

---

## Limitations

- Works best with broadcast-angle footage (wide shot showing most of the pitch)
- Accuracy depends on pitch keypoint detection quality (needs clear pitch lines)
- CPU inference is slow — expect 1–3 minutes per clip on a free-tier server
- Team classification may struggle if team jerseys are very similar in color
- Does not handle multiple simultaneous offside situations independently

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Supervision](https://github.com/roboflow/supervision)
- [Roboflow Sports](https://github.com/roboflow/sports)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
