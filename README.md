# CiviScan

**AI-Powered Civic Infrastructure Intelligence**

CiviScan is a Streamlit application for submitting and reviewing civic infrastructure reports. It combines YOLO image inference, citizen reports, municipal issue management, maps, and a separate worker for processing incoming images and videos.

> **Model limitation:** The checkpoint currently included in this repository is pothole-only. Garbage and damaged streetlight detection are available only when you configure a model trained for those classes. CiviScan uses the class names stored in the selected model; it does not pretend the pothole model detects other categories.

## Features

- Citizen registration, login, profile management, report submission, and report history.
- Municipal-officer login, issue dashboards, issue status updates, GIS markers, and geographic hotspot summaries.
- YOLO inference for uploaded images and incoming media, with annotated images, confidence scores, bounding boxes, model version, and inference timing.
- Incoming-media worker that processes images and sampled video frames outside the Streamlit process.
- SQLite persistence with additive schema initialization and media-job history.
- Explainable image-relative severity suggestions and priority scoring.
- Optional repair-cost estimates based on rates configured by an officer. Estimates are not official quotations.
- Shared CiviScan light and dark themes.

## Requirements

- Python 3.10 or later
- pip
- A CPU or CUDA-capable PyTorch environment

The application uses PyTorch, Ultralytics, OpenCV, Streamlit, SQLite, Folium, and other packages listed in [`requirements.txt`](requirements.txt). Model loading and inference can take longer on CPU.

## Quick start (Windows PowerShell)

Run these commands from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the first municipal-officer administrator. This command prompts for account details and a password without echoing the password:

```powershell
python -m civiscan.admin_setup
```

Start the application:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Citizens can register from the sign-in page. Public registration cannot create officer accounts.

## Configure the detection model

By default, CiviScan checks `model.pt` first. If it is not present, it uses `yolov8_pothole_best.pt`. In this repository, the latter is the available fine-tuned model and contains only the `pothole` class.

To explicitly select another Ultralytics checkpoint, set `CIVISCAN_MODEL_PATH` before starting the application or worker:

```powershell
$env:CIVISCAN_MODEL_PATH = ".\best.pt"
python -m streamlit run app.py
```

The selected checkpoint must be compatible with Ultralytics. Its class names are read from `model.names`. For multi-category detection, provide a checkpoint trained for the desired categories; changing the path alone does not add classes to a model.

The default confidence threshold is `0.44`. Change it with `CIVISCAN_CONFIDENCE_THRESHOLD`.

## Process incoming images and videos

The worker is a separate process. The Streamlit app does not start a continuous watcher automatically.

1. Copy supported files into `data/incoming_images/`.
2. In another terminal, run one of these commands:

   ```powershell
   python -m civiscan.workers.incoming_worker --once
   ```

   Processes the files currently in the folder once.

   ```powershell
   python -m civiscan.workers.incoming_worker --watch
   ```

   Continues monitoring for new files until stopped with **Ctrl+C**.

Supported images: JPG, JPEG, PNG, WEBP.

Supported videos: MP4, AVI, MOV, MKV.

The worker samples video frames rather than processing every frame. It tracks jobs, avoids reprocessing identical completed files, stores annotated results, and moves invalid media to the failed folder. The municipal **Incoming Media** page shows job history and recent annotated detections; it can also queue files and launch a one-time scan.

Runtime folders are created as needed:

| Folder | Purpose |
| --- | --- |
| `data/incoming_images/` | New image and video files |
| `data/processing/` | Files currently being processed |
| `data/processed/` | Successfully processed original media |
| `data/failed/` | Invalid media or files that could not be processed |
| `data/extracted_frames/` | Sampled video frames |
| `data/annotated_results/` | Annotated image and video-frame results |

These are runtime data folders and are excluded from Git.

## Configuration

[`config.py`](config.py) reads configuration from environment variables. [`.env.example`](.env.example) lists example values, but the project does **not** automatically load a `.env` file. Set variables in your shell or use an environment manager; do not commit credentials.

| Variable | Purpose | Default |
| --- | --- | --- |
| `CIVISCAN_DATABASE_PATH` | SQLite database file | `civiscan.db` |
| `CIVISCAN_MODEL_PATH` | Preferred YOLO checkpoint | `model.pt` |
| `CIVISCAN_YOLOV8_FALLBACK_PATH` | Fallback YOLO checkpoint | `yolov8_pothole_best.pt` |
| `CIVISCAN_CONFIDENCE_THRESHOLD` | Inference confidence threshold | `0.44` |
| `CIVISCAN_FRAME_INTERVAL_SECONDS` | Video sampling interval | `2` |
| `CIVISCAN_MAX_VIDEO_FRAMES` | Maximum sampled frames per video | `100` |
| `CIVISCAN_INCOMING_POLL_SECONDS` | Watch-mode polling interval | `3` |
| `CIVISCAN_THEME` | Initial theme (`Light` or `Dark`) | `Light` |
| `CIVISCAN_DATA_DIR` | Runtime data root | `data/` |

Each media subfolder and the database path can also be configured independently; see [`config.py`](config.py).

SMTP settings are available for existing password-reset email configuration. Email delivery of new civic-issue alerts is not implemented.

## Database and security

CiviScan uses SQLite and initializes or extends the database without intentionally deleting existing users or issues. The schema includes users, issues, detections, media jobs, media assets, status history, repair records, notifications, maintenance rates, and model-evaluation records.

Passwords are hashed. Officer accounts are created by an administrator; citizens cannot register as officers. Municipal-only pages and database operations enforce role checks. Keep database backups and configure SMTP or model paths through trusted environment settings.

## Development and tests

Run the test suite:

```powershell
python -m unittest discover -s tests -v
```

The tests cover authentication and role checks, Streamlit routes, severity and cost assessment, incoming image processing, duplicate handling, corrupt media, and sampled video processing.

Existing model and training utilities include:

```powershell
python inference_yolov8.py path\to\image.jpg
python inference.py path\to\image.jpg
python train_yolov8.py --validate-only
python evaluate.py
```

Training is not started automatically when the app launches. Use an appropriate dataset and evaluation split when training or evaluating models.

## Project layout

```text
app.py                         Streamlit entry point
auth.py                        Authentication and role management
database.py                    SQLite schema, migrations, and persistence
config.py                      Model, database, and runtime configuration
civiscan/
  ai/                          Shared inference and assessment services
  pages/                       Citizen, municipal, and incoming-media pages
  ui/                          Navigation, components, and theme
  workers/                     Incoming image/video worker
data/                          Runtime media folders (Git-ignored)
dataset_yolo/                  YOLO-format dataset, if supplied
evaluation_metrics/            Existing model evaluation artifacts
tests/                         Authentication, UI, and media-pipeline tests
```

## Current limitations

- The included fine-tuned checkpoint detects potholes only. Configure a suitable multi-class checkpoint for garbage or damaged streetlights.
- Severity is an image-based suggestion, not a physical measurement or an on-site safety assessment.
- Repair-cost estimates are unavailable until an officer configures local material, labour, equipment, and transportation assumptions. They are indicative only.
- GPS is stored only when supplied by the reporter or available in image metadata; CiviScan does not invent coordinates.
- Full repair assignment and before/after verification workflows, issue-alert email delivery, dataset administration/augmentation, and model-evaluation comparison are not complete.
- Segmentation is not implemented. It requires pixel-level mask annotations; bounding-box labels are not segmentation masks.

See [`docs/architecture.md`](docs/architecture.md) for additional architecture and implementation notes.
