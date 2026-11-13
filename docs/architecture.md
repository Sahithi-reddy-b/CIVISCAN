# CiviScan architecture and implementation status

## Application flow

- `app.py` is the Streamlit entry point. Authentication, navigation, and the shared theme remain in the existing `auth.py` and `civiscan/` modules.
- `civiscan/ai/model_service.py` resolves configured Ultralytics weights, loads the model once per path, and returns class-aware detections, annotated image data, model metadata, and inference timing. Labels come from `model.names`; class numbers are not used to infer categories.
- Citizen image uploads and the independent incoming-media worker use that same inference service.
- `civiscan/workers/incoming_worker.py` scans `data/incoming_images`, moves work through `processing`, and stores successful inputs under `processed`; invalid files go to `failed`. It is a separate process, not a Streamlit background loop.
- `database.py` performs additive SQLite migrations. Existing `issues` and user data are retained. Media jobs, detections, assets, status history, notifications, repair records, maintenance assumptions, and evaluation records have separate tables.
- `civiscan/ai/assessment.py` reports image-relative visible extent, not physical dimensions. Cost estimates remain unavailable until an officer configures category rates in Settings.

## Model selection

`CIVISCAN_MODEL_PATH` takes precedence. The default is `model.pt`; if it is absent, the configured existing `yolov8_pothole_best.pt` fallback is used. Set `CIVISCAN_MODEL_PATH` to an explicitly supplied multi-class checkpoint (for example `best.pt`) to use its model-provided classes. The repository currently contains a pothole-only checkpoint, so it must not be interpreted as a garbage or streetlight detector.

## Processing and storage

Supported incoming images: JPG, JPEG, PNG, WEBP. Supported videos: MP4, AVI, MOV, MKV. Video frames are sampled using `CIVISCAN_FRAME_INTERVAL_SECONDS` and capped by `CIVISCAN_MAX_VIDEO_FRAMES`. SHA-256 deduplication prevents repeat inference for completed inputs while retaining a processing-job entry for the duplicate.

All paths can be redirected with environment variables; see `.env.example`. The worker writes a status heartbeat at `data/worker_status.json`. Logs go to its standard output/error stream.

## Limitations and next work

- The included checkpoint is pothole-only. Garbage and streetlight classification require configuring a checkpoint trained for those classes.
- Severity is a conservative image-relative suggestion; field review is needed for road exposure, obstruction, and public-safety context. Box area is not a physical measurement.
- Cost values are deliberately not pre-filled. Officers must configure local indicative component assumptions; displayed values are estimates, not quotations.
- Historical analytics and GIS use saved reports with valid coordinates. No coordinate is synthesized when GPS is missing. Predictive forecasting, SMTP issue-alert delivery, full repair assignment/verification workflows, dataset augmentation/validation UI, and model evaluation comparison remain follow-on work.
- Segmentation is not trained or claimed. It requires pixel-level masks for each target class and a separate Mask R-CNN or DeepLabV3+ pipeline; YOLO box labels cannot be converted into truthful masks.

## Submission checklist

- Working Streamlit authentication and role-specific navigation.
- Shared model-loading/inference path, annotated images, inference timing, and database-backed prediction output.
- Incoming image/video worker, duplicate handling, failure tracking, sampled video frames, and officer monitoring page.
- Database-backed camera detections and civic issues, configurable severity/cost foundations, and additive schema migrations.
- Existing GIS, hotspots, officer issue list, citizen report history, and light/dark theme retained.
- External requirements: configure the desired model checkpoint and, for repair estimates or email delivery, provide local rates or SMTP credentials through environment settings.
