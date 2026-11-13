"""Focused tests for shared inference assessment and incoming-media persistence."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import database
from civiscan.ai.assessment import assess_severity, estimate_repair_cost
from civiscan.workers import incoming_worker


class _Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value

    def item(self):
        return self.value


class _Box:
    xyxy = [_Tensor([10.0, 10.0, 90.0, 90.0])]
    conf = [_Tensor(0.91)]
    cls = [_Tensor(2)]


class _Prediction:
    boxes = [_Box()]

    def plot(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)


class _Model:
    names = {2: "pothole"}

    def __init__(self):
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        return [_Prediction()]


class MediaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.worker_paths = {
            name: getattr(incoming_worker, name)
            for name in (
                "INCOMING_IMAGE_DIR",
                "PROCESSING_DIR",
                "DATA_PROCESSED_DIR",
                "FAILED_MEDIA_DIR",
                "EXTRACTED_FRAMES_DIR",
                "ANNOTATED_MEDIA_DIR",
                "_wait_until_stable",
            )
        }
        self.database_path = root / "civiscan-test.db"
        database.DATABASE_PATH = self.database_path
        database.initialize_database()
        incoming_worker.INCOMING_IMAGE_DIR = root / "data" / "incoming_images"
        incoming_worker.PROCESSING_DIR = root / "data" / "processing"
        incoming_worker.DATA_PROCESSED_DIR = root / "data" / "processed"
        incoming_worker.FAILED_MEDIA_DIR = root / "data" / "failed"
        incoming_worker.EXTRACTED_FRAMES_DIR = root / "data" / "extracted_frames"
        incoming_worker.ANNOTATED_MEDIA_DIR = root / "data" / "annotated_results"
        incoming_worker.INCOMING_IMAGE_DIR.mkdir(parents=True)
        incoming_worker._wait_until_stable = lambda path, retries=3, delay=0.25: path.is_file()
        self.model = _Model()

    def tearDown(self):
        for name, value in self.worker_paths.items():
            setattr(incoming_worker, name, value)
        self.temp_dir.cleanup()

    def test_worker_processes_and_deduplicates_an_image(self):
        image_path = incoming_worker.INCOMING_IMAGE_DIR / "road.jpg"
        success, encoded = cv2.imencode(".jpg", np.full((100, 100, 3), 127, dtype=np.uint8))
        self.assertTrue(success)
        image_path.write_bytes(encoded.tobytes())

        first = incoming_worker.process_file(image_path, model=self.model, model_path=Path("test.pt"))
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["detections"], 1)
        self.assertTrue(Path(first["processed_path"]).is_file())
        self.assertEqual(database.get_media_statistics()["detections"], 1)
        self.assertEqual(self.model.calls, 1)

        duplicate_path = incoming_worker.INCOMING_IMAGE_DIR / "duplicate.jpg"
        duplicate_path.write_bytes(encoded.tobytes())
        duplicate = incoming_worker.process_file(duplicate_path, model=self.model, model_path=Path("test.pt"))
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(database.get_media_statistics()["images_processed"], 2)

        connection = sqlite3.connect(self.database_path)
        try:
            row = connection.execute(
                "SELECT category, severity, source FROM issues ORDER BY issue_id LIMIT 1"
            ).fetchone()
            self.assertEqual(row, ("pothole", "High", "incoming_media"))
        finally:
            connection.close()

    def test_corrupt_image_is_recorded_as_failed(self):
        corrupt_path = incoming_worker.INCOMING_IMAGE_DIR / "broken.png"
        corrupt_path.write_bytes(b"not an image")
        result = incoming_worker.process_file(corrupt_path, model=self.model, model_path=Path("test.pt"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("decode image", result["error"])
        self.assertEqual(database.get_media_statistics()["failed_jobs"], 1)
        self.assertEqual(self.model.calls, 0)

    def test_video_worker_samples_frames_and_persists_the_job(self):
        video_path = incoming_worker.INCOMING_IMAGE_DIR / "road.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (100, 100),
        )
        if not writer.isOpened():
            writer.release()
            self.skipTest("OpenCV MJPG video writer is unavailable.")
        for value in (20, 40, 60, 80, 100):
            writer.write(np.full((100, 100, 3), value, dtype=np.uint8))
        writer.release()

        result = incoming_worker.process_file(video_path, model=self.model, model_path=Path("test.pt"))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["frames_processed"], 1)
        self.assertTrue(list(incoming_worker.EXTRACTED_FRAMES_DIR.glob("*.jpg")))
        self.assertEqual(database.get_media_statistics()["videos_processed"], 1)

    def test_severity_and_cost_assessment_are_explicit_and_configurable(self):
        small = assess_severity("pothole", [0, 0, 10, 10], (100, 100))
        large = assess_severity("pothole", [0, 0, 80, 80], (100, 100))
        self.assertEqual(small["severity"], "Low")
        self.assertEqual(large["severity"], "High")
        self.assertIn("physical dimensions", small["severity_reason"])
        self.assertIsNone(estimate_repair_cost("pothole", None))
        estimate = estimate_repair_cost(
            "pothole",
            {"material": 10, "labour": 20, "equipment": 30, "transportation": 40},
        )
        self.assertEqual(estimate["total_inr"], 100)


if __name__ == "__main__":
    unittest.main()
