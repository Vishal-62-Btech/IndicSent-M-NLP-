"""
IndicSent-M: Comprehensive Inference Test Suite
================================================
Automated test harness verifying the compatibility, accuracy, error handling,
and latency profiling of all IndicSent-M inference modes:
1. Classical Text (TF-IDF + Logistic Regression, SVM)
2. Transformer Text (Fine-Tuned XLM-RoBERTa, MuRIL)
3. Image-Only (ResNet-50, EfficientNet-B0)
4. Multimodal Fusion (MuRIL + ResNet-50 Gated Fusion, Concat Fusion)
5. Input Validation & Error Handling (empty text, missing image, corrupt file, missing modalities)
6. Device Placement (CPU mode vs. CUDA GPU mode)
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
import torch
import numpy as np

# Path setup
TESTS_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = TESTS_DIR.parent
PROJECT_ROOT = MODEL_TRAINING_DIR.parent
SRC_DIR = MODEL_TRAINING_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference.predictor import IndicSentPredictor, LABEL_MAP

# Locate sample image for vision and multimodal tests
SAMPLE_IMAGE_CANDIDATE = PROJECT_ROOT / "dataset" / "Memotion 3" / "images" / "train" / "2842.jpg"
if not SAMPLE_IMAGE_CANDIDATE.is_file():
    # Fallback to searching any available image in dataset
    all_imgs = list((PROJECT_ROOT / "dataset").rglob("*.jpg"))
    if all_imgs:
        SAMPLE_IMAGE_CANDIDATE = all_imgs[0]
    else:
        raise RuntimeError("No sample test image found in dataset directory!")

TEST_IMAGE_PATH = str(SAMPLE_IMAGE_CANDIDATE)


class TestIndicSentInference(unittest.TestCase):
    """Unit and Integration Test Suite for IndicSentPredictor."""

    def _assert_valid_prediction(self, res: dict, expected_mode: str):
        """Helper to assert the standard contract of the prediction dictionary."""
        self.assertIsInstance(res, dict)
        self.assertIn("predicted_sentiment", res)
        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertIn("model_name", res)
        self.assertIn("model_mode", res)
        self.assertIn("device", res)
        self.assertIn("timings_ms", res)

        self.assertIn(res["predicted_sentiment"], ["Negative", "Neutral", "Positive"])
        self.assertIn(res["predicted_class"], [0, 1, 2])
        self.assertEqual(res["predicted_sentiment"], LABEL_MAP[res["predicted_class"]])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

        # Check probabilities sum to ~1.0
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.02)

        # Check timings
        timings = res["timings_ms"]
        self.assertIn("preprocessing", timings)
        self.assertIn("model_execution", timings)
        self.assertIn("total", timings)
        self.assertGreaterEqual(timings["preprocessing"], 0.0)
        self.assertGreaterEqual(timings["model_execution"], 0.0)
        self.assertAlmostEqual(timings["total"], timings["preprocessing"] + timings["model_execution"], delta=0.05)

        self.assertEqual(res["model_mode"], expected_mode)

    # -------------------------------------------------------------
    # 1. Classical Text Mode Tests
    # -------------------------------------------------------------
    def test_01_classical_logistic_regression(self):
        """Verify TF-IDF + Logistic Regression classical prediction."""
        predictor = IndicSentPredictor(mode="classical", model_name="logistic_regression")
        res = predictor.predict(text="Wah bhai kya mast meme banaya hai maza aa gaya")
        self._assert_valid_prediction(res, "classical")
        self.assertIn("Logistic Regression", res["model_name"])
        print(f"\n[PASS] Classical Logistic Regression: {res['predicted_sentiment']} ({res['confidence_percent']}%) in {res['timings_ms']['total']} ms")

    def test_02_classical_linear_svm(self):
        """Verify TF-IDF + Linear SVM classical prediction."""
        predictor = IndicSentPredictor(mode="classical", model_name="linear_svm")
        res = predictor.predict(text="Bahut bura aur ganda joke tha ye")
        self._assert_valid_prediction(res, "classical")
        self.assertIn("Linear SVM", res["model_name"])
        print(f"[PASS] Classical Linear SVM: {res['predicted_sentiment']} ({res['confidence_percent']}%) in {res['timings_ms']['total']} ms")

    # -------------------------------------------------------------
    # 2. Transformer Text Mode Tests
    # -------------------------------------------------------------
    def test_03_transformer_xlm_roberta(self):
        """Verify fine-tuned XLM-RoBERTa text prediction."""
        predictor = IndicSentPredictor(mode="transformer_text", model_name="xlm_roberta")
        res = predictor.predict(text="Padhai chhod ke sab ho raha hai life me")
        self._assert_valid_prediction(res, "transformer_text")
        self.assertIn("XLM-RoBERTa", res["model_name"])
        print(f"[PASS] Transformer XLM-RoBERTa: {res['predicted_sentiment']} ({res['confidence_percent']}%) in {res['timings_ms']['total']} ms")

    def test_04_transformer_muril(self):
        """Verify fine-tuned MuRIL text prediction."""
        predictor = IndicSentPredictor(mode="transformer_text", model_name="muril")
        res = predictor.predict(text="Aaj ka din theek thaak raha na acha na bura")
        self._assert_valid_prediction(res, "transformer_text")
        self.assertIn("MuRIL", res["model_name"])
        print(f"[PASS] Transformer MuRIL: {res['predicted_sentiment']} ({res['confidence_percent']}%) in {res['timings_ms']['total']} ms")

    # -------------------------------------------------------------
    # 3. Image-Only Mode Tests
    # -------------------------------------------------------------
    def test_05_image_resnet50(self):
        """Verify ResNet-50 vision classification without downloading ImageNet weights."""
        predictor = IndicSentPredictor(mode="image", model_name="resnet50")
        res = predictor.predict(image_path=TEST_IMAGE_PATH)
        self._assert_valid_prediction(res, "image")
        self.assertIn("ResNet-50", res["model_name"])
        print(f"[PASS] Vision ResNet-50: {res['predicted_sentiment']} ({res['confidence_percent']}%) in {res['timings_ms']['total']} ms")

    def test_06_image_efficientnet_b0(self):
        """Verify EfficientNet-B0 vision classification."""
        predictor = IndicSentPredictor(mode="image", model_name="efficientnet_b0")
        res = predictor.predict(image_path=TEST_IMAGE_PATH)
        self._assert_valid_prediction(res, "image")
        self.assertIn("EfficientNet-B0", res["model_name"])
        print(f"[PASS] Vision EfficientNet-B0: {res['predicted_sentiment']} ({res['confidence_percent']}%) in {res['timings_ms']['total']} ms")

    # -------------------------------------------------------------
    # 4. Multimodal Fusion Mode Tests
    # -------------------------------------------------------------
    def test_07_multimodal_gated_fusion(self):
        """Verify MuRIL + ResNet-50 Gated Fusion multimodal classification."""
        predictor = IndicSentPredictor(mode="multimodal", model_name="gated_fusion")
        res = predictor.predict(
            text="Troll Bollywood Memes BRACE YOURSELVES MITRON",
            image_path=TEST_IMAGE_PATH
        )
        self._assert_valid_prediction(res, "multimodal")
        self.assertIn("Gated Fusion", res["model_name"])
        print(f"[PASS] Multimodal Gated Fusion: {res['predicted_sentiment']} ({res['confidence_percent']}%) in {res['timings_ms']['total']} ms")

    def test_08_multimodal_concat_fusion(self):
        """Verify MuRIL + ResNet-50 Concat Fusion multimodal classification."""
        predictor = IndicSentPredictor(mode="multimodal", model_name="muril_resnet50")
        res = predictor.predict(
            text="Troll Bollywood Memes BRACE YOURSELVES MITRON",
            image_path=TEST_IMAGE_PATH
        )
        self._assert_valid_prediction(res, "multimodal")
        self.assertIn("Concat Fusion", res["model_name"])
        print(f"[PASS] Multimodal Concat Fusion: {res['predicted_sentiment']} ({res['confidence_percent']}%) in {res['timings_ms']['total']} ms")

    # -------------------------------------------------------------
    # 5. Input Validation & Error Handling Tests
    # -------------------------------------------------------------
    def test_09_error_empty_or_whitespace_text(self):
        """Ensure empty and whitespace-only text inputs raise descriptive ValueError."""
        predictor = IndicSentPredictor(mode="classical")
        with self.assertRaises(ValueError) as ctx:
            predictor.predict(text="")
        self.assertIn("empty or whitespace-only", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            predictor.predict(text="   \t\n  ")
        self.assertIn("empty or whitespace-only", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            predictor.predict(text=None)
        self.assertIn("cannot be None", str(ctx.exception))
        print("[PASS] Validation: Empty/None text properly rejected with ValueError.")

    def test_10_error_missing_image_path(self):
        """Ensure non-existent image paths raise FileNotFoundError."""
        predictor = IndicSentPredictor(mode="image", model_name="resnet50")
        with self.assertRaises(FileNotFoundError) as ctx:
            predictor.predict(image_path="non_existent_meme_file_99999.jpg")
        self.assertIn("Image file not found", str(ctx.exception))
        print("[PASS] Validation: Missing image file properly rejected with FileNotFoundError.")

    def test_11_error_corrupted_image(self):
        """Ensure corrupted image bytes raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"NOT_A_VALID_JPEG_IMAGE_HEADER_DATA_123456789")
            corrupt_path = f.name

        try:
            predictor = IndicSentPredictor(mode="image", model_name="resnet50")
            with self.assertRaises(ValueError) as ctx:
                predictor.predict(image_path=corrupt_path)
            self.assertIn("corrupted or invalid", str(ctx.exception))
            print("[PASS] Validation: Corrupted image file properly caught with ValueError.")
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    def test_12_error_multimodal_missing_modality(self):
        """Ensure multimodal mode strictly requires BOTH text and image."""
        predictor = IndicSentPredictor(mode="multimodal", model_name="gated_fusion")

        # Missing image
        with self.assertRaises(ValueError) as ctx:
            predictor.predict(text="Valid Hinglish meme text", image_path=None)
        self.assertIn("Image input cannot be None", str(ctx.exception))

        # Missing text
        with self.assertRaises(ValueError) as ctx:
            predictor.predict(text=None, image_path=TEST_IMAGE_PATH)
        self.assertIn("Text input cannot be None", str(ctx.exception))

        # Empty text with valid image
        with self.assertRaises(ValueError) as ctx:
            predictor.predict(text="   ", image_path=TEST_IMAGE_PATH)
        self.assertIn("empty or whitespace-only", str(ctx.exception))

        print("[PASS] Validation: Multimodal missing modalities properly rejected (no blank canvas fallback).")

    # -------------------------------------------------------------
    # 6. Device Placement Tests
    # -------------------------------------------------------------
    def test_13_cpu_device_execution(self):
        """Ensure explicit CPU execution works accurately."""
        predictor = IndicSentPredictor(mode="transformer_text", model_name="xlm_roberta", device="cpu")
        res = predictor.predict(text="Test execution strictly on CPU processor")
        self.assertEqual(res["device"], "cpu")
        self._assert_valid_prediction(res, "transformer_text")
        print(f"[PASS] CPU Device Placement: Verified device={res['device']}")

    def test_14_cuda_device_execution(self):
        """Ensure CUDA GPU execution works when available."""
        if torch.cuda.is_available():
            predictor = IndicSentPredictor(mode="transformer_text", model_name="xlm_roberta", device="cuda")
            res = predictor.predict(text="Test execution on dedicated NVIDIA GPU")
            self.assertTrue(res["device"].startswith("cuda"))
            self._assert_valid_prediction(res, "transformer_text")
            print(f"[PASS] CUDA Device Placement: Verified device={res['device']}")
        else:
            print("[SKIP] CUDA Device Placement test skipped (CUDA not available).")


def run_all_tests():
    """Runs test suite and reports summary."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIndicSentInference)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
