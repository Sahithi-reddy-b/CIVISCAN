"""
=============================================================
  Pothole Detection — Grad-CAM Explainability
  Gradient-weighted Class Activation Mapping
=============================================================

Grad-CAM (Interview Showstopper):
──────────────────────────────────
  "How do you know WHAT the model is looking at?"

  Grad-CAM answers this by:
  1. Forward pass → get predictions
  2. Backward pass → compute gradients of target class w.r.t.
     the last convolutional feature map
  3. Global Average Pool the gradients → importance weights
  4. Weighted sum of feature maps → heatmap
  5. ReLU → keep only positive activations
  6. Overlay on original image

  This demonstrates MODEL INTERPRETABILITY — a key topic in
  responsible AI and production deployment interviews.
"""

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from config import (
    DEVICE, IMAGE_SIZE, NUM_CLASSES,
    GRADCAM_OUTPUT_DIR, DEFAULT_WEIGHTS,
)
from model import load_model, YOLOv3Tiny


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks.

    Generates a heatmap showing which regions of the input image
    contributed most to the model's prediction.

    Usage:
        gradcam = GradCAM(model)
        heatmap, overlay = gradcam.generate(image_path)
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: YOLOv3Tiny model
            target_layer: Which layer to visualize (default: last conv in backbone)
        """
        self.model = model
        self.model.eval()

        # Hook into the last convolutional layer of the backbone
        # This is where the highest-level features are
        if target_layer is None:
            # The backbone's last Conv2d is at index -3 (before BN and LeakyReLU)
            self.target_layer = model.backbone[-3]
        else:
            self.target_layer = target_layer

        # Storage for activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_path, target_class=0):
        """
        Generate Grad-CAM heatmap for a given image.

        Args:
            image_path: Path to input image
            target_class: Class index to explain (0 = pothole)

        Returns:
            heatmap: Colored heatmap (numpy array, BGR)
            overlay: Heatmap overlaid on original image
            original: Original image (numpy array, BGR)
        """
        # Load and preprocess image
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
        tensor = transforms.ToTensor()(resized).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        output = self.model(tensor)

        # Target: max objectness score for the target class
        # output shape: [1, anchors, grid, grid, 5+classes]
        class_scores = output[..., :NUM_CLASSES]
        objectness = output[..., NUM_CLASSES]

        # Combined score for target class
        target_score = (class_scores[..., target_class] * torch.sigmoid(objectness)).max()

        # Backward pass
        target_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]     # [C, H, W]
        activations = self.activations[0] # [C, H, W]

        # Global average pooling of gradients → channel importance weights
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only keep positive contributions
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Convert to numpy and resize to original image size
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (original.shape[1], original.shape[0]))

        # Create colored heatmap (JET colormap)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )

        # Overlay on original image
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        # Add label
        cv2.putText(
            overlay, "Grad-CAM: Model Attention", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        return heatmap, overlay, original

    def generate_and_save(self, image_path, save_dir=GRADCAM_OUTPUT_DIR,
                          target_class=0):
        """
        Generate Grad-CAM and save all visualizations.

        Saves:
        - Original image
        - Heatmap only
        - Overlay (heatmap on original)
        - Side-by-side comparison
        """
        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]

        heatmap, overlay, original = self.generate(image_path, target_class)

        # Save individual outputs
        cv2.imwrite(os.path.join(save_dir, f"{basename}_original.jpg"), original)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_heatmap.jpg"), heatmap)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_overlay.jpg"), overlay)

        # Create side-by-side comparison
        h = min(original.shape[0], 500)
        scale = h / original.shape[0]
        w = int(original.shape[1] * scale)

        orig_resized = cv2.resize(original, (w, h))
        heat_resized = cv2.resize(heatmap, (w, h))
        over_resized = cv2.resize(overlay, (w, h))

        # Labels
        cv2.putText(orig_resized, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(heat_resized, "Heatmap", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(over_resized, "Overlay", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        sep = np.full((h, 3, 3), 128, dtype=np.uint8)
        comparison = np.hstack([orig_resized, sep, heat_resized, sep, over_resized])

        comp_path = os.path.join(save_dir, f"{basename}_comparison.jpg")
        cv2.imwrite(comp_path, comparison)

        return comp_path


if __name__ == "__main__":
    """
    Demo: Generate Grad-CAM visualizations for test images.

    Usage:
        python gradcam.py                    # Process 5 test images
        python gradcam.py path/to/image.jpg  # Process specific image
    """
    import sys
    from config import TEST_IMAGES
    from glob import glob

    print("\n" + "=" * 60)
    print("  🔥 Grad-CAM — Model Explainability Demo")
    print("=" * 60)

    model = load_model()
    gradcam = GradCAM(model)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        result = gradcam.generate_and_save(path)
        print(f"\n  ✅ Saved: {result}")
    else:
        # Process first 5 test images
        images = sorted(glob(os.path.join(TEST_IMAGES, "*.jpg")))[:5]
        print(f"\n  Processing {len(images)} test images...")

        for img_path in images:
            try:
                result = gradcam.generate_and_save(img_path)
                print(f"  ✅ {os.path.basename(img_path)} → saved")
            except Exception as e:
                print(f"  ⚠️ {os.path.basename(img_path)} → error: {e}")

    print(f"\n  📁 Results saved to: {GRADCAM_OUTPUT_DIR}")
