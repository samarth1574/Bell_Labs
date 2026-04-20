import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from src.models.detector import DenseObjectDetector
from src.utils.visualization import COLORS

def run_real_world_validation():
    # Setup
    sample_dir = Path("data/raw/samples")
    output_dir = Path("reports/figures/real_world_inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Phase 2 Detector (Soft-NMS enabled by default in config)
    config_path = "configs/dl_softnms_density.yaml"
    detector = DenseObjectDetector(config_path=config_path, device="cpu")
    
    samples = sorted(list(sample_dir.glob("*.jpg")))
    print(f"Found {len(samples)} samples. Starting inference...")
    
    for img_path in samples:
        print(f"Processing {img_path.name}...")
        
        # Run Detection
        boxes, scores, labels = detector.detect(str(img_path))
        
        # Load image for visualization
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plotting
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        ax = plt.gca()
        
        # Draw top 100 detections for visibility in dense scenes
        n_show = min(len(boxes), 100)
        for i in range(n_show):
            x1, y1, x2, y2 = boxes[i]
            s = scores[i].item()
            if s < 0.3: continue # Confidence threshold
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=COLORS['primary'], linewidth=1)
            ax.add_patch(rect)
        
        plt.title(f"Real-World Detection: {img_path.name}\nObjects Detected (Top 100): {len(boxes)}", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        out_path = output_dir / f"detection_{img_path.name}"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved result to {out_path}")

if __name__ == "__main__":
    run_real_world_validation()
