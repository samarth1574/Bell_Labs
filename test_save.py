import cv2
import numpy as np
from pathlib import Path
import os

print(f"Current Working Directory: {os.getcwd()}")
PROJECT_ROOT = Path(__file__).resolve().parent
print(f"Project Root: {PROJECT_ROOT}")

test_img = np.zeros((100, 100, 3), dtype=np.uint8)
out_path = PROJECT_ROOT / "reports" / "figures" / "test_write.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

success = cv2.imwrite(str(out_path), test_img)
print(f"Save Success: {success}")
print(f"Abs Path: {out_path.resolve()}")

if out_path.exists():
    print(f"Confirmed: File exists at {out_path}")
else:
    print(f"FAILED: File does not exist.")
