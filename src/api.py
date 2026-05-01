import json
import base64
import tempfile
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Local imports moved inside functions for lazy loading

import threading

# Initialize detector lazily
_detector = None
_detector_lock = threading.Lock()

def get_detector():
    global _detector
    with _detector_lock:
        if _detector is None:
            # Lazy import to avoid startup hang
            from src.models.hybrid_sku_detector import HybridSkuDetector, HybridConfig
            
            weights_path = Path("runs/phase3/yolo11_sku110k/weights/best.pt")
            weights = str(weights_path) if weights_path.exists() else "yolo11n.pt"
            print(f"Loading detector with weights: {weights}")
            config = HybridConfig(weights=weights)
            _detector = HybridSkuDetector(config)
        return _detector

class APIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/analyze':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                mode = data.get('mode', 'hybrid')
                image_base64 = data.get('image')
                
                if not image_base64:
                    self._send_error(400, "No image provided")
                    return
                
                # Strip data URL prefix if present (e.g., "data:image/jpeg;base64,...")
                if "base64," in image_base64:
                    image_base64 = image_base64.split("base64,")[1]
                
                image_bytes = base64.b64decode(image_base64)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name
                
                try:
                    detector = get_detector()
                    result = detector.predict(tmp_path, mode=mode)
                    
                  