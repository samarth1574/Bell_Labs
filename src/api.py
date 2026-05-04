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
                    
                    response_data = {
                        "mode": result["mode"],
                        "count": result["count"],
                        "classical_count": result["classical_count"],
                        "boxes": result["boxes"],
                        "scores": result["scores"],
                    }
                    self._send_json(200, response_data)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_error(500, str(e))
        else:
            self._send_error(404, "Not Found")

    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_error(self, status_code, message):
        self._send_json(status_code, {"detail": message})

def preload_detector():
    print("Pre-loading ML model in background...")
    try:
        get_detector()
        print("ML model pre-loaded successfully!")
    except Exception as e:
        print(f"Failed to pre-load model: {e}")

def run(server_class=HTTPServer, handler_class=APIHandler, port=8000):
    server_class.allow_reuse_address = True
    server_address = ('0.0.0.0', port)
    print(f"Creating server on {server_address}...")
    try:
        httpd = server_class(server_address, handler_class)
        print(f"Starting lightweight Python backend on port {port}...")
        
        # Start background thread to preload the detector
        threading.Thread(target=preload_detector, daemon=True).start()
        
        httpd.serve_forever()
    except Exception as e:
        print(f"FAILED TO START SERVER: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
