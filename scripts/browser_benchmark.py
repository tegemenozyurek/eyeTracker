"""Step 15: measure how long each part of the live pipeline takes in a real browser.

Starts a small local server (repo root, port 8030) and waits for tools/benchmark.html to report back.
The page runs the demo's own code on LFW photos used as video frames: MediaPipe face tracking, the two
aligned eye crops, the eye models (ONNX Runtime Web, WASM), the 13 eyeTrack1 features, the rules,
and eyeTrack1 on a full 60 s window. Results are saved to models/browser_benchmark.json.

Usage:
  python scripts/browser_benchmark.py      # then open http://localhost:8030/tools/benchmark.html
                                           # in a visible browser tab and wait about a minute
"""
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "browser_benchmark.json"
PORT = 8030
done = threading.Event()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path != "/results":
            return self.send_error(404)
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        OUT.write_text(json.dumps(data, indent=2) + "\n")
        self.send_response(204)
        self.end_headers()
        done.set()

    def log_message(self, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Open http://localhost:{PORT}/tools/benchmark.html in a visible browser tab (waiting up to 10 minutes)", flush=True)
    if not done.wait(600):
        sys.exit("No results received.")
    time.sleep(0.2)
    server.shutdown()
    r = json.loads(OUT.read_text())
    print(f"\nBrowser: {r['browser']}  ({r['cores']} logical cores, page visible: {r['visible']})")
    print(f"{r['frames']} frames from {r['photos']} LFW photos; median time per frame:")
    for name, ms in r["per_frame_ms"].items():
        print(f"  {name:<34} {ms:7.2f} ms")
    print(f"  {'eyeTrack1, one 60 s window (every 2 s)':<34} {r['eyetrack1_window_ms']:7.2f} ms")
    print(f"Pipeline per frame with eyeTrack0.5: {r['total_ms_eyetrack05']:.1f} ms -> up to {1000 / r['total_ms_eyetrack05']:.0f} frames "
          f"per second on this machine")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
