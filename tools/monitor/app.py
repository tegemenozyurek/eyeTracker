"""Training monitor: a local dashboard for every run in runs/<name>/metrics.jsonl.

    python tools/monitor/app.py            ->  http://127.0.0.1:8501 (opens in the browser)

Standard library only and read-only, except for the Stop button, which creates
runs/<name>/STOP; the training script sees it, keeps its best checkpoint and exits.
The page polls every few seconds and only fetches the lines added since its last poll.
"""
import argparse
import json
import re
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs"
NAME = re.compile(r"^[\w.\-]+$")


class Summary:
    """Incrementally parsed summary of one metrics.jsonl (only new bytes are read on each poll)."""

    def __init__(self, path):
        self.path, self.offset, self.info = path, 0, {}

    def update(self):
        size = self.path.stat().st_size
        if size < self.offset:  # the run was restarted: the file is new
            self.offset, self.info = 0, {}
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        complete = chunk[:chunk.rfind(b"\n") + 1]  # ignore a line that is still being written
        self.offset += len(complete)
        for line in complete.splitlines():
            event = json.loads(line)
            kind = event["type"]
            if kind == "start":
                self.info = {"kind": event.get("kind"), "started": event["time"], "status": "running",
                             "epochs": event.get("epochs"), "total": event.get("total")}
            elif kind == "epoch":
                self.info["epoch"] = event["epoch"]
                for key in ("val_acc", "val_loss"):
                    if key in event:
                        self.info[key] = event[key]
            elif kind == "progress":
                self.info.update(done=event["done"], total=event["total"])
            elif kind == "end":
                self.info["status"] = event["status"]
            self.info["updated"] = event["time"]
        return self.info


summaries = {}
lock = threading.Lock()


def run_dir(name):
    if not name or not NAME.match(name) or not (RUNS / name / "metrics.jsonl").exists():
        return None
    return RUNS / name


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/runs":
            runs = []
            with lock:
                for path in sorted(RUNS.glob("*/metrics.jsonl")) if RUNS.exists() else []:
                    s = summaries.setdefault(path.parent.name, Summary(path))
                    runs.append({"name": path.parent.name, **s.update()})
            runs.sort(key=lambda r: r.get("updated", 0), reverse=True)
            return self.send_json({"runs": runs, "now": time.time()})
        if url.path == "/api/events":
            folder = run_dir(query.get("name"))
            if not folder:
                return self.send_json({"error": "unknown run"}, 404)
            start = int(query.get("from", 0))
            path = folder / "metrics.jsonl"
            reset = path.stat().st_size < start
            with open(path, "rb") as f:
                f.seek(0 if reset else start)
                chunk = f.read()
            complete = chunk[:chunk.rfind(b"\n") + 1]
            events = [json.loads(line) for line in complete.splitlines()]
            return self.send_json({"events": events, "next": (0 if reset else start) + len(complete),
                                   "reset": reset, "now": time.time()})
        if url.path == "/api/samples":
            folder = run_dir(query.get("name"))
            path = folder / "samples.json" if folder else None
            if not path or not path.exists():
                return self.send_json({"epoch": None, "items": []})
            return self.send_json(json.loads(path.read_text()))
        if url.path in ("/", "/index.html"):
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        folder = run_dir(parse_qs(url.query).get("name", [None])[0])
        if url.path != "/api/stop" or not folder:
            return self.send_error(404)
        (folder / "STOP").write_text(f"stop requested from the monitor at {time.ctime()}\n")
        return self.send_json({"ok": True})

    def log_message(self, *args):
        pass  # keep the console quiet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Training monitor: {url}  (reading {RUNS})  Ctrl+C to quit")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
