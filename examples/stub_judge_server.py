"""A local stand-in for an OpenAI-compatible judge, for trying ``judgebias run``.

It answers every ``POST /v1/chat/completions`` with the slot of the longer
response, except that it says "1" (the first response) whenever the two
lengths are within ``--margin`` characters of each other. That makes a judge
with a length bias and, on near-equal pairs, a position bias. No model is
involved and nothing leaves the machine.

    uv run python examples/stub_judge_server.py --port 8765 &
    uv run judgebias run examples/pairs.jsonl --out judged.jsonl \\
        --base-url http://127.0.0.1:8765/v1 --model stub
    uv run judgebias audit judged.jsonl
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def verdict(user_message: str, margin: int) -> str:
    first = user_message.split("[Response 1]\n", 1)[1].split("\n\n[Response 2]\n", 1)[0]
    second = user_message.split("[Response 2]\n", 1)[1].rsplit("\n\nWhich response", 1)[0]
    if abs(len(first) - len(second)) <= margin:
        return "1"
    return "1" if len(first) > len(second) else "2"


def make_handler(margin: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - name fixed by http.server
            if not self.path.endswith("/chat/completions"):
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = verdict(body["messages"][-1]["content"], margin)
            reply = {"choices": [{"message": {"role": "assistant", "content": text}}]}
            data = json.dumps(reply).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--margin", type=int, default=20)
    args = ap.parse_args()
    HTTPServer(("127.0.0.1", args.port), make_handler(args.margin)).serve_forever()


if __name__ == "__main__":
    main()
