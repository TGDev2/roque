"""Receive remote frames over HTTP and run the live board-to-FEN pipeline.

Example:
    PYTHONPATH=src ./.venv/bin/python scripts/remote_live.py --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

import cv2
import numpy as np

# Allow running as `python scripts/remote_live.py` without install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chesslab.live import LiveProcessorConfig, LivePositionProcessor

logger = logging.getLogger("chesslab.remote_live")
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    parser.add_argument("--theme", default="chess.com-green", help="Manual theme ID (default: chess.com-green).")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Recognition confidence threshold (default: 0.60).",
    )
    parser.add_argument(
        "--black-perspective",
        action="store_true",
        help="Treat the incoming board as black perspective.",
    )
    parser.add_argument(
        "--allow-uncertain",
        action="store_true",
        help="Continue to FEN generation even if recognition is uncertain.",
    )
    parser.add_argument(
        "--active-color",
        default="w",
        choices=("w", "b"),
        help="FEN side to move used for generated positions (default: w).",
    )
    parser.add_argument(
        "--halfmove",
        type=int,
        default=0,
        help="FEN halfmove clock (default: 0).",
    )
    parser.add_argument(
        "--fullmove",
        type=int,
        default=1,
        help="FEN fullmove number (default: 1).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


class RemoteFrameService:
    """Decode uploaded images and process them through the live pipeline."""

    def __init__(self, processor: LivePositionProcessor) -> None:
        self._processor = processor
        self._lock = threading.Lock()
        self._last_result: dict[str, object] | None = None
        self._last_log_key: tuple[object, ...] | None = None

    @property
    def last_result(self) -> dict[str, object] | None:
        return self._last_result

    def process_bytes(self, payload: bytes) -> dict[str, object]:
        if not payload:
            raise ValueError("Request body is empty.")
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image too large ({len(payload)} bytes > {MAX_IMAGE_BYTES} bytes)."
            )

        frame = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if frame is None:
            raise ValueError("Unable to decode image payload.")

        with self._lock:
            result = self._processor.process_frame(frame)
            payload_dict = result.to_dict()
            self._last_result = payload_dict
            self._maybe_log(payload_dict)
            return payload_dict

    def _maybe_log(self, result: dict[str, object]) -> None:
        status = str(result["status"])
        key = (
            status,
            result.get("changed"),
            result.get("fen"),
            tuple(result.get("uncertain_squares", [])),
            result.get("message"),
        )
        if key == self._last_log_key:
            return
        self._last_log_key = key

        if status == "ready":
            logger.info(
                "status=%s changed=%s fen=%s total=%.1fms",
                status,
                result["changed"],
                result.get("fen"),
                float(result["total_elapsed_ms"]),
            )
            return

        logger.info(
            "status=%s message=%s total=%.1fms",
            status,
            result.get("message"),
            float(result["total_elapsed_ms"]),
        )


def _make_handler(service: RemoteFrameService) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        server_version = "ChessLabRemoteLive/0.1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return

            if self.path == "/last-result":
                if service.last_result is None:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "No frame has been processed yet."},
                    )
                    return
                self._send_json(HTTPStatus.OK, service.last_result)
                return

            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Unknown endpoint. Use POST /frame, GET /health or GET /last-result."},
            )

        def do_POST(self) -> None:
            if self.path != "/frame":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length."})
                return

            if content_length <= 0:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body is empty."})
                return

            payload = self.rfile.read(content_length)
            try:
                result = service.process_bytes(payload)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:  # pragma: no cover - defensive server guard
                logger.exception("Unexpected error while processing frame.")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, status: HTTPStatus, data: dict[str, object]) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return RequestHandler


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Ignore expected disconnect errors from short-lived clients."""

    def handle_error(self, request: object, client_address: tuple[str, int]) -> None:
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError)):
            logger.debug("Client disconnected %s:%s (%s).", client_address[0], client_address[1], exc)
            return
        super().handle_error(request, client_address)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    processor = LivePositionProcessor(
        LiveProcessorConfig(
            theme=args.theme,
            confidence_threshold=args.confidence_threshold,
            white_perspective=not args.black_perspective,
            active_color=args.active_color,
            allow_uncertain=args.allow_uncertain,
            halfmove_clock=args.halfmove,
            fullmove_number=args.fullmove,
        )
    )
    server = QuietThreadingHTTPServer(
        (args.host, args.port),
        _make_handler(RemoteFrameService(processor)),
    )

    logger.info(
        "Listening on http://%s:%d with theme=%s white_perspective=%s",
        args.host,
        args.port,
        args.theme,
        not args.black_perspective,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping remote live server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
