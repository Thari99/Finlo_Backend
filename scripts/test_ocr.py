"""
Quick CLI for testing the OCR pipeline without going through Flutter or
the HTTP layer. Reads an image from disk, sends it to Gemini, prints the
parsed JSON.

Usage (from inside the backend container):
    docker compose exec backend python scripts/test_ocr.py /path/to/bill.jpg

Or directly with a relative path if you copied an image into the project:
    docker compose exec backend python scripts/test_ocr.py images/bill1.jpg

Exits 0 on success, 1 on any failure — useful for piping into other tools.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import pathlib
import sys

# When you run `python scripts/test_ocr.py ...`, Python adds the script's
# OWN directory (`scripts/`) to sys.path, not the backend root. Push the
# parent dir on so `from services.gemini_ocr import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


_ALLOWED = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def _guess_mime(path: pathlib.Path) -> str:
    # mimetypes is fast enough; falls back to looking at the file header
    # for obvious cases if the extension is wrong.
    mt, _ = mimetypes.guess_type(str(path))
    if mt and mt.lower() in _ALLOWED:
        return mt.lower()
    # Sniff the first few bytes — covers extension-less uploads.
    with path.open("rb") as f:
        head = f.read(16)
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # last-resort guess


async def main(image_path: str) -> int:
    path = pathlib.Path(image_path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    image_bytes = path.read_bytes()
    mime = _guess_mime(path)
    print(f"→ {path}  ({len(image_bytes):,} bytes, {mime})", file=sys.stderr)

    # Imported lazily so a missing GEMINI_API_KEY surfaces a clean error
    # instead of a top-of-file crash before we've printed our header.
    from services.gemini_ocr import OcrError, scan_receipt  # noqa: WPS433

    try:
        result = await scan_receipt(image_bytes, mime)
    except OcrError as e:
        print(f"OCR_ERROR ({e.code}): {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"UNEXPECTED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_ocr.py <image_path>", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1])))
