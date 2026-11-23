#!/usr/bin/env python3
"""Bundle the firmware and its verified model assets for the guarded flasher."""
import hashlib
import json
from pathlib import Path
import zipfile

from flash_x4 import FILES


def main():
    root = Path(__file__).resolve().parent.parent
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    sources = {name: root / "assets" / name for name in FILES}
    sources["firmware.bin"] = root / ".pio/build/x4/firmware.bin"
    with zipfile.ZipFile(dist / "x4-tinystories-flash.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        hashes = {}
        for name, path in sources.items():
            data = path.read_bytes()
            z.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        z.writestr("sha256.json", json.dumps(hashes, indent=2) + "\n")
        for path in (root / "scripts/flash_x4.py", root / "scripts/benchmark.py", root / "README.md"):
            z.write(path, path.name)


if __name__ == "__main__":
    main()
