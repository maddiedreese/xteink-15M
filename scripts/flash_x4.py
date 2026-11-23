#!/usr/bin/env python3
"""Install Tiny Stories on an original ESP32-C3 X4 with the standard OTA layout."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

FLASH_SIZE = 16 * 1024 * 1024
APP0, APP1, SLOT = 0x10000, 0x650000, 0x640000
LAYERS, VOCAB = 0x200000, 0x540000
FILES = {
    "firmware.bin": (APP0, b"\xe9", LAYERS - APP0),
    "layers.bin": (LAYERS, b"X4LYR001", 3406472),
    "tokenizer.bin": (VOCAB, b"X4TOK001", 529881),
    "embedding.bin": (APP1, b"X4EMB001", 5248008),
}


def check_layout(data):
    slots = []
    for at in range(0, 0x1000, 32):
        magic, kind, subtype, offset, size, label, flags = struct.unpack_from(
            "<HBBII16sI", data, at
        )
        if magic != 0x50AA:
            break
        if kind == 0:
            slots.append((subtype, offset, size, flags))
    if sorted(slots) != [(0x10, APP0, SLOT, 0), (0x11, APP1, SLOT, 0)]:
        raise ValueError("not the supported X4 OTA layout")
    for offset in (0x6000, 0x7000):
        seq, = struct.unpack_from("<I", data, offset)
        if seq not in (0, 0xFFFFFFFF) and (seq - 1) % 2:
            raise ValueError("OTA metadata may select app1; refusing to use it for model weights")


def verify_files(directory):
    manifest = json.loads((directory / "sha256.json").read_text())
    for name, (address, header, size) in FILES.items():
        data = (directory / name).read_bytes()
        valid_size = 0 < len(data) <= size if address == APP0 else len(data) == size
        if not valid_size or not data.startswith(header):
            raise ValueError(f"bad image for {name}")
        if hashlib.sha256(data).hexdigest() != manifest.get(name):
            raise ValueError(f"checksum mismatch in {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="X4 USB serial device")
    parser.add_argument("--backup", required=True, type=Path, help="original full flash backup")
    parser.add_argument("--backup-sha256", required=True, help="recorded SHA-256 of that backup")
    parser.add_argument("--baud", default="921600")
    parser.add_argument("--images", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    verify_files(args.images)
    backup = args.backup.read_bytes()
    if len(backup) != FLASH_SIZE or hashlib.sha256(backup).hexdigest() != args.backup_sha256:
        raise ValueError("full backup size or SHA-256 does not match")
    check_layout(backup[0x8000:APP0])
    cmd = [sys.executable, "-m", "esptool", "--chip", "esp32c3",
           "--port", args.port, "--baud", args.baud]

    def run(action, *extra):
        subprocess.run([*cmd, action, *extra], check=True)

    # Preserve bootloader, partition table, OTA metadata, NVS and SPIFFS. Read
    # back the live layout and only use app1 if it is not a bootable fallback.
    with tempfile.TemporaryDirectory() as temp:
        sectors = Path(temp) / "sectors.bin"
        run("read-flash", "0x8000", "0x8000", str(sectors))
        live = sectors.read_bytes()
        check_layout(live)
        if live[:0x1000] != backup[0x8000:0x9000]:
            raise ValueError("live partition table does not match the backup")
        run("read-flash", hex(APP1), "0x1000", str(sectors))
        first = sectors.read_bytes()
        if first != b"\xff" * 4096 and not first.startswith(b"X4EMB001"):
            raise ValueError("app1 is in use by something other than Tiny Stories")

    order = ("layers.bin", "tokenizer.bin", "embedding.bin", "firmware.bin")
    offsets = [item for name in order for item in (hex(FILES[name][0]), str(args.images / name))]
    run("write-flash", *offsets)
    print("Tiny Stories written and verified by esptool. Wake the X4 to try it.")


if __name__ == "__main__":
    main()
