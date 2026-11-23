#!/usr/bin/env python3
"""Request a deterministic on-device benchmark over USB CDC; inference stays on the X4."""
import argparse
import re
import time

import serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    ser = serial.Serial(port=None, baudrate=115200, timeout=0.5)
    ser.port = args.port
    ser.rts = False
    # Assert DTR for USB CDC output, without using the RTS reset line.
    with ser:
        time.sleep(0.5)
        ser.write(b"?b")
        limit = time.monotonic() + args.timeout
        result = None
        while time.monotonic() < limit:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(line, flush=True)
            if line.startswith("BENCH DONE"):
                result = line
            if result and line.startswith("STORY "):
                break
        match = re.search(r"tokens=(\d+) decode_us=(\d+) first_us=(\d+)", result or "")
        if not match:
            raise SystemExit("No valid on-device benchmark result.")
        tokens, compute, first = map(int, match.groups())
        if not tokens or not compute:
            raise SystemExit("The X4 did not generate a token.")
        print(f"{tokens * 1e6 / compute:.3f} tokens/s compute; first token in {first / 1e6:.2f} s.")


if __name__ == "__main__":
    main()
