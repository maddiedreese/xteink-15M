# Tiny Stories on Xteink X4

Karpathy's **stories15M** model runs entirely on the original **Xteink X4**
(ESP32-C3, 16 MiB flash). It needs no network, paired computer, or SD card
after installation. It replaces the reader application until you restore
your original firmware. It does not target X3 or other X4 hardware.

## Controls

| Buttons | Composer | While writing | Finished story |
| --- | --- | --- | --- |
| Left/Right and Up/Down | Move between keys | — | — |
| Confirm | Select a key | — | Run again |
| Back | Delete one character | Stop writing | Edit the prompt |
| Power | Sleep / wake | Sleep / wake | Sleep / wake |

On the keyboard, **SHIFT** changes case and punctuation, **SPACE** inserts a
space, and **GO** begins generation. Choose **DEMO** to fill in “Once upon a
time” and move the highlight to **GO**. Press Confirm again to start. A blank
prompt starts from the beginning-of-story token.

## Build and test

```sh
git submodule update --init
python3 -m venv .venv
. .venv/bin/activate
pip install platformio numpy "esptool>=5,<6"
python scripts/pack.py
pio run
python scripts/package.py
```

The packer downloads and checksum-verifies the original stories15M checkpoint
and Llama tokenizer. The model is stored as 4-bit symmetric weights in groups
of 32, with bfloat16 scales. Inference uses int8 activations for matrix
multiplication and an int8 key/value cache. The firmware reserves a
32–64-token window at boot according to free RAM. The window includes the
prompt and the beginning-of-story token; generation stops at the limit.
The internal flash holds the weights; the ESP32-C3 cannot map them all at
once, so it can stream the shared embeddings in chunks.

Run the host smoke test after packing the assets:

```sh
c++ -std=c++17 -O3 -Wall -Wextra -Werror -Isrc tests/smoke.cpp \
  src/model.cpp src/controls.cpp src/screen.cpp -o tests/smoke
./tests/smoke
```

## Install on a compatible X4

Install using the flash ZIP, not `pio run -t upload`: only the former includes
the model. This assumes an original X4 with the two 6.25 MiB OTA slots shown
in the included partition layout, currently booting app0, and an unused app1.
Do not use this installer with a different partition layout or a used app1.

1. Connect the awake X4 with a USB **data** cable and find its serial port
   (`/dev/cu.usbmodem…` on macOS).
2. Save a full backup somewhere safe. Read it twice and compare hashes
   before proceeding. Set `PORT` to the X4's actual device path.

   ```sh
   python -m esptool --chip esp32c3 --port "$PORT" read-flash 0x0 0x1000000 x4-original.bin
   python -m esptool --chip esp32c3 --port "$PORT" read-flash 0x0 0x1000000 x4-check.bin
   cmp x4-original.bin x4-check.bin
   shasum -a 256 x4-original.bin
   ```

3. Extract the flash ZIP. From its directory, provide the backup path and
   recorded SHA-256:

   ```sh
   python flash_x4.py --port "$PORT" --backup /path/to/x4-original.bin \
     --backup-sha256 <recorded-sha256>
   ```

The installer verifies file checksums, the live flash layout, and that app1
does not hold another application. It leaves the partition table, bootloader,
OTA metadata, NVS, SPIFFS, and SD card untouched. It writes the app and two
model assets in app0, and the embeddings in app1. **Restore your original
firmware** from the verified backup:

```sh
python -m esptool --chip esp32c3 --port "$PORT" write-flash 0x0 /path/to/x4-original.bin
```

## Measure on the device

`python benchmark.py "$PORT"` requests 16 output tokens or stops earlier at
EOS. It reports the allocated context, compute-only token rate, and time to
first output token. The USB connection only sends the benchmark command and
reads results. It provides no inference compute.

On an original X4 with the SSD1677 panel, the benchmark allocated **40
context tokens**, averaged **3.03 s per generated token** over 16 generated
tokens, and returned the first token in **7.25 s** including prompt processing.

The inference and tokenizer are adapted from
[Karpathy's llama2.c](https://github.com/karpathy/llama2.c) (see
`LICENSE-llama2.c`). The display and power path uses
[FreeInk SDK](https://github.com/Free-Ink/freeink-sdk); button and wiring setup
comes from the X4 tree firmware. The bitmap font is public-domain font8x8.
