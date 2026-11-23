#!/usr/bin/env python3
"""Pack Karpathy's stories15M checkpoint and tokenizer into X4 flash assets."""
import argparse
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

FILES = {
    "stories15M.bin": (
        "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
        "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a",
    ),
    "tokenizer.bin": (
        "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin",
        "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361",
    ),
}
DIM, HIDDEN, LAYERS, HEADS, VOCAB, SEQ = 288, 768, 6, 6, 32000, 256
GROUP = 32


def source(directory, name):
    url, expected = FILES[name]
    path = directory / name
    data = path.read_bytes() if path.exists() else urllib.request.urlopen(url, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"{name} does not match the pinned checkpoint")
    if not path.exists():
        path.write_bytes(data)
    return data


def q4(weights, width):
    groups = weights.reshape(-1, GROUP)
    scales = np.max(np.abs(groups), axis=1) / np.float32(7)
    # Store rounded bfloat16 scales, and quantize against the stored values.
    bits = scales.view(np.uint32)
    bf16 = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype("<u2")
    decoded = (bf16.astype(np.uint32) << 16).view(np.float32)
    q = np.clip(np.rint(groups / np.where(decoded == 0, 1, decoded)[:, None]),
                -7, 7).astype(np.int8) + 8
    packed = np.empty((len(groups), 18), dtype=np.uint8)
    packed[:, :16] = q[:, ::2] | (q[:, 1::2] << 4)
    packed[:, 16:] = bf16.view(np.uint8).reshape(-1, 2)
    rows = packed.reshape(-1, width // GROUP * 18)
    return np.pad(rows, ((0, 0), (0, -rows.shape[1] % 4))).tobytes()


def pack_model(data, output):
    shape = struct.unpack_from("<7i", data)
    if shape != (DIM, HIDDEN, LAYERS, HEADS, HEADS, VOCAB, SEQ):
        raise ValueError(f"unexpected checkpoint shape: {shape}")
    weights = np.frombuffer(data, dtype="<f4", offset=28)
    offset = 0

    def take(count):
        nonlocal offset
        values = weights[offset:offset + count]
        offset += count
        if len(values) != count:
            raise ValueError("truncated checkpoint")
        return values

    with (output / "embedding.bin").open("wb") as f:
        f.write(b"X4EMB001")
        f.write(q4(take(VOCAB * DIM), DIM))
    att_norm = take(LAYERS * DIM)
    attention = [take(LAYERS * DIM * DIM) for _ in range(4)]
    ffn_norm = take(LAYERS * DIM)
    ffn = [take(LAYERS * DIM * HIDDEN) for _ in range(3)]
    final_norm = take(DIM)
    with (output / "layers.bin").open("wb") as f:
        f.write(b"X4LYR001")
        for norm in (att_norm, ffn_norm, final_norm):
            f.write(norm.tobytes())
        for matrix, width in zip(attention + ffn, (DIM, DIM, DIM, DIM, DIM, HIDDEN, DIM)):
            f.write(q4(matrix, width))
    if len(weights) - offset != SEQ * (DIM // HEADS):
        raise ValueError("unexpected contents after matrices")


def pack_tokenizer(data, output):
    (max_length,) = struct.unpack_from("<I", data)
    offset, tokens, index, blob = 4, [], [], bytearray()
    for _ in range(VOCAB):
        score, size = struct.unpack_from("<fi", data, offset)
        offset += 8
        token = data[offset:offset + size]
        if not 0 <= size <= max_length or len(token) != size:
            raise ValueError("invalid tokenizer entry")
        tokens.append(token)
        index.append(struct.pack("<If", len(blob), score))
        blob += token + b"\0"
        offset += size
    if offset != len(data):
        raise ValueError("unexpected contents after tokenizer")
    order = sorted(range(VOCAB), key=tokens.__getitem__)
    result = b"X4TOK001" + struct.pack("<II", max_length, len(blob))
    result += b"".join(index) + struct.pack(f"<{VOCAB}H", *order) + blob
    (output / "tokenizer.bin").write_bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("upstream"))
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pack_model(source(args.cache, "stories15M.bin"), args.output)
    pack_tokenizer(source(args.cache, "tokenizer.bin"), args.output)


if __name__ == "__main__":
    main()
