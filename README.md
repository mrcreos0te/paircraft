# paircraft

A C++23 Byte-Pair Encoding (BPE) tokenizer library designed to replace the
Python reference implementation in this repo. Built for large language model
pre-training pipelines where corpora routinely exceed available RAM.

---

## Goals

| Goal | Status |
|------|--------|
| Correct BPE training (naive O(N·M)) | ✅ implemented |
| Correct BPE encoding / decoding | ✅ implemented |
| Memory-mapped incremental training (O(N + M·k·log P)) | 🔲 TODO |
| Priority-queue O(N log M) encoder | 🔲 TODO |
| Binary save / load | 🔲 TODO |
| Handles 16 GB+ corpora (corpus stays on disk) | 🔲 TODO (tied to mmap training) |

---

## Design

### Memory layout (training)

The Python `train_fast()` is the target for `train_from_file()`:

```
vals    uint16   token values; 0xFFFF = deletion sentinel   2 B/token
prev    int32    linked-list predecessor index               4 B/token
nxt     int32    linked-list successor index                 4 B/token
pair_positions   per-pair position lists                     4 B/pos
──────────────────────────────────────────────────────────────────────
Total ≈ 14 B/token  (corpus itself stays on disk via mmap)
```

A 16 GB UTF-8 corpus → ~1.1 B tokens at ~1.5 B/char → ~15 GB RAM for a
pure in-memory approach. With `MmapView` the corpus bytes are never copied;
only the frequency table and linked-list arrays live in RAM, cutting peak
usage by roughly 10×.

### Deletion sentinel

Merged-away nodes are marked `vals[j] = 0xFFFF` (never a valid token ID
since `vocab_size ≤ 65 534` is asserted). This avoids a separate boolean
`deleted[]` array and makes liveness checks a single integer comparison.

### Heap staleness

Pair counts are monotonically non-increasing after a merge. A heap entry
`(-count, pair)` is stale iff `counts[pair] != count` at pop time — no
explicit invalidation needed.

---

## Build

Requirements: CMake ≥ 3.25, a C++23 compiler (GCC 13+, Clang 17+, MSVC 19.38+).
GTest is fetched automatically at configure time.

**Windows (MSVC / Visual Studio generator)**

The VS generator is multi-config: `CMAKE_BUILD_TYPE` is ignored at configure
time; pass `--config` at build time instead.

```bat
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Executables land in `build\Release\`.

**Linux / macOS — or Windows with Ninja (single-config)**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # optionally add -G Ninja on Windows
cmake --build build
ctest --test-dir build --output-on-failure
```

Executables land in `build/`.

---

## Tools

All tools live in `tools/` and are compiled into standalone executables linked
against the `paircraft` library.

| Executable | Source | Description |
|------------|--------|-------------|
| `paircraft_cli` | `tools/tokenize.cpp` | Train + encode a text file, print stats |

Planned tools (not yet implemented):

| Executable | Source | Description |
|------------|--------|-------------|
| `paircraft_inspect` | `tools/inspect.cpp` | Inspect a saved `.bin` tokenizer (vocab dump, merge table) |
| `paircraft_bench` | `tools/bench.cpp` | Benchmark training and encoding throughput |

---

### `paircraft_cli`

Reads a corpus file, trains a BPE tokenizer on it, encodes the corpus, and
prints compression statistics. All heavy work goes through the `paircraft`
library; the tool itself only handles CLI argument parsing and I/O.

Source: [tools/tokenize.cpp](tools/tokenize.cpp)

```
Usage: paircraft_cli <corpus.txt> [options]

Options:
  --vocab-size N      Vocabulary size (default: 512, min: 256)
  --output FILE.npy   Write token IDs as NumPy uint16 array (np.load compatible)
  --print-vocab N     Print first N vocab entries after training (0 = skip)
  --help              Show this message
```

```bash
# Train on a file, encode it, show stats
./build/paircraft_cli corpus.txt --vocab-size 1024

# Write tokens as a .npy file and dump first 50 vocab entries
./build/paircraft_cli corpus.txt --vocab-size 1024 \
    --output tokens.npy --print-vocab 50
```

The output file is a standard NumPy v1.0 `.npy` file — dtype `int16`, shape `(N,)`,
matching the dtype produced by the Python `bpe.py` reference implementation:

```python
import numpy as np
encoded = np.load("tokens.npy")   # dtype=int16, shape=(N,)
```

> **Note:** `int16` caps the usable vocab at 32 767 tokens (values 0–32 767 are
> identical in `int16` and `uint16` representation). If you need a larger vocab,
> change `'<i2'` to `'<u2'` in `write_tokens_npy()` and cast to `int32` in your
> training code before feeding the embedding table.

Sample output:
```
Reading 'corpus.txt'…
  148 bytes loaded.
BPE training: 20 merges on 148 tokens
  [20/20] token 275 = 'the' | freq=8 | tokens=118 (-30)
Encoding 148 chars...

── Stats ──────────────────────────────────────────────
  Input bytes     :          148
  Encoded tokens  :           98
  Compression     :        1.51x  (1.5 bytes/token)
  Vocab size      :          276
  Merges learned  :           20
───────────────────────────────────────────────────────
```

> **Note:** `paircraft_cli` currently calls `train()` and `encode()` (the
> naive O(N·M) implementations). Once `train_from_file()` and `encode_fast()`
> are implemented, the tool will switch to those automatically — the TODOs in
> `tools/tokenize.cpp` mark the two call sites.

---

## Project structure

```
paircraft/
├── CMakeLists.txt          ← library + paircraft_cli + GTest
├── include/
│   └── paircraft/
│       └── bpe.h           ← public API (types + BPETokenizer)
├── src/
│   └── bpe.cpp             ← library implementation (+ MmapView RAII)
├── tools/
│   └── tokenize.cpp        ← paircraft_cli entry point
└── tests/
    ├── CMakeLists.txt
    └── test_bpe.cpp        ← GTest suite
```

---

## API overview

```cpp
#include "paircraft/bpe.h"
using namespace paircraft;

BPETokenizer tok;

// Train on a small string (in-memory, O(N·M))
tok.train("the cat sat on the mat", 300);

// Train on a large file (memory-mapped, corpus stays on disk) [TODO]
tok.train_from_file("/data/corpus.txt", 32000);

// Encode
std::vector<TokenId> ids = tok.encode("the cat sat");

// Decode
std::string text = tok.decode(ids);

// Fast encoder (priority-queue, chunked) [TODO]
std::vector<TokenId> ids2 = tok.encode_fast(very_long_text);

// Persist [TODO]
tok.save("my_tokenizer.bin");
auto tok2 = BPETokenizer::load("my_tokenizer.bin");
```

---

## Test corpus

The test suite uses a hand-crafted paragraph chosen to make BPE merges
predictable and auditable:

```
"the cat sat on the mat near the flat hat. the cat ran and the rat sat.
 a fat cat and a fat rat sat on the mat. the cat chased the rat past the hat."
```

Key properties:
- `"the "` appears 8 times — near-certain first or second merge
- `"at"` appears in *cat / rat / sat / mat / flat / hat* (15 hits) — frequent subword
- Shared suffixes (`-at`, `-an`) make merge order deterministic and testable

---

## Python reference

`bpe.py` contains the original Python implementation this library mirrors:

| Python | C++ |
|--------|-----|
| `train()` | `BPETokenizer::train()` |
| `train_fast()` | `BPETokenizer::train_from_file()` |
| `encode()` | `BPETokenizer::encode()` |
| `_encode_chunk()` / `encode_fast()` | `BPETokenizer::encode_chunk()` / `encode_fast()` |
| `decode()` | `BPETokenizer::decode()` |
| `save()` / `load()` | `BPETokenizer::save()` / `load()` |
