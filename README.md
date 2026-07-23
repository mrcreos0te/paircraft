# paircraft

A C++23 Byte-Pair Encoding (BPE) tokenizer library designed for large language
model pre-training pipelines where corpora routinely exceed available RAM.

---

## Features

- Memory-mapped training — corpus stays on disk, peak RAM ~3–4 GB for an 8 GB corpus
- Priority-queue encoder — O(N log N) per chunk instead of naive O(N·M)
- Binary `.bpec` model format — fast save/load
- Special token support (`<|endoftext|>` and others)
- NumPy-compatible `.npy` output (`dtype=int16`)

---

## Build

Requirements: CMake ≥ 3.25, a C++23 compiler (GCC 13+, Clang 17+, MSVC 19.38+).
GTest is fetched automatically at configure time.

### Windows (MSVC)

```bat
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Executables land in `build\Release\exe\`.

### Linux / macOS

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Executables land in `build/`.

---

## CLI usage

### Train on a corpus and save the model

```bat
paircraft_cli corpus.txt --vocab-size 50176 --save-model model.bpec
```

`<|endoftext|>` is automatically registered as a special token. The training
loop reserves one vocab slot for it, so the final vocab size equals exactly
`--vocab-size`.

### Train, encode, and write tokens

```bat
paircraft_cli corpus.txt --vocab-size 50176 --save-model model.bpec --output tokens.npy
```

### Encode with an existing model (skip retraining)

```bat
paircraft_cli corpus.txt --load-model model.bpec --output tokens.npy
```

### Decode tokens back to text

```bat
paircraft_cli tokens.npy --decode --load-model model.bpec --output decoded.txt
```

Omit `--output` to print to stdout.

### Inspect the vocabulary

```bat
# Print the 20 longest tokens
paircraft_cli --load-model model.bpec --print-longest 20

# Print the first 50 vocab entries (by ID)
paircraft_cli corpus.txt --load-model model.bpec --print-vocab 50
```

### All options

```
Usage: paircraft_cli <corpus.txt> [options]

Options:
  --vocab-size N      Vocabulary size (default: 512, min: 256)
  --output FILE       Encode: write .npy token IDs / Decode: write text file
  --save-model FILE   Save trained model to .bpec file after training
  --load-model FILE   Load model from .bpec file instead of training
  --decode            Decode mode: input is .npy tokens, output is text
  --print-vocab N     Print first N vocab entries after training
  --print-longest N   Print N longest tokens from a loaded model
  --help              Show this message
```

---

## Output format

Tokens are written as a standard NumPy v1.0 `.npy` file — `dtype=int16`,
shape `(N,)`. Load in Python with:

```python
import numpy as np
tokens = np.load("tokens.npy")          # int16, shape (N,)
tokens = tokens.astype(np.int64)        # cast per-batch before embedding lookup
```

> **Note:** `int16` supports vocab sizes up to 32 767 without sign issues.
> For larger vocabs the bit pattern is still correct for values up to 65 535 —
> cast to `uint16` or `int32` before use.

---

## Model format (`.bpec`)

Binary format, little-endian:

```
[4B]  magic "BPEC"
[4B]  version (uint32) — currently 2
[4B]  merge count
      per merge: [2B a][2B b][2B result]
[4B]  vocab entry count
      per entry: [2B id][4B byte_len][bytes]
[4B]  special token count
      per entry: [2B id][4B text_len][text bytes]
```

Version 1 files (no special tokens section) are still loadable.

---

## Special tokens

`<|endoftext|>` is registered on every fresh training run with ID =
`vocab_size - 1`. To add more special tokens, extend the `kSpecialTokens`
array in [tools/tokenize.cpp](tools/tokenize.cpp):

```cpp
static constexpr std::string_view kSpecialTokens[] = {
    "<|endoftext|>",
    "<|pad|>",     // <- just add here
};
```

The training loop automatically reserves the right number of slots.

---

## Project structure

```
paircraft/
├── CMakeLists.txt
├── include/
│   └── paircraft/
│       └── bpe.h           <- public API
├── src/
│   └── bpe.cpp             <- library implementation
├── tools/
│   └── tokenize.cpp        <- paircraft_cli entry point
└── tests/
    └── test_bpe.cpp        <- GTest suite
```

---

## API overview

```cpp
#include "paircraft/bpe.h"
using namespace paircraft;

BPETokenizer tok;

// Train on a large file (memory-mapped, corpus stays on disk)
tok.train_from_file("/data/corpus.txt", 50176);

// Register a special token
tok.add_special_token("<|endoftext|>", tok.vocab_size());

// Encode (fast, chunked, priority-queue)
std::vector<TokenId> ids = tok.encode_fast(very_long_text);

// Decode
std::string text = tok.decode(ids);

// Persist
tok.save("model.bpec");
auto tok2 = BPETokenizer::load("model.bpec");
```

---

## Python reference

`bpe.py` contains the original Python implementation this library mirrors:

| Python | C++ |
|--------|-----|
| `train_fast()` | `BPETokenizer::train_from_file()` |
| `encode_fast()` / `_encode_chunk()` | `BPETokenizer::encode_fast()` / `encode_chunk()` |
| `decode()` | `BPETokenizer::decode()` |
| `save()` / `load()` | `BPETokenizer::save()` / `BPETokenizer::load()` |
