import time
from collections import Counter
import numpy as np

def get_stats(ids):
    return dict(Counter(zip(ids, ids[1:])))
    # counts = {}
    # for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
    #     counts[pair] = counts.get(pair, 0) + 1 #get returns 0 if pair is not found
    # return counts

def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> bytes


    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        tokens = list(text.encode("utf-8"))
        print(f"BPE training: {num_merges} merges on {len(tokens):,} tokens")
        _train_start = time.time()

        log_every = max(1, num_merges // 100)

        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        _interval_start = time.time()
        _tokens_at_interval_start = len(tokens)

        for i in range(num_merges):
            _iter_start = time.time()

            stats = get_stats(tokens)
            _stats_time = time.time() - _iter_start

            if not stats:
                print(f"  [!] No pairs left at merge {i+1} — stopping early.")
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i

            _merge_start = time.time()
            tokens = merge(tokens, pair, idx)
            _merge_time = time.time() - _merge_start

            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            _iter_time = time.time() - _iter_start

            # warn if a single iteration is unusually slow
            if _iter_time > 2.0:
                print(f"  [SLOW] merge {i+1}: {_iter_time:.2f}s total "
                      f"(get_stats={_stats_time:.2f}s, merge={_merge_time:.2f}s) "
                      f"| tokens={len(tokens):,} | pair_freq={stats[pair]} "
                      f"| unique_pairs={len(stats):,}")

            if (i + 1) % log_every == 0 or i == num_merges - 1:
                _interval_time = time.time() - _interval_start
                _tokens_merged = _tokens_at_interval_start - len(tokens)
                print(f"  [{i+1:>{len(str(num_merges))}}/{num_merges}] token {idx} = {self.vocab[idx]!r} "
                      f"| freq={stats[pair]} | unique_pairs={len(stats):,} "
                      f"| tokens={len(tokens):,} (-{_tokens_merged:,}) "
                      f"| last {log_every} merges: {_interval_time:.1f}s "
                      f"({_interval_time/log_every*1000:.0f}ms/merge)")
                _interval_start = time.time()
                _tokens_at_interval_start = len(tokens)

        _train_elapsed = time.time() - _train_start
        print(f"BPE training done. Vocab size: {len(self.vocab)}  ({_train_elapsed:.1f}s)")

    def train_fast(self, text: str, vocab_size: int):
        """
        Incremental BPE training — O(N + M·k·log P) vs O(N·M) for train().

        Memory layout (critical for large corpora):
          - vals    np.uint16  — token values 0..vocab_size-1, 0xFFFF=deleted (2 B/token)
          - prev    np.int32   — linked-list predecessor                       (4 B/token)
          - nxt     np.int32   — linked-list successor                         (4 B/token)
          - pair_positions values: array.array('i')                            (4 B/pos)
          Total ≈ 14 B/token + init-time argsort peak (~12 B/token)
          (no separate deleted[] array: sentinel vals[j]=0xFFFF replaces it)
          Python lists would cost ~140 B/token — causing MemoryError.

        Correctness note on adjacent pairs (e.g. A B A B with pair=(A,B)):
          Positions are processed left→right.  When pos=0 is merged its right
          neighbour update creates a transient (idx, A) entry.  When pos=2 is
          then processed, its left-neighbour update decrements (idx, A) and
          increments (idx, idx) — the correct final state.
        """
        import heapq
        from array import array as carray

        assert vocab_size >= 256
        assert vocab_size <= 65535, "vocab_size must be ≤ 65535 (65535 = 0xFFFF is the deletion sentinel in vals)"
        num_merges = vocab_size - 256

        raw = text.encode("utf-8")
        del text   # free the (potentially large) source string immediately
        n   = len(raw)
        assert n < 2**31, (
            f"Corpus has {n:,} bytes; int32 indices support up to ~2.1 B tokens. "
            "Split the corpus or use a smaller dataset."
        )
        print(f"BPE training (fast): {num_merges} merges on {n:,} tokens")
        _train_start = time.time()

        log_every = max(1, num_merges // 100)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        # --- doubly linked list (numpy arrays, not Python lists) ---
        # uint16: token IDs 0..65534; vocab_size ≤ 65535 required
        #         0xFFFF (65535) is the deletion sentinel — never a valid token ID
        # int32:  indices  -1..n;  n < 2^31 asserted above
        # No separate deleted[] array: merged-away nodes get vals[j] = 0xFFFF.
        vals    = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)  # writable copy
        del raw    # frombuffer view was temporary; raw no longer needed (saves 1 B/token)
        prev    = np.arange(-1, n - 1, dtype=np.int32)   # prev[0] = -1 sentinel
        nxt     = np.arange(1,  n + 1, dtype=np.int32)   # nxt[n-1] = n  sentinel
        surviving = n

        # --- pair counts (plain dict, at most ~65536 initial byte pairs) ---
        counts = {}

        # --- pair_positions: array.array('i') uses 4 B/entry vs 36 B for Python list ---
        pair_positions = {}

        # Vectorised initialisation: argsort positions by pair value so we can
        # slice out each pair's positions in one pass (avoids O(N) Python loop).
        print(f"  Initialising pair counts ({n:,} tokens)…")
        _init_t = time.time()

        # At init time all token values are raw bytes (0-255), so pairs fit in
        # uint16 (max = 255*256+255 = 65535 = 2^16-1).  Two advantages vs uint32:
        #   • 2 B/token instead of 4 B (saves ~3 GB for a 1.5 B-token corpus)
        #   • numpy's radix argsort uses 1 pass per byte → 2 passes for uint16
        #     vs 4 passes for uint32, roughly halving sort time on large corpora
        print("    [1/4] encoding pairs…", end=" ", flush=True)
        pair_ints = np.empty(n - 1, dtype=np.uint16)
        np.multiply(vals[:-1], 256, out=pair_ints, casting='unsafe')  # left * 256
        np.add(pair_ints, vals[1:], out=pair_ints, casting='unsafe')  # + right
        print(f"done ({time.time() - _init_t:.1f}s)")

        # argsort returns int64; immediately cast to int32 (n < 2^31) to halve memory
        _t = time.time()
        print("    [2/4] sorting…", end=" ", flush=True)
        order_i32  = pair_ints.argsort(kind='stable').astype(np.int32)
        sorted_pi  = pair_ints[order_i32]
        del pair_ints
        print(f"done ({time.time() - _t:.1f}s)")

        # find group boundaries
        _t = time.time()
        print("    [3/4] finding boundaries…", end=" ", flush=True)
        diffs      = np.empty(len(sorted_pi) + 1, dtype=np.bool_)
        diffs[0]   = True
        diffs[-1]  = True
        diffs[1:-1] = sorted_pi[1:] != sorted_pi[:-1]
        boundaries = np.flatnonzero(diffs)
        del diffs
        print(f"done ({time.time() - _t:.1f}s, {len(boundaries) - 1:,} unique pairs)")

        _t = time.time()
        print("    [4/4] building position lists…", end=" ", flush=True)
        for k in range(len(boundaries) - 1):
            s, e   = int(boundaries[k]), int(boundaries[k + 1])
            pi     = int(sorted_pi[s])
            a, b   = pi >> 8, pi & 0xFF
            key    = (a, b)
            counts[key] = e - s
            chunk  = order_i32[s:e]
            pa     = carray('i')
            pa.frombytes(chunk.tobytes())
            pair_positions[key] = pa

        del order_i32, sorted_pi, boundaries
        print(f"done ({time.time() - _t:.1f}s)")
        print(f"  Init complete ({time.time() - _init_t:.1f}s total, {len(counts):,} unique pairs)")

        # --- max-heap: (-count, pair) with lazy staleness detection ---
        # pair counts are monotonically non-increasing (new merged tokens are
        # fresh IDs that cannot reconstitute an old pair), so a popped entry
        # (-c, p) is stale iff counts.get(p, 0) != c.
        heap = [(-cnt, pair) for pair, cnt in counts.items()]
        heapq.heapify(heap)

        _interval_start           = time.time()
        _tokens_at_interval_start = surviving

        for merge_i in range(num_merges):
            # find highest-frequency valid pair
            best_pair = None
            best_cnt  = 0
            while heap:
                neg_cnt, candidate = heapq.heappop(heap)
                cur = counts.get(candidate, 0)
                if cur > 0 and cur == -neg_cnt:
                    best_pair = candidate
                    best_cnt  = cur
                    break
            if best_pair is None:
                print(f"  [!] No pairs left at merge {merge_i + 1} — stopping early.")
                break
            idx      = 256 + merge_i
            bp0, bp1 = best_pair
            n_merged = 0
            for pos in pair_positions.pop(best_pair, carray('i')):
                
                # pos is a Python int (carray iteration yields Python ints)
                j = int(nxt[pos])
                # validate: both nodes alive with correct values
                # deleted nodes have vals == 0xFFFF, which never equals a valid bp0/bp1
                if vals[pos] != bp0 or j >= n or vals[j] != bp1:
                    continue

                k = int(nxt[j])   # right-right neighbour (may be sentinel n)

                # update left-neighbour pair
                p = int(prev[pos])
                if p >= 0 and vals[p] != 0xFFFF:
                    vp     = int(vals[p])
                    lp_old = (vp, bp0)
                    new_c  = counts.get(lp_old, 0) - 1
                    counts[lp_old] = new_c
                    if new_c <= 0:
                        pair_positions.pop(lp_old, None)  # pair is dead; free its stale positions
                    lp_new = (vp, idx)
                    counts[lp_new] = counts.get(lp_new, 0) + 1
                    pair_positions.setdefault(lp_new, carray('i')).append(p)
                    heapq.heappush(heap, (-counts[lp_new], lp_new))

                # update right-neighbour pair
                if k < n and vals[k] != 0xFFFF:
                    vk     = int(vals[k])
                    rp_old = (bp1, vk)
                    new_c  = counts.get(rp_old, 0) - 1
                    counts[rp_old] = new_c
                    if new_c <= 0:
                        pair_positions.pop(rp_old, None)  # pair is dead; free its stale positions
                    rp_new = (idx, vk)
                    counts[rp_new] = counts.get(rp_new, 0) + 1
                    pair_positions.setdefault(rp_new, carray('i')).append(pos)
                    heapq.heappush(heap, (-counts[rp_new], rp_new))

                # perform merge: absorb j into pos
                vals[pos]  = idx
                vals[j] = np.uint16(0xFFFF)  # mark j deleted via sentinel
                nxt[pos]   = k
                if k < n:
                    prev[k] = pos
                n_merged += 1

            # clear any residual count (stale pair_positions entries weren't visited)
            counts[best_pair] = 0
            surviving -= n_merged
            self.merges[best_pair] = idx
            self.vocab[idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if (merge_i + 1) % log_every == 0 or merge_i == num_merges - 1:
                _interval_time = time.time() - _interval_start
                _tokens_merged = _tokens_at_interval_start - surviving
                active_pairs   = sum(1 for c in counts.values() if c > 0)
                print(f"  [{merge_i+1:>{len(str(num_merges))}}/{num_merges}] "
                      f"token {idx} = {self.vocab[idx]!r} "
                      f"| freq={best_cnt} | unique_pairs={active_pairs:,} "
                      f"| tokens={surviving:,} (-{_tokens_merged:,}) "
                      f"| last {log_every} merges: {_interval_time:.1f}s "
                      f"({_interval_time / log_every * 1000:.0f}ms/merge)")
                _interval_start           = time.time()
                _tokens_at_interval_start = surviving

        _train_elapsed = time.time() - _train_start
        print(f"BPE training (fast) done. Vocab size: {len(self.vocab)}  ({_train_elapsed:.1f}s)")

    def encode(self, text: str) -> list[int]:
        print(f"Encoding {len(text):,} chars...")
        _t = time.time()
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = merge(tokens, pair, self.merges[pair])
        print(f"Encoding done: {len(tokens):,} tokens  ({time.time() - _t:.1f}s)")
        return np.array(tokens, dtype=np.int16)

    def _encode_chunk(self, raw: bytes) -> np.ndarray:
        """
        Encode a single raw-bytes chunk. Called by encode_fast() per chunk.

        Memory layout per n input bytes:
          vals  np.uint16  token values; 0xFFFF = deleted sentinel   (2 B/token)
          prev  np.int32   linked-list predecessor                    (4 B/token)
          nxt   np.int32   linked-list successor                      (4 B/token)
          heap  list[int]  packed int64 entries: rank<<32|pos         (28 B/entry)
          out   np.uint16  pre-allocated result buffer                (2 B/token)
        Peak ≈ 12 B/token + heap (heap shrinks as merges are applied).
        """
        import heapq
        n = len(raw)
        if n <= 1:
            return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)

        # doubly linked list; 0xFFFF sentinel marks deleted nodes
        vals = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
        prev = np.arange(-1, n - 1, dtype=np.int32)
        nxt  = np.arange(1,  n + 1, dtype=np.int32)

        # Heap entries packed as a single Python int: rank<<32 | pos.
        # A plain int costs ~28 B vs ~112 B for a (rank, pos) tuple.
        # rank fits in 16 bits (≤65534), pos in 32 bits (chunk_size < 2^31).
        heap = [(self.merges[pair] << 32 | i)
                for i in range(n - 1)
                if (pair := (int(vals[i]), int(vals[i + 1]))) in self.merges]
        heapq.heapify(heap)

        while heap:
            entry = heapq.heappop(heap)
            rank  = entry >> 32
            i     = entry & 0xFFFF_FFFF

            # discard stale: node deleted or pair no longer matches rank
            if vals[i] == 0xFFFF:
                continue
            j = int(nxt[i])
            if j >= n or vals[j] == 0xFFFF:
                continue
            if self.merges.get((int(vals[i]), int(vals[j]))) != rank:
                continue

            # apply merge: absorb j into i, mark j deleted via sentinel
            vals[i] = np.uint16(rank)
            vals[j] = np.uint16(0xFFFF)
            k = int(nxt[j])
            nxt[i] = k
            if k < n:
                prev[k] = i

            # push new pairs at the two newly formed boundaries
            pi = int(prev[i])
            if pi >= 0:
                new_pair = (int(vals[pi]), int(vals[i]))
                if new_pair in self.merges:
                    heapq.heappush(heap, self.merges[new_pair] << 32 | pi)
            if k < n:
                new_pair = (int(vals[i]), int(vals[k]))
                if new_pair in self.merges:
                    heapq.heappush(heap, self.merges[new_pair] << 32 | i)

        # collect surviving tokens via linked-list traversal
        out = np.empty(n, dtype=np.uint16)
        k = 0
        i = 0
        while i < n:
            out[k] = vals[i]
            k += 1
            i = int(nxt[i])
        return out[:k]

    def encode_fast(self, text: str, chunk_size: int = 10_000_000) -> np.ndarray:
        """
        Priority-queue BPE encoder. O(n log m) vs O(n * m) for encode().

        For large inputs the text is split into chunks of `chunk_size` bytes
        (default 10 MB) so that peak memory stays bounded regardless of corpus
        size.  Merges cannot span chunk boundaries, which is acceptable for
        document-separated corpora; increase chunk_size if boundary effects
        matter.

        Peak memory per chunk ≈ 12 B/token + heap entries (~28 B each).
        At chunk_size=10M: ~400 MB worst-case per chunk.
        """
        print(f"Encoding (fast) {len(text):,} chars...")
        _t = time.time()

        raw = text.encode("utf-8")
        del text
        n = len(raw)

        if n <= chunk_size:
            result = self._encode_chunk(raw)
            del raw
        else:
            n_chunks = (n + chunk_size - 1) // chunk_size
            parts = []
            for ci in range(n_chunks):
                s = ci * chunk_size
                chunk = raw[s : s + chunk_size]
                parts.append(self._encode_chunk(bytes(chunk)))
                if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
                    print(f"  chunk {ci + 1}/{n_chunks} done  ({time.time() - _t:.0f}s elapsed)")
            del raw
            result = np.concatenate(parts)

        print(f"Encoding (fast) done: {len(result):,} tokens  ({time.time() - _t:.1f}s)")
        return result

    def decode(self, ids: list[int]) -> str:
        print(f"Decoding {len(ids):,} tokens...")
        _t = time.time()
        tokens = b"".join(self.vocab[idx] for idx in ids)
        result = tokens.decode("utf-8", errors="replace")
        print(f"Decoding done: {len(result):,} chars  ({time.time() - _t:.1f}s)")
        return result

    def print_vocab(self, max_tokens: int = None):
        """Print the vocabulary, one token per line.

        Merge tokens show their constituent bytes in angle-bracket hex;
        raw bytes show their printable ASCII character when available.
        """
        tokens = sorted(self.vocab.items())
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        id_w = len(str(tokens[-1][0]))
        for idx, byt in tokens:
            text = byt.decode("utf-8", errors="replace")
            # show non-printable / replacement chars as hex escape
            display = "".join(
                c if c.isprintable() and c != "\ufffd" else f"\\x{ord(c):02x}"
                for c in text
            )
            print(f"  {idx:{id_w}d}  {byt.hex():<24s}  {display!r}")

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'merges': self.merges, 'vocab': self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        import pickle
        obj = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj.merges = data['merges']
        obj.vocab = data['vocab']
        return obj
