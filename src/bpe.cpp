#include "paircraft/bpe.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <print>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace paircraft
{

    // ── Helpers ──────────────────────────────────────────────────────────────────

    namespace
    {

        std::size_t peak_rss_mb()
        {
#ifdef _WIN32
            PROCESS_MEMORY_COUNTERS pmc{};
            GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
            return pmc.PeakWorkingSetSize / (1024 * 1024);
#else
            struct rusage ru{};
            getrusage(RUSAGE_SELF, &ru);
            return static_cast<std::size_t>(ru.ru_maxrss) / 1024; // Linux: kB
#endif
        }

        double elapsed_since(std::chrono::steady_clock::time_point t0)
        {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0)
                .count();
        }

    } // namespace

    // ── MmapView ─────────────────────────────────────────────────────────────────

    BPETokenizer::MmapView::MmapView(const std::filesystem::path &path)
    {
#ifdef _WIN32
        HANDLE fh = CreateFileW(
            path.c_str(),
            GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fh == INVALID_HANDLE_VALUE)
            throw std::runtime_error("MmapView: cannot open file: " + path.string());

        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(fh, &sz))
        {
            CloseHandle(fh);
            throw std::runtime_error("MmapView: GetFileSizeEx failed");
        }
        size_ = static_cast<std::size_t>(sz.QuadPart);

        HANDLE mh = CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mh)
        {
            CloseHandle(fh);
            throw std::runtime_error("MmapView: CreateFileMapping failed");
        }

        void *ptr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
        if (!ptr)
        {
            CloseHandle(mh);
            CloseHandle(fh);
            throw std::runtime_error("MmapView: MapViewOfFile failed");
        }

        data_ = static_cast<const uint8_t *>(ptr);
        file_handle_ = fh;
        map_handle_ = mh;
#else
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("MmapView: cannot open file: " + path.string());

        struct stat st{};
        if (::fstat(fd, &st) < 0)
        {
            ::close(fd);
            throw std::runtime_error("MmapView: fstat failed");
        }
        size_ = static_cast<std::size_t>(st.st_size);

        void *ptr = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED)
        {
            ::close(fd);
            throw std::runtime_error("MmapView: mmap failed");
        }

        // Hint sequential access pattern to the kernel
        ::madvise(ptr, size_, MADV_SEQUENTIAL);

        data_ = static_cast<const uint8_t *>(ptr);
        fd_ = fd;
#endif
    }

    BPETokenizer::MmapView::~MmapView()
    {
#ifdef _WIN32
        if (data_)
            UnmapViewOfFile(const_cast<uint8_t *>(data_));
        if (map_handle_)
            CloseHandle(map_handle_);
        if (file_handle_)
            CloseHandle(file_handle_);
#else
        if (data_)
            ::munmap(const_cast<uint8_t *>(data_), size_);
        if (fd_ >= 0)
            ::close(fd_);
#endif
    }

    BPETokenizer::MmapView::MmapView(MmapView &&o) noexcept
        : data_(o.data_), size_(o.size_)
#ifdef _WIN32
          ,
          file_handle_(o.file_handle_), map_handle_(o.map_handle_)
#else
          ,
          fd_(o.fd_)
#endif
    {
        o.data_ = nullptr;
        o.size_ = 0;
#ifdef _WIN32
        o.file_handle_ = o.map_handle_ = nullptr;
#else
        o.fd_ = -1;
#endif
    }

    BPETokenizer::MmapView &BPETokenizer::MmapView::operator=(MmapView &&o) noexcept
    {
        if (this != &o)
        {
            this->~MmapView();
            new (this) MmapView(std::move(o));
        }
        return *this;
    }

    // ── BPETokenizer — private helpers ───────────────────────────────────────────

    void BPETokenizer::init_byte_vocab()
    {
        vocab_.reserve(256);
        for (int i = 0; i < 256; ++i)
            vocab_[static_cast<TokenId>(i)] = {static_cast<uint8_t>(i)};
    }

    PairFreqMap BPETokenizer::count_pairs(std::span<const TokenId> tokens)
    {
        PairFreqMap freq;
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i)
            ++freq[{tokens[i], tokens[i + 1]}];
        return freq;
    }

    std::vector<TokenId> BPETokenizer::merge_pair(std::span<const TokenId> tokens,
                                                  Pair pair, TokenId idx)
    {
        std::vector<TokenId> out;
        out.reserve(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            if (i + 1 < tokens.size() &&
                tokens[i] == pair.first && tokens[i + 1] == pair.second)
            {
                out.push_back(idx);
                ++i; // skip right element
            }
            else
            {
                out.push_back(tokens[i]);
            }
        }
        return out;
    }

    // ── BPETokenizer::train ───────────────────────────────────────────────────────

    void BPETokenizer::train(std::string text, int vocab_size)
    {
        assert(vocab_size >= 256);
        const int num_merges = vocab_size - 256;

        init_byte_vocab();
        merges_.reserve(num_merges);

        // Convert text to initial byte token sequence, then immediately free the
        // source string.  Mirrors Python train_fast()'s `del text` after encoding.
        // Without this, both the string (N bytes) and the token vector (2N bytes)
        // would be live simultaneously, tripling peak RAM for large corpora.
        std::vector<TokenId> tokens(text.size());
        for (std::size_t i = 0; i < text.size(); ++i)
            tokens[i] = static_cast<uint8_t>(text[i]);
        {
            std::string{}.swap(text);
        } // release source string memory

        std::println("BPE training: {} merges on {:L} tokens",
                     num_merges, tokens.size());
        const auto t0 = std::chrono::steady_clock::now();
        const int log_every = std::max(1, num_merges / 100);

        for (int i = 0; i < num_merges; ++i)
        {
            auto freq = count_pairs(tokens);
            if (freq.empty())
            {
                std::println("  [!] No pairs left at merge {} — stopping early.", i + 1);
                break;
            }

            // Pick the most frequent pair (ties broken by pair value for determinism)
            auto best = std::max_element(freq.begin(), freq.end(),
                                         [](const auto &a, const auto &b)
                                         { return a.second < b.second; });
            Pair pair = best->first;
            TokenId idx = static_cast<TokenId>(256 + i);

            tokens = merge_pair(tokens, pair, idx);

            merges_[pair] = idx;
            auto &bytes = vocab_[idx];
            bytes = vocab_[pair.first];
            const auto &r = vocab_[pair.second];
            bytes.insert(bytes.end(), r.begin(), r.end());

            if ((i + 1) % log_every == 0 || i == num_merges - 1)
            {
                std::println("  [{:>{}}/{}] token {} = {:?} | freq={} | tokens={:L}",
                             i + 1, std::to_string(num_merges).size(), num_merges,
                             idx,
                             std::string(vocab_[idx].begin(), vocab_[idx].end()),
                             best->second, tokens.size());
            }
        }
        std::println("BPE training done. Vocab size: {}  ({:.1f}s)",
                     vocab_.size(), elapsed_since(t0));
    }

    // ── BPETokenizer::train_from_file (memory-mapped, incremental) ───────────────

    void BPETokenizer::train_from_file(const std::filesystem::path &corpus_path,
                                       int vocab_size)
    {
        assert(vocab_size >= 256);

        // ── Stage 1: count word frequencies ──────────────────────────────────────
        // Stream the mmap'd corpus once. Split on ASCII whitespace.
        // Each unique word string maps to its total occurrence count.
        // Peak RAM here: only the word-frequency table (no corpus copy).

        std::println("Before training: [peak {} MB]", peak_rss_mb());

        MmapView view(corpus_path);
        auto bytes = view.as_span();

        std::unordered_map<std::string, uint32_t> word_freq;
        word_freq.reserve(1 << 22); // pre-allocate ~4M buckets; avoids rehashing

        std::println("after freq alloc: [peak {} MB]", peak_rss_mb());

        const char *word_start = nullptr;
        for (std::size_t i = 0; i <= bytes.size(); ++i)
        {
            const bool is_ws = (i == bytes.size()) ||
                               bytes[i] == ' ' || bytes[i] == '\t' ||
                               bytes[i] == '\n' || bytes[i] == '\r';
            const bool is_punct = !is_ws && std::ispunct(bytes[i]);

            if (is_ws || is_punct)
            {
                // emit accumulated alphabetic word
                if (word_start)
                {
                    ++word_freq[std::string(word_start,
                                            reinterpret_cast<const char *>(&bytes[i]))];
                    word_start = nullptr;
                }
                // emit punctuation as its own word type
                if (is_punct)
                    ++word_freq[std::string(1, static_cast<char>(bytes[i]))];
            }
            else
            {
                if (!word_start)
                    word_start = reinterpret_cast<const char *>(&bytes[i]);
            }
        }

        std::println("Stage 1: {:L} unique words  ({:L} bytes scanned)  [peak {} MB]",
                     word_freq.size(), bytes.size(), peak_rss_mb());

        // ── Stage 2: build word token inventory ──────────────────────────────────
        // Convert each unique word string to its initial byte token sequence.
        // freq is carried alongside so pair counts can be weighted later.
        // After this stage, word_freq is no longer needed and is freed.

        struct WordEntry
        {
            std::vector<TokenId> tokens; // current token sequence for this word type
            uint32_t freq;               // how many times this word appears in corpus
        };

        std::vector<WordEntry> words;
        words.reserve(word_freq.size());

        for (auto &[word, freq] : word_freq)
        {
            WordEntry e;
            e.freq = freq;
            e.tokens.reserve(word.size());
            for (unsigned char c : word)
                e.tokens.push_back(static_cast<TokenId>(c));
            words.push_back(std::move(e));
        }
        {
            decltype(word_freq){}.swap(word_freq);
        } // free the string map — no longer needed

        std::size_t total_tokens = 0;
        for (const auto &e : words)
            total_tokens += e.tokens.size();
        std::println("Stage 2: {:L} word types, {:L} tokens in inventory  [peak {} MB]",
                     words.size(), total_tokens, peak_rss_mb());

        // ── Stage 3: build pair frequency table ──────────────────────────────────
        // For each word type, count every adjacent token pair weighted by the
        // word's corpus frequency. A pair in a word seen 5000 times contributes
        // 5000 to the global count — same result as scanning the full corpus.

        PairFreqMap pair_freq;
        pair_freq.reserve(1 << 20); // pre-allocate ~1M buckets

        for (const auto &e : words)
        {
            for (std::size_t i = 0; i + 1 < e.tokens.size(); ++i)
                pair_freq[{e.tokens[i], e.tokens[i + 1]}] += e.freq;
        }

        std::println("Stage 3: {:L} unique pairs  [peak {} MB]",
                     pair_freq.size(), peak_rss_mb());

        // ── Stage 4: build position lists ────────────────────────────────────────
        // For each pair, record every (word_index, position) where it occurs.
        // This lets the merge loop update only the affected positions instead of
        // re-scanning all word types.

        struct Position
        {
            uint32_t word_idx; // index into words[]
            uint32_t pos;      // index into words[word_idx].tokens
        };

        std::unordered_map<Pair, std::vector<Position>, PairHash> pair_positions;
        pair_positions.reserve(pair_freq.size());

        for (std::size_t wi = 0; wi < words.size(); ++wi)
        {
            const auto &toks = words[wi].tokens;
            for (std::size_t i = 0; i + 1 < toks.size(); ++i)
                pair_positions[{toks[i], toks[i + 1]}].push_back(
                    {static_cast<uint32_t>(wi), static_cast<uint32_t>(i)});
        }

        std::println("Stage 4: position lists built  [peak {} MB]", peak_rss_mb());

        // ── Stage 5: build max-heap ───────────────────────────────────────────────
        // Each entry is (count, pair). std::priority_queue is a max-heap by
        // default, so the highest count surfaces first.
        // Entries are never updated in-place — when a count changes we push a new
        // entry and discard stale ones at pop time (lazy deletion).

        using HeapEntry = std::pair<std::size_t, Pair>;
        std::priority_queue<HeapEntry> heap;

        for (const auto &[pair, count] : pair_freq)
            heap.push({count, pair});

        std::println("Stage 5: heap built with {:L} entries  [peak {} MB]",
                     heap.size(), peak_rss_mb());

        // ── Stage 6: merge loop ───────────────────────────────────────────────────
        // Repeat for each of the (vocab_size - 256) requested merges:
        //   a. Pop heap until a non-stale entry is found.
        //   b. For each word that contains the pair: subtract its old pair counts,
        //      apply the merge, add new pair counts, push updated heap entries.
        //   c. Record the merge in merges_ and vocab_.

        init_byte_vocab();
        const int num_merges = vocab_size - 256;
        merges_.reserve(num_merges);

        const auto t0 = std::chrono::steady_clock::now();
        const int log_every = std::max(1, num_merges / 1000);

        for (int merge_i = 0; merge_i < num_merges; ++merge_i)
        {

            // — find best valid pair ——————————————————————————————————————————————
            while (!heap.empty())
            {
                auto [count, pair] = heap.top();
                auto it = pair_freq.find(pair);
                if (it != pair_freq.end() && it->second == count && count > 0)
                    break; // entry is current
                heap.pop();
            }
            if (heap.empty())
            {
                std::println("  [!] No pairs left at merge {} — stopping early.", merge_i + 1);
                break;
            }
            auto [best_count, best_pair] = heap.top();
            heap.pop();

            const TokenId new_idx = static_cast<TokenId>(256 + merge_i);

            // — record merge + vocab entry ————————————————————————————————————————
            merges_[best_pair] = new_idx;
            auto &new_bytes = vocab_[new_idx];
            new_bytes = vocab_[best_pair.first];
            const auto &rb = vocab_[best_pair.second];
            new_bytes.insert(new_bytes.end(), rb.begin(), rb.end());

            // — update affected words —————————————————————————————————————————————
            // Each word is processed at most once per merge (processed set).
            // For each affected word: remove its pair contributions, apply the
            // merge to its token sequence, then re-add the new pair contributions.
            std::unordered_set<uint32_t> processed;

            for (const auto &[wi, pos] : pair_positions[best_pair])
            {
                if (!processed.insert(wi).second)
                    continue;

                auto &toks = words[wi].tokens;

                // validate: word might no longer contain best_pair (stale entry)
                bool has_pair = false;
                for (std::size_t i = 0; i + 1 < toks.size(); ++i)
                    if (toks[i] == best_pair.first && toks[i + 1] == best_pair.second)
                    {
                        has_pair = true;
                        break;
                    }
                if (!has_pair)
                    continue;

                const uint32_t freq = words[wi].freq;

                // subtract this word's contribution to all current pair counts
                for (std::size_t i = 0; i + 1 < toks.size(); ++i)
                {
                    Pair p{toks[i], toks[i + 1]};
                    pair_freq[p] -= freq;
                    if (pair_freq[p] > 0)
                        heap.push({pair_freq[p], p}); // lazy: push updated count
                }

                // apply merge: replace every non-overlapping (best_pair) with new_idx
                std::vector<TokenId> new_toks;
                new_toks.reserve(toks.size());
                for (std::size_t i = 0; i < toks.size(); ++i)
                {
                    if (i + 1 < toks.size() &&
                        toks[i] == best_pair.first && toks[i + 1] == best_pair.second)
                    {
                        new_toks.push_back(new_idx);
                        ++i; // skip the right token
                    }
                    else
                    {
                        new_toks.push_back(toks[i]);
                    }
                }
                toks = std::move(new_toks);

                // add this word's contribution to all new pair counts
                for (std::size_t i = 0; i + 1 < toks.size(); ++i)
                {
                    Pair p{toks[i], toks[i + 1]};
                    pair_freq[p] += freq;
                    heap.push({pair_freq[p], p});
                    pair_positions[p].push_back({wi, static_cast<uint32_t>(i)});
                }
            }

            // position list for merged pair is now fully consumed
            pair_positions.erase(best_pair);
            pair_freq[best_pair] = 0;

            if ((merge_i + 1) % log_every == 0 || merge_i == num_merges - 1)
            {
                std::println("  [{:>{}}/{}] token {} = {:?} | freq={:L}",
                             merge_i + 1, std::to_string(num_merges).size(), num_merges,
                             new_idx,
                             std::string(vocab_[new_idx].begin(), vocab_[new_idx].end()),
                             best_count);
            }
        }

        std::println("BPE training done. Vocab size: {}  ({:.1f}s)",
                     vocab_.size(), elapsed_since(t0));
    }

    // ── BPETokenizer::add_special_token ──────────────────────────────────────────

    void BPETokenizer::add_special_token(std::string_view text, TokenId id)
    {
        special_tokens_[std::string(text)] = id;
        special_token_ids_[id] = std::string(text);
    }

    // ── Special token splitting ───────────────────────────────────────────────────
    // Splits `text` into alternating segments: plain text and special token IDs.
    // Result: pairs of (segment_text, special_id) where special_id = kDeletedSentinel
    // means the segment is plain text to be BPE-encoded, otherwise it IS the token.

    namespace
    {
        struct Segment
        {
            std::string_view text;
            TokenId          special_id; // kDeletedSentinel = plain text segment
        };

        std::vector<Segment> split_on_special(
            std::string_view text,
            const std::unordered_map<std::string, TokenId>& specials)
        {
            if (specials.empty())
                return {{text, kDeletedSentinel}};

            std::vector<Segment> result;
            std::size_t pos = 0;

            while (pos < text.size()) {
                // Find the earliest special token match from current position
                std::size_t best_pos = std::string_view::npos;
                const std::string* best_str = nullptr;
                TokenId best_id = kDeletedSentinel;

                for (const auto& [s, id] : specials) {
                    auto found = text.find(s, pos);
                    if (found != std::string_view::npos &&
                        (best_pos == std::string_view::npos || found < best_pos)) {
                        best_pos = found;
                        best_str = &s;
                        best_id  = id;
                    }
                }

                if (best_pos == std::string_view::npos) {
                    // no more special tokens — rest is plain text
                    result.push_back({text.substr(pos), kDeletedSentinel});
                    break;
                }

                // plain text segment before the special token
                if (best_pos > pos)
                    result.push_back({text.substr(pos, best_pos - pos), kDeletedSentinel});

                // the special token itself
                result.push_back({text.substr(best_pos, best_str->size()), best_id});
                pos = best_pos + best_str->size();
            }
            return result;
        }
    } // namespace

    // ── BPETokenizer::encode ─────────────────────────────────────────────────────

    std::vector<TokenId> BPETokenizer::encode(std::string_view text) const
    {
        // Handle special tokens: split text, BPE-encode plain segments only
        const auto segments = split_on_special(text, special_tokens_);
        if (segments.size() > 1 ||
            (!segments.empty() && segments[0].special_id != kDeletedSentinel))
        {
            std::vector<TokenId> out;
            for (const auto& seg : segments) {
                if (seg.special_id != kDeletedSentinel)
                    out.push_back(seg.special_id);
                else {
                    auto part = encode(seg.text); // recurse on plain segment
                    out.insert(out.end(), part.begin(), part.end());
                }
            }
            return out;
        }

        std::vector<TokenId> tokens(text.size());
        for (std::size_t i = 0; i < text.size(); ++i)
            tokens[i] = static_cast<uint8_t>(text[i]);

        while (tokens.size() >= 2)
        {
            auto freq = count_pairs(tokens);

            // Find the pair whose merge rank is lowest (i.e. was added first)
            Pair best{};
            TokenId best_rank = std::numeric_limits<TokenId>::max();
            bool found = false;
            for (const auto &[pair, _] : freq)
            {
                auto it = merges_.find(pair);
                if (it != merges_.end() && it->second < best_rank)
                {
                    best_rank = it->second;
                    best = pair;
                    found = true;
                }
            }
            if (!found)
                break;

            tokens = merge_pair(tokens, best, best_rank);
        }
        return tokens;
    }

    // ── BPETokenizer::encode_chunk ───────────────────────────────────────────────

    std::vector<TokenId> BPETokenizer::encode_chunk(
        std::span<const uint8_t> chunk) const
    {
        const std::size_t N = chunk.size();
        if (N == 0)
            return {};

        // ── Stage 1: linked list initialization ──────────────────────────────
        // Three parallel arrays of length N:
        //   vals[i] — current token at position i (0xFFFF = deleted)
        //   nxt[i]  — index of the next live position after i
        //   prv[i]  — index of the previous live position before i

        std::vector<TokenId> vals(N);
        std::vector<uint32_t> nxt(N);
        std::vector<uint32_t> prv(N);

        for (std::size_t i = 0; i < N; ++i)
        {
            vals[i] = static_cast<TokenId>(chunk[i]);
            nxt[i] = static_cast<uint32_t>(i + 1); // points past end for last node
            prv[i] = static_cast<uint32_t>(i - 1); // wraps for first node (never read)
        }

        // ── Stage 2: build candidate heap ────────────────────────────────────
        // For every adjacent pair, look it up in merges_. If found, push
        // (rank, position) to a min-heap — lowest rank surfaces first.
        // rank = result_token_id - 256  (earlier merge → smaller rank → higher priority)

        // min-heap: smallest rank at top
        using HeapEntry = std::pair<int, uint32_t>; // (rank, left_position)
        std::priority_queue<HeapEntry,
                            std::vector<HeapEntry>,
                            std::greater<HeapEntry>>
            heap;

        for (std::size_t i = 0; i + 1 < N; ++i)
        {
            auto it = merges_.find({vals[i], vals[i + 1]}); // find pair of tokens in merge map

            if (it != merges_.end())
                heap.push({it->second - 256, static_cast<uint32_t>(i)});
        }

        // ── Stage 3: priority queue merge loop ───────────────────────────────
        // Pop the lowest-rank entry. Validate it is still live — both tokens
        // must still match what the heap entry was pushed for. Apply the merge,
        // relink the list, then push any newly exposed neighbor pairs.

        while (!heap.empty())
        {
            auto [rank, pos] = heap.top();
            heap.pop();

            // right neighbour of pos
            const uint32_t rgt = nxt[pos];
            if (rgt >= N)
                continue; // pos is the last node, no right neighbour

            // stale check: tokens at pos and rgt must still match the merge
            const TokenId result = static_cast<TokenId>(256 + rank);
            auto it = merges_.find({vals[pos], vals[rgt]});
            if (it == merges_.end() || it->second != result)
                continue;

            // — apply merge ───────────────────────────────────────────────────
            vals[pos] = result; // left node takes the merged token
            vals[rgt] = 0xFFFF; // right node marked deleted

            // relink: skip over rgt
            nxt[pos] = nxt[rgt];
            if (nxt[rgt] < N)
                prv[nxt[rgt]] = pos;

            // — push newly exposed neighbor pairs ─────────────────────────────
            // left boundary: (vals[prv[pos]], vals[pos])
            if (pos > 0 && prv[pos] < N && vals[prv[pos]] != 0xFFFF)
            {
                auto lit = merges_.find({vals[prv[pos]], vals[pos]});
                if (lit != merges_.end())
                    heap.push({lit->second - 256, prv[pos]});
            }
            // right boundary: (vals[pos], vals[nxt[pos]])
            if (nxt[pos] < N)
            {
                auto rit = merges_.find({vals[pos], vals[nxt[pos]]});
                if (rit != merges_.end())
                    heap.push({rit->second - 256, pos});
            }
        }

        // ── Stage 4: collect output ──────────────────────────────────────────
        // Walk the linked list from position 0, skipping deleted nodes.
        // Every surviving vals[i] is a final token.

        std::vector<TokenId> out;
        out.reserve(N); // worst case: no merges applied
        for (uint32_t i = 0; i < N; i = nxt[i])
            if (vals[i] != 0xFFFF)
                out.push_back(vals[i]);

        return out;
    }

    // ── BPETokenizer::encode_fast ────────────────────────────────────────────────

    std::vector<TokenId> BPETokenizer::encode_fast(std::string_view text,
                                                   std::size_t chunk_size) const
    {
        // Encode plain text (no special tokens) in fixed-size chunks.
        // When log=true, prints a progress line per chunk (used only at the top level).
        auto encode_plain = [&](std::string_view plain, bool log) -> std::vector<TokenId>
        {
            std::vector<TokenId> result;
            result.reserve(plain.size());

            const std::size_t total_bytes = plain.size();
            if (total_bytes == 0) return result;

            const std::size_t total_chunks =
                (total_bytes + chunk_size - 1) / chunk_size;
            const int width =
                static_cast<int>(std::to_string(total_chunks).size());
            const auto t0 = std::chrono::steady_clock::now();
            std::size_t chunk_idx = 0;

            for (std::size_t offset = 0; offset < total_bytes;
                 offset += chunk_size, ++chunk_idx)
            {
                const std::size_t len = std::min(chunk_size, total_bytes - offset);
                const auto *ptr =
                    reinterpret_cast<const uint8_t *>(plain.data() + offset);
                auto chunk_out = encode_chunk({ptr, len});
                result.insert(result.end(), chunk_out.begin(), chunk_out.end());

                if (log)
                {
                    const double pct = 100.0 *
                        static_cast<double>(offset + len) /
                        static_cast<double>(total_bytes);
                    std::print("\r  chunk {:>{}} / {}  ({:.1f}%)  {:L} tokens  {:.1f}s",
                               chunk_idx + 1, width, total_chunks,
                               pct, result.size(), elapsed_since(t0));
                }
            }
            if (log) std::println("");
            return result;
        };

        // Split on special tokens first.
        // Sub-segments are encoded without logging (log=false) to avoid
        // one "chunk 1/1" line per document when the corpus is split on <|endoftext|>.
        const auto segments = split_on_special(text, special_tokens_);
        if (segments.size() > 1 ||
            (!segments.empty() && segments[0].special_id != kDeletedSentinel))
        {
            std::vector<TokenId> out;
            for (const auto &seg : segments)
            {
                if (seg.special_id != kDeletedSentinel)
                    out.push_back(seg.special_id);
                else
                {
                    auto part = encode_plain(seg.text, /*log=*/false);
                    out.insert(out.end(), part.begin(), part.end());
                }
            }
            return out;
        }

        // No special tokens — encode the whole text with progress logging.
        return encode_plain(text, /*log=*/true);
    }

    // ── BPETokenizer::decode ─────────────────────────────────────────────────────

    std::string BPETokenizer::decode(std::span<const TokenId> ids) const
    {
        std::string out;
        for (TokenId id : ids)
        {
            // special tokens decode to their literal text
            auto sit = special_token_ids_.find(id);
            if (sit != special_token_ids_.end()) {
                out += sit->second;
                continue;
            }

            auto it = vocab_.find(id);
            if (it == vocab_.end())
                throw std::out_of_range(std::format("decode: unknown token id {}", id));
            for (uint8_t b : it->second)
                out += static_cast<char>(b);
        }
        return out;
    }

    // ── BPETokenizer::save / load ────────────────────────────────────────────────

    void BPETokenizer::save(const std::filesystem::path &path) const
    {
        std::ofstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("save: cannot open " + path.string());

        auto write = [&](const void *data, std::size_t n)
        {
            f.write(static_cast<const char *>(data), static_cast<std::streamsize>(n));
        };
        auto write_u32 = [&](uint32_t v)
        { write(&v, 4); };
        auto write_u16 = [&](uint16_t v)
        { write(&v, 2); };

        // header: magic + version
        // v1 = no special tokens section; v2 = has special tokens section
        write("BPEC", 4);
        write_u32(2);

        // merges: [4B count] then each [2B a][2B b][2B result]
        write_u32(static_cast<uint32_t>(merges_.size()));
        for (const auto &[pair, idx] : merges_)
        {
            write_u16(pair.first);
            write_u16(pair.second);
            write_u16(idx);
        }

        // vocab: [4B count] then each [2B id][4B byte_len][bytes]
        write_u32(static_cast<uint32_t>(vocab_.size()));
        for (const auto &[id, bytes] : vocab_)
        {
            write_u16(id);
            write_u32(static_cast<uint32_t>(bytes.size()));
            write(bytes.data(), bytes.size());
        }

        // special tokens: [4B count] then each [2B id][4B text_len][text bytes]
        write_u32(static_cast<uint32_t>(special_tokens_.size()));
        for (const auto& [text, id] : special_tokens_) {
            write_u16(id);
            write_u32(static_cast<uint32_t>(text.size()));
            write(text.data(), text.size());
        }

        if (!f)
            throw std::runtime_error("save: write error");
        std::println("Saved model to '{}'  ({} merges, {} vocab entries, {} special tokens)",
                     path.string(), merges_.size(), vocab_.size(), special_tokens_.size());
    }

    BPETokenizer BPETokenizer::load(const std::filesystem::path &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("load: cannot open " + path.string());

        auto read = [&](void *data, std::size_t n)
        {
            f.read(static_cast<char *>(data), static_cast<std::streamsize>(n));
            if (!f)
                throw std::runtime_error("load: unexpected end of file");
        };
        auto read_u32 = [&]()
        { uint32_t v{}; read(&v, 4); return v; };
        auto read_u16 = [&]()
        { uint16_t v{}; read(&v, 2); return v; };

        // header
        char magic[4]{};
        read(magic, 4);
        if (std::string_view(magic, 4) != "BPEC")
            throw std::runtime_error("load: not a paircraft model file");
        const uint32_t version = read_u32();
        if (version != 1 && version != 2)
            throw std::runtime_error("load: unsupported file version");

        BPETokenizer tok;

        // merges
        const uint32_t num_merges = read_u32();
        tok.merges_.reserve(num_merges);
        for (uint32_t i = 0; i < num_merges; ++i)
        {
            const uint16_t a = read_u16();
            const uint16_t b = read_u16();
            const uint16_t idx = read_u16();
            tok.merges_[{a, b}] = idx;
        }

        // vocab
        const uint32_t vocab_sz = read_u32();
        for (uint32_t i = 0; i < vocab_sz; ++i)
        {
            const uint16_t id = read_u16();
            const uint32_t len = read_u32();
            std::vector<uint8_t> bytes(len);
            read(bytes.data(), len);
            tok.vocab_[id] = std::move(bytes);
        }

        // special tokens (v2+ only; v1 files have none)
        if (version < 2) return tok;
        const uint32_t num_special = read_u32();
        for (uint32_t i = 0; i < num_special; ++i) {
            const uint16_t id  = read_u16();
            const uint32_t len = read_u32();
            std::string text(len, '\0');
            read(text.data(), len);
            tok.special_tokens_[text] = id;
            tok.special_token_ids_[id] = text;
        }

        std::println("Loaded model from '{}'  ({} merges, {} vocab entries, {} special tokens)",
                     path.string(), tok.merges_.size(), tok.vocab_.size(),
                     tok.special_tokens_.size());
        return tok;
    }

    // ── BPETokenizer::print_vocab ────────────────────────────────────────────────

    void BPETokenizer::print_vocab(int max_tokens) const
    {
        std::vector<std::pair<TokenId, std::vector<uint8_t>>> entries(
            vocab_.begin(), vocab_.end());
        std::sort(entries.begin(), entries.end(),
                  [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        if (max_tokens >= 0 && static_cast<std::size_t>(max_tokens) < entries.size())
            entries.resize(max_tokens);

        const int id_w = std::to_string(entries.back().first).size();
        for (const auto &[id, bytes] : entries)
        {
            // Hex representation
            std::string hex;
            hex.reserve(bytes.size() * 2);
            for (uint8_t b : bytes)
                hex += std::format("{:02x}", b);

            // Printable representation (replace non-printable with \xNN)
            std::string display;
            for (uint8_t b : bytes)
            {
                if (b >= 0x20 && b < 0x7F)
                    display += static_cast<char>(b);
                else
                    display += std::format("\\x{:02x}", b);
            }
            std::println("  {:>{}}  {:<24}  {:?}", id, id_w, hex, display);
        }
    }

} // namespace paircraft
