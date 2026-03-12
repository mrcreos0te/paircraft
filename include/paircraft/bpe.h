#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace paircraft
{

    // ── Core types ───────────────────────────────────────────────────────────────

    /// Token identifier. uint16 covers vocab ≤ 65 534; 0xFFFF is the
    /// deletion sentinel used by the linked-list representation during merges.
    using TokenId = uint16_t;

    static constexpr TokenId kDeletedSentinel = 0xFFFFu;

    /// A consecutive token pair — the fundamental unit of BPE.
    struct Pair
    {
        TokenId first{};
        TokenId second{};
        bool operator==(const Pair &) const noexcept = default;
        bool operator<(const Pair &o) const noexcept
        {
            return first != o.first ? first < o.first : second < o.second;
        }
    };

    struct PairHash
    {
        std::size_t operator()(Pair p) const noexcept
        {
            // Combine two 16-bit values into one 32-bit key, then mix.
            std::uint32_t v = (std::uint32_t{p.first} << 16) | p.second;
            v ^= v >> 16;
            v *= 0x45d9f3bu;
            v ^= v >> 16;
            return v;
        }
    };

    using PairFreqMap = std::unordered_map<Pair, std::size_t, PairHash>;
    using MergeMap = std::unordered_map<Pair, TokenId, PairHash>; // third element - we need to provide hash for a bucket as we are storing our own struct
    // vocab: token id → raw UTF-8 bytes it represents
    using VocabMap = std::unordered_map<TokenId, std::vector<std::uint8_t>>;

    // ── BPETokenizer ─────────────────────────────────────────────────────────────

    class BPETokenizer
    {
    public:
        BPETokenizer() = default;

        // ── Training ─────────────────────────────────────────────────────────────

        /// In-memory training. Mirrors Python train().
        /// Takes the text by value so the source string can be freed immediately
        /// after the initial byte→token conversion, halving peak RAM.
        /// Call with std::move(text) from the caller to avoid a copy.
        /// std::string_view callers must construct explicitly: train(std::string(sv), n)
        /// because std::string's string_view constructor is explicit in C++17.
        /// Complexity: O(N · M) — suitable for small corpora / unit tests.
        void train(std::string text, int vocab_size);

        /// Memory-mapped training. Mirrors Python train_fast().
        /// The corpus file is mapped into virtual address space; only the
        /// pair-frequency table lives in RAM — handles 16 GB+ corpora.
        /// Complexity: O(N + M · k · log P)
        void train_from_file(const std::filesystem::path &corpus_path,
                             int vocab_size);

        // ── Encoding ─────────────────────────────────────────────────────────────

        /// Naive O(N · M) encoder. Mirrors Python encode().
        std::vector<TokenId> encode(std::string_view text) const;

        /// Priority-queue O(N log M) encoder, chunked for bounded peak memory.
        /// Mirrors Python encode_fast().
        std::vector<TokenId> encode_fast(std::string_view text,
                                         std::size_t chunk_size = 10'000'000) const;

        // ── Decoding ─────────────────────────────────────────────────────────────

        /// Convert token ids back to a UTF-8 string. Mirrors Python decode().
        std::string decode(std::span<const TokenId> ids) const;

        // ── Persistence ──────────────────────────────────────────────────────────

        void save(const std::filesystem::path &path) const;
        static BPETokenizer load(const std::filesystem::path &path);

        // ── Special tokens ───────────────────────────────────────────────────────

        /// Register a special token. Special tokens are matched literally in the
        /// input before BPE runs — they are never split or merged with other tokens.
        /// Typical use: add_special_token("<|endoftext|>", vocab_size())
        void add_special_token(std::string_view text, TokenId id);

        const std::unordered_map<std::string, TokenId> &
        special_tokens() const noexcept { return special_tokens_; }

        // ── Introspection ────────────────────────────────────────────────────────

        void print_vocab(int max_tokens = -1) const;

        const MergeMap &merges() const noexcept { return merges_; }
        const VocabMap &vocab() const noexcept { return vocab_; }
        std::size_t vocab_size() const noexcept { return vocab_.size(); }

    private:
        MergeMap merges_;
        VocabMap vocab_;

        // text → id  (for encoding)
        std::unordered_map<std::string, TokenId> special_tokens_;
        // id → text  (for decoding)
        std::unordered_map<TokenId, std::string> special_token_ids_;

        // ── Internal helpers ─────────────────────────────────────────────────────

        /// Seed vocab with the 256 raw-byte tokens (called at training start).
        void init_byte_vocab();

        /// Count pair frequencies in a token sequence. Used by train().
        static PairFreqMap count_pairs(std::span<const TokenId> tokens);

        /// Replace all occurrences of `pair` in `tokens` with `idx`.
        /// Returns the new (shorter) token sequence.
        static std::vector<TokenId> merge_pair(std::span<const TokenId> tokens,
                                               Pair pair, TokenId idx);

        /// Encode one contiguous byte chunk with the priority-queue algorithm.
        /// Called repeatedly by encode_fast() for large inputs.
        std::vector<TokenId> encode_chunk(std::span<const std::uint8_t> chunk) const;

        // ── Memory-mapped file view (RAII) ────────────────────────────────────────
        // Platform-specific implementation lives in bpe.cpp.
        class MmapView
        {
        public:
            explicit MmapView(const std::filesystem::path &path);
            ~MmapView();

            MmapView(const MmapView &) = delete;
            MmapView &operator=(const MmapView &) = delete;
            MmapView(MmapView &&) noexcept;
            MmapView &operator=(MmapView &&) noexcept;

            const std::uint8_t *data() const noexcept { return data_; }
            std::size_t size() const noexcept { return size_; }

            /// Span over the mapped bytes — zero-copy, read-only.
            std::span<const std::uint8_t> as_span() const noexcept
            {
                return {data_, size_};
            }

        private:
            const std::uint8_t *data_{nullptr};
            std::size_t size_{0};

#ifdef _WIN32
            void *file_handle_{nullptr}; // HANDLE
            void *map_handle_{nullptr};  // HANDLE
#else
            int fd_{-1};
#endif
        };
    };

} // namespace paircraft
