#include "paircraft/bpe.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace paircraft;

// ── Sample corpus ─────────────────────────────────────────────────────────────
//
// Chosen to exercise BPE's core behaviour:
//   • "the " appears 8 times  → likely first or second merge
//   • "at"  appears in cat/rat/sat/mat/flat/hat (15 occurrences) → frequent merge
//   • "cat" / "rat" / "sat" share the "at" suffix
// This lets tests assert on specific merge order and compression ratio.
static constexpr std::string_view kSampleParagraph =
    "the cat sat on the mat near the flat hat. "
    "the cat ran and the rat sat. "
    "a fat cat and a fat rat sat on the mat. "
    "the cat chased the rat past the hat.";

// ── Fixtures ──────────────────────────────────────────────────────────────────

/// Minimal tokenizer trained on a tiny repetitive corpus.
class TinyTokenizer : public ::testing::Test {
protected:
    void SetUp() override {
        // "aaabdaaabac" is the classic BPE example from the literature.
        // With vocab_size=258 we get 2 merges.
        tok_.train("aaabdaaabac", 258);
    }
    BPETokenizer tok_;
};

// ── Construction ──────────────────────────────────────────────────────────────

TEST(BPETokenizerTest, DefaultConstructible) {
    BPETokenizer tok;
    EXPECT_EQ(tok.vocab_size(), 0u);
    EXPECT_TRUE(tok.merges().empty());
}

// ── Vocabulary initialisation ─────────────────────────────────────────────────

TEST(BPETokenizerTest, TrainInitialises256ByteVocab) {
    BPETokenizer tok;
    tok.train("hello", 256); // 0 merges requested
    EXPECT_EQ(tok.vocab_size(), 256u);
}

TEST(BPETokenizerTest, TrainAddsRequestedMerges) {
    BPETokenizer tok;
    tok.train("aaabdaaabac", 258);
    // 258 - 256 = 2 merges → vocab size must be 258
    EXPECT_EQ(tok.vocab_size(), 258u);
    EXPECT_EQ(tok.merges().size(), 2u);
}

// ── Pair counting ─────────────────────────────────────────────────────────────

TEST(BPETokenizerTest, MostFrequentPairIsSelectedFirst) {
    // In "aaabdaaabac" the pair (a,a) appears most often.
    BPETokenizer tok;
    tok.train("aaabdaaabac", 257); // 1 merge
    ASSERT_EQ(tok.merges().size(), 1u);
    Pair merged_pair = tok.merges().begin()->first;
    EXPECT_EQ(merged_pair.first,  static_cast<TokenId>('a'));
    EXPECT_EQ(merged_pair.second, static_cast<TokenId>('a'));
}

// ── Encode / decode round-trip ────────────────────────────────────────────────

TEST_F(TinyTokenizer, EncodeProducesFewerTokensThanBytes) {
    auto ids = tok_.encode("aaabdaaabac");
    EXPECT_LT(ids.size(), std::string("aaabdaaabac").size());
}

TEST_F(TinyTokenizer, DecodeRevertsEncode) {
    const std::string text = "aaabdaaabac";
    auto ids    = tok_.encode(text);
    auto result = tok_.decode(ids);
    EXPECT_EQ(result, text);
}

TEST_F(TinyTokenizer, EncodeEmptyString) {
    auto ids = tok_.encode("");
    EXPECT_TRUE(ids.empty());
}

TEST_F(TinyTokenizer, DecodeSingleByteToken) {
    // Token 'd' (ASCII 100) should decode back to "d"
    std::vector<TokenId> ids = {static_cast<TokenId>('d')};
    EXPECT_EQ(tok_.decode(ids), "d");
}

// ── Vocab content ─────────────────────────────────────────────────────────────

TEST(BPETokenizerTest, ByteVocabEntryIsOneByte) {
    BPETokenizer tok;
    tok.train("abc", 256);
    for (int i = 0; i < 256; ++i) {
        const auto& bytes = tok.vocab().at(static_cast<TokenId>(i));
        ASSERT_EQ(bytes.size(), 1u);
        EXPECT_EQ(bytes[0], static_cast<uint8_t>(i));
    }
}

TEST(BPETokenizerTest, MergeVocabEntryIsConcatenation) {
    // After merging ('a','a') → 256, vocab[256] should equal {0x61, 0x61}
    BPETokenizer tok;
    tok.train("aaabdaaabac", 257); // 1 merge
    const auto it = tok.vocab().find(256);
    ASSERT_NE(it, tok.vocab().end());
    EXPECT_EQ(it->second, (std::vector<uint8_t>{0x61, 0x61}));
}

// ── Sample paragraph tests ────────────────────────────────────────────────────

class SampleParagraphTokenizer : public ::testing::Test {
protected:
    void SetUp() override {
        // 20 merges on the sample paragraph — enough to merge common byte n-grams
        // ("th", "the", "at", "cat", "sat", …) without over-tokenising.
        tok_.train(std::string(kSampleParagraph), 276);
    }
    BPETokenizer tok_;
};

TEST_F(SampleParagraphTokenizer, VocabSizeMatchesRequest) {
    EXPECT_EQ(tok_.vocab_size(), 276u); // 256 bytes + 20 merges
}

TEST_F(SampleParagraphTokenizer, CompressionReducesTokenCount) {
    auto ids = tok_.encode(kSampleParagraph);
    // 20 merges on a repetitive paragraph should noticeably compress.
    EXPECT_LT(ids.size(), kSampleParagraph.size());
}

TEST_F(SampleParagraphTokenizer, RoundTripIsLossless) {
    auto ids    = tok_.encode(kSampleParagraph);
    auto result = tok_.decode(ids);
    EXPECT_EQ(result, std::string(kSampleParagraph));
}

TEST_F(SampleParagraphTokenizer, FrequentSubwordIsMerged) {
    // "th" or "at" must appear among the learned merges — they are by far
    // the most frequent byte pairs in the sample paragraph.
    bool found_th = tok_.merges().count(Pair{'t', 'h'}) > 0;
    bool found_at = tok_.merges().count(Pair{'a', 't'}) > 0;
    EXPECT_TRUE(found_th || found_at)
        << "Expected at least one of ('t','h') or ('a','t') to be merged";
}

TEST_F(SampleParagraphTokenizer, EncodedTokensAreAllInVocab) {
    auto ids = tok_.encode(kSampleParagraph);
    for (TokenId id : ids)
        EXPECT_TRUE(tok_.vocab().count(id) > 0)
            << "Token " << id << " not found in vocab";
}

// ── train_from_file ───────────────────────────────────────────────────────────

TEST(BPETokenizerTest, TrainFromFileThrowsOnMissingFile) {
    BPETokenizer tok;
    // nonexistent.txt → MmapView constructor throws std::runtime_error
    EXPECT_THROW(tok.train_from_file("nonexistent.txt", 300),
                 std::runtime_error);
}

// ── encode_fast (placeholder) ─────────────────────────────────────────────────

TEST_F(TinyTokenizer, EncodeFastThrowsUntilImplemented) {
    EXPECT_THROW(tok_.encode_fast("aaabdaaabac"), std::logic_error);
}

// ── save / load (placeholder) ─────────────────────────────────────────────────

TEST_F(TinyTokenizer, SaveThrowsUntilImplemented) {
    EXPECT_THROW(tok_.save("tok.bin"), std::logic_error);
}

TEST(BPETokenizerTest, LoadThrowsUntilImplemented) {
    EXPECT_THROW(BPETokenizer::load("tok.bin"), std::logic_error);
}

// ── PairHash sanity ───────────────────────────────────────────────────────────

TEST(PairHashTest, SamePairSameHash) {
    PairHash h;
    EXPECT_EQ(h(Pair{1, 2}), h(Pair{1, 2}));
}

TEST(PairHashTest, DifferentPairsDifferentHash) {
    PairHash h;
    // This is probabilistic; the specific values here are chosen to
    // almost certainly differ for any reasonable hash function.
    EXPECT_NE(h(Pair{1, 2}), h(Pair{2, 1}));
    EXPECT_NE(h(Pair{0, 0}), h(Pair{1, 0}));
}
