// paircraft — CLI tokenizer
//
// Usage:
//   paircraft_cli <corpus.txt> [options]
//
// Options:
//   --vocab-size N      Vocabulary size (default: 512, minimum: 256)
//   --output FILE.npy   Write encoded token IDs as NumPy uint16 array (np.load compatible)
//   --print-vocab N     Print first N vocab entries after training (0 = skip)
//   --help              Show this message
//
// Example:
//   paircraft_cli corpus.txt --vocab-size 1024 --output tokens.npy

#include "paircraft/bpe.h"

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args
{
    fs::path input_file;
    int vocab_size = 512;
    fs::path output_file;     // empty = don't write
    fs::path save_model_file; // empty = don't save
    fs::path load_model_file; // empty = train from scratch
    int print_vocab = 0;      // 0 = don't print
    int print_longest = 0;    // 0 = don't print; requires --load-model
    bool decode_mode = false; // --decode: input_file is .npy, output is text
};

[[noreturn]] static void usage(std::string_view argv0, int exit_code = 0)
{
    std::println(stderr,
                 "Usage: {} <corpus.txt> [options]\n"
                 "\n"
                 "Options:\n"
                 "  --vocab-size N      Vocabulary size (default: 512, min: 256)\n"
                 "  --output FILE       Encode: write .npy token IDs / Decode: write text file\n"
                 "  --save-model FILE   Save trained model to binary file after training\n"
                 "  --load-model FILE   Load model from file instead of training\n"
                 "  --decode            Decode mode: input is .npy tokens, output is text\n"
                 "  --print-vocab N     Print first N vocab entries after training\n"
                 "  --print-longest N   Print N longest tokens from a loaded model\n"
                 "  --help              Show this message\n",
                 argv0);
    std::exit(exit_code);
}

static int parse_int(std::string_view flag, std::string_view val)
{
    int result{};
    auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), result);
    if (ec != std::errc{} || ptr != val.data() + val.size())
        throw std::invalid_argument(
            std::format("'{}' expects an integer, got '{}'", flag, val));
    return result;
}

static Args parse_args(std::span<const char *const> argv)
{
    if (argv.size() < 2)
        usage(argv[0], 1);

    Args args;
    for (std::size_t i = 1; i < argv.size(); ++i)
    {
        std::string_view arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            usage(argv[0]);
        }
        else if (arg == "--vocab-size")
        {
            if (++i >= argv.size())
                throw std::invalid_argument("--vocab-size requires a value");
            args.vocab_size = parse_int("--vocab-size", argv[i]);
        }
        else if (arg == "--output")
        {
            if (++i >= argv.size())
                throw std::invalid_argument("--output requires a value");
            args.output_file = argv[i];
        }
        else if (arg == "--save-model")
        {
            if (++i >= argv.size())
                throw std::invalid_argument("--save-model requires a value");
            args.save_model_file = argv[i];
        }
        else if (arg == "--load-model")
        {
            if (++i >= argv.size())
                throw std::invalid_argument("--load-model requires a value");
            args.load_model_file = argv[i];
        }
        else if (arg == "--decode")
        {
            args.decode_mode = true;
        }
        else if (arg == "--print-vocab")
        {
            if (++i >= argv.size())
                throw std::invalid_argument("--print-vocab requires a value");
            args.print_vocab = parse_int("--print-vocab", argv[i]);
        }
        else if (arg == "--print-longest")
        {
            if (++i >= argv.size())
                throw std::invalid_argument("--print-longest requires a value");
            args.print_longest = parse_int("--print-longest", argv[i]);
        }
        else if (arg.starts_with('-'))
        {
            throw std::invalid_argument(std::format("Unknown option: {}", arg));
        }
        else
        {
            if (!args.input_file.empty())
                throw std::invalid_argument("Only one input file is supported");
            args.input_file = arg;
        }
    }

    if (args.input_file.empty() && args.print_longest == 0 && !args.decode_mode)
    {
        std::println(stderr, "Error: no input file specified.");
        usage(argv[0], 1);
    }
    if (args.vocab_size < 256)
    {
        std::println(stderr, "Error: --vocab-size must be >= 256.");
        std::exit(1);
    }
    return args;
}

// ── File I/O ──────────────────────────────────────────────────────────────────

static std::string read_file(const fs::path &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Cannot open: " + path.string());

    const auto size = f.tellg();
    f.seekg(0);

    std::string buf(static_cast<std::size_t>(size), '\0');
    f.read(buf.data(), size);
    if (!f)
        throw std::runtime_error("Read error: " + path.string());
    return buf;
}

// Write a NumPy v1.0 .npy file containing a 1-D int16 array.
// Loadable in Python with: arr = np.load("tokens.npy")  # dtype int16, shape (N,)
// Matches the dtype produced by the Python bpe.py encode() (np.array(..., dtype=np.int16)).
//
// Format: [6B magic][2B version=1.0][2B header_len LE][header_len B header][data]
// Spec invariant: (10 + header_len) % 64 == 0
// dtype '<i2' = little-endian int16. TokenId values 0..32767 are identical in
// int16 and uint16 representation, so no information is lost for vocab_size ≤ 32768.
static void write_tokens_npy(const fs::path &path,
                             std::span<const paircraft::TokenId> ids)
{
    // Build the NumPy header dict (shape known at this point)
    std::string dict = std::format(
        "{{'descr': '<i2', 'fortran_order': False, 'shape': ({},), }}",
        ids.size());

    // Pad with spaces so (10 + header_len) is a multiple of 64 (NumPy spec).
    // header_len = dict.size() + pad + 1  (+1 for the mandatory '\n')
    const std::size_t raw_size = 10 + dict.size() + 1;
    const std::size_t pad = (64 - raw_size % 64) % 64;
    dict.append(pad, ' ');
    dict += '\n';

    const auto header_len = static_cast<uint16_t>(dict.size());

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot write: " + path.string());

    // Preamble: magic (\x93NUMPY) + version (1.0)
    static constexpr uint8_t kPreamble[] = {
        0x93, 'N', 'U', 'M', 'P', 'Y',
        0x01, 0x00};
    f.write(reinterpret_cast<const char *>(kPreamble), sizeof(kPreamble));

    // Header length field (little-endian uint16)
    const uint8_t hlen[2] = {
        static_cast<uint8_t>(header_len & 0xFF),
        static_cast<uint8_t>(header_len >> 8)};
    f.write(reinterpret_cast<const char *>(hlen), 2);

    // Header string (ASCII dict, space-padded, newline-terminated)
    f.write(dict.data(), static_cast<std::streamsize>(dict.size()));

    // Token data: contiguous uint16 little-endian values (native on x86/x64)
    f.write(reinterpret_cast<const char *>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(paircraft::TokenId)));

    if (!f)
        throw std::runtime_error("Write error: " + path.string());
}

// ── Read .npy token file ──────────────────────────────────────────────────────

// static std::vector<paircraft::TokenId> read_tokens_npy(const fs::path &path)
// {
//     std::ifstream f(path, std::ios::binary);
//     if (!f)
//         throw std::runtime_error("Cannot open: " + path.string());

//     // magic + version (8 bytes)
//     uint8_t preamble[8]{};
//     f.read(reinterpret_cast<char *>(preamble), 8);
//     if (preamble[0] != 0x93 || std::string_view(reinterpret_cast<char *>(preamble + 1), 5) != "NUMPY")
//         throw std::runtime_error("Not a NumPy file: " + path.string());

//     // header length (little-endian uint16)
//     uint8_t hlen_bytes[2]{};
//     f.read(reinterpret_cast<char *>(hlen_bytes), 2);
//     const uint16_t header_len = static_cast<uint16_t>(hlen_bytes[0] | (hlen_bytes[1] << 8));

//     // skip the header dict — we trust the file was written by paircraft
//     std::string header(header_len, '\0');
//     f.read(header.data(), header_len);

//     // read remaining bytes as int16 token IDs
//     const auto data_start = f.tellg();
//     f.seekg(0, std::ios::end);
//     const auto data_bytes = static_cast<std::size_t>(f.tellg() - data_start);
//     f.seekg(data_start);

//     if (data_bytes % sizeof(paircraft::TokenId) != 0)
//         throw std::runtime_error("Token data length not divisible by 2: " + path.string());

//     std::vector<paircraft::TokenId> ids(data_bytes / sizeof(paircraft::TokenId));
//     f.read(reinterpret_cast<char *>(ids.data()),
//            static_cast<std::streamsize>(data_bytes));
//     if (!f)
//         throw std::runtime_error("Read error: " + path.string());

//     return ids;
// }
static std::vector<paircraft::TokenId> read_tokens_npy(const fs::path &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open: " + path.string());

    // magic + version (8 bytes)
    uint8_t preamble[8]{};
    f.read(reinterpret_cast<char *>(preamble), 8);
    if (preamble[0] != 0x93 || std::string_view(reinterpret_cast<char *>(preamble + 1), 5) != "NUMPY")
        throw std::runtime_error("Not a NumPy file: " + path.string());

    // header length (little-endian uint16)
    uint8_t hlen_bytes[2]{};
    f.read(reinterpret_cast<char *>(hlen_bytes), 2);
    const uint16_t header_len = static_cast<uint16_t>(hlen_bytes[0] | (hlen_bytes[1] << 8));

    // skip the header dict
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // read remaining bytes as int32
    const auto data_start = f.tellg();
    f.seekg(0, std::ios::end);
    const auto data_bytes = static_cast<std::size_t>(f.tellg() - data_start);
    f.seekg(data_start);

    if (data_bytes % 4 != 0) // each token is 4 bytes in file
        throw std::runtime_error("Token data length not divisible by 4: " + path.string());

    const size_t token_count = data_bytes / 4;
    std::vector<int32_t> temp(token_count);
    f.read(reinterpret_cast<char *>(temp.data()), static_cast<std::streamsize>(data_bytes));
    if (!f)
        throw std::runtime_error("Read error: " + path.string());

    // convert to 16-bit TokenId (truncate)
    std::vector<paircraft::TokenId> ids(token_count);
    for (size_t i = 0; i < token_count; ++i)
        ids[i] = static_cast<paircraft::TokenId>(temp[i]);

    return ids;
}

// ── Stats ─────────────────────────────────────────────────────────────────────

static void print_stats(const paircraft::BPETokenizer &tok,
                        std::size_t byte_len,
                        std::span<const paircraft::TokenId> ids)
{
    const double ratio = static_cast<double>(byte_len) /
                         static_cast<double>(ids.size());
    std::println("\n── Stats ──────────────────────────────────────────────");
    std::println("  Input bytes     : {:>12L}", byte_len);
    std::println("  Encoded tokens  : {:>12L}", ids.size());
    std::println("  Compression     : {:>11.2f}x  ({:.1f} bytes/token)",
                 ratio, ratio);
    std::println("  Vocab size      : {:>12L}", tok.vocab_size());
    std::println("  Merges learned  : {:>12L}", tok.merges().size());
    std::println("───────────────────────────────────────────────────────");
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, const char *argv[])
{
    try
    {
        const Args args = parse_args({argv, static_cast<std::size_t>(argc)});

        // ── Decode mode ───────────────────────────────────────────────────────
        if (args.decode_mode)
        {
            if (args.load_model_file.empty())
                throw std::invalid_argument("--decode requires --load-model");

            const auto tok = paircraft::BPETokenizer::load(args.load_model_file);
            const auto ids = read_tokens_npy(args.input_file);
            std::println("Decoding {:L} tokens…", ids.size());

            const std::string text = tok.decode(ids);

            if (args.output_file.empty())
            {
                std::cout << text;
            }
            else
            {
                std::ofstream out(args.output_file, std::ios::binary);
                if (!out)
                    throw std::runtime_error("Cannot write: " + args.output_file.string());
                out.write(text.data(), static_cast<std::streamsize>(text.size()));
                std::println("Decoded text written to '{}'  ({:L} bytes).",
                             args.output_file.string(), text.size());
            }
            return 0;
        }

        // ── Print longest vocab entries ───────────────────────────────────────
        if (args.print_longest > 0)
        {
            if (args.load_model_file.empty())
                throw std::invalid_argument("--print-longest requires --load-model");

            const auto tok = paircraft::BPETokenizer::load(args.load_model_file);

            // collect all vocab entries, sort by byte length descending
            using Entry = std::pair<std::size_t, std::string>; // (len, text)
            std::vector<Entry> entries;
            entries.reserve(tok.vocab().size());
            for (const auto &[id, bytes] : tok.vocab())
                entries.push_back({bytes.size(),
                                   std::string(bytes.begin(), bytes.end())});

            std::sort(entries.begin(), entries.end(),
                      [](const Entry &a, const Entry &b)
                      { return a.first > b.first; });

            const int n = std::min(args.print_longest,
                                   static_cast<int>(entries.size()));
            std::println("── {} longest tokens ──────────────────────────────", n);
            for (int i = 0; i < n; ++i)
                std::println("  {:>4} bytes  {:?}", entries[i].first, entries[i].second);

            return 0;
        }

        // Special tokens added to every freshly trained model.
        // Extend this list to add more — the training loop automatically
        // reserves the right number of slots so vocab_size stays exact.
        static constexpr std::string_view kSpecialTokens[] = {
            "<|endoftext|>",
        };
        static constexpr int kNumSpecial =
            static_cast<int>(std::size(kSpecialTokens));

        // ── Train or load ─────────────────────────────────────────────────────
        paircraft::BPETokenizer tok;
        if (!args.load_model_file.empty())
        {
            tok = paircraft::BPETokenizer::load(args.load_model_file);
        }
        else
        {
            // Train (vocab_size - kNumSpecial) BPE merges so that after
            // registering special tokens the total equals args.vocab_size.
            tok.train_from_file(args.input_file, args.vocab_size - kNumSpecial);
            const auto base_id =
                static_cast<paircraft::TokenId>(tok.vocab_size());
            for (int i = 0; i < kNumSpecial; ++i)
                tok.add_special_token(kSpecialTokens[i],
                                      static_cast<paircraft::TokenId>(base_id + i));
            if (!args.save_model_file.empty())
                tok.save(args.save_model_file);
        }

        // ── Encode ────────────────────────────────────────────────────────────
        // Still reads the corpus into RAM for encoding.
        // TODO: replace with tok.encode_fast() once implemented.
        std::println("Reading '{}' for encoding…", args.input_file.string());
        std::string text = read_file(args.input_file);
        const auto ids = tok.encode_fast(text);

        // ── Stats ─────────────────────────────────────────────────────────────
        print_stats(tok, text.size(), ids);

        // ── Optional vocab dump ───────────────────────────────────────────────
        if (args.print_vocab > 0)
        {
            std::println("\n── Vocabulary (first {} entries) ──────────────────",
                         args.print_vocab);
            tok.print_vocab(args.print_vocab);
        }

        // ── Optional token output ─────────────────────────────────────────────
        if (!args.output_file.empty())
        {
            write_tokens_npy(args.output_file, ids);
            std::println("\nTokens written to '{}'  ({:L} tokens, dtype=int16).",
                         args.output_file.string(), ids.size());
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
}
