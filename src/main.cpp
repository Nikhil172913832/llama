#include <omp.h>

#include <charconv>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <chrono>
#include "model.h"
#include "tokenizer.h"
namespace {
struct CLIConfig {
    std::filesystem::path model_checkpoint_path;
    std::filesystem::path tokenizer_path;
    std::string prompt;
    int max_tokens;
    int threads;
    bool help;
};
CLIConfig parse_args(int argc, char* argv[]) {
    std::filesystem::path model_checkpoint_path = "testdata/stories15M.bin";
    std::filesystem::path tokenizer_path = "testdata/tokenizer.bin";
    std::string prompt = "";
    int max_tokens = 256;
    int threads = 1;
    bool help = false;
    for (int i = 1; i < argc; i++) {
        std::string_view arg(argv[i]);
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                model_checkpoint_path = argv[++i];
            } else {
                throw std::runtime_error("Insufficient arguements");
            }
        } else if (arg == "-z" || arg == "--tokenizer") {
            if (i + 1 < argc) {
                tokenizer_path = argv[++i];
            } else {
                throw std::runtime_error("Insufficient arguements");
            }
        } else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 < argc) {
                prompt = argv[++i];
            } else {
                throw std::runtime_error("Insufficient arguements");
            }
        } else if (arg == "-n" || arg == "--steps") {
            if (i + 1 < argc) {
                max_tokens = std::stoi(argv[++i]);
                if (max_tokens <= 0 || max_tokens > 256) {
                    throw std::runtime_error("Invalid max tokens");
                }
            } else {
                throw std::runtime_error("Insufficient arguments");
            }
        } else if (arg == "-h" || arg == "--help") {
            help = true;
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                threads = std::stoi(argv[++i]);
                if (threads <= 0) {
                    throw std::runtime_error("Invalid number of threads");
                }
            } else {
                throw std::runtime_error("Insufficient arguements");
            }
        } else {
            throw std::runtime_error("Invalid arguement call -h or --help");
        }
    }
    return CLIConfig{
        .model_checkpoint_path = model_checkpoint_path,
        .tokenizer_path = tokenizer_path,
        .prompt = prompt,
        .max_tokens = max_tokens,
        .threads = threads,
        .help = help,
    };
}

void print_token(std::string_view piece) {
    if (piece.size() == 6 && piece.starts_with("<0x") && piece.ends_with('>')) {
        unsigned char byte_val = 0;
        auto [ptr, ec] =
            std::from_chars(piece.data() + 3, piece.data() + 5, byte_val, 16);
        if (ec == std::errc{} && ptr == piece.data() + 5) {
            std::cout << static_cast<char>(byte_val);
        } else {
            std::cout << piece;
        }
    } else {
        std::cout << piece;
    }
}

void print_usage() {
    constexpr std::string_view usage =
        "Usage: llama_run [options]\n"
        "Options:\n"
        "  -m, --model <path>       "
        "Path to model checkpoint (default: testdata/stories15M.bin)\n"
        "  -z, --tokenizer <path>   "
        "Path to tokenizer file (default: testdata/tokenizer.bin)\n"
        "  -p, --prompt <string>    "
        "Input prompt (default: \"\")\n"
        "  -n, --steps <int>        "
        "Maximum tokens to generate (default: 256)\n"
        "  -t, --threads <int>      "
        "Number of OpenMP threads (default: 1)\n"
        "  -h, --help               "
        "Show this help message and exit\n";
    std::cout << usage;
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        CLIConfig config = parse_args(argc, argv);
        if (config.help) {
            print_usage();
            return 0;
        }
        llama::Engine engine = llama::Engine(config.model_checkpoint_path);
        llama::Tokenizer tokenizer =
            llama::Tokenizer(config.tokenizer_path, 32000);
        std::vector<int> tokens = tokenizer.encode(config.prompt);
        omp_set_num_threads(config.threads);
        int prev_token = tokens.back();
        int generated_count = 0;
        auto on_token = [&](int token_id) {
            if (token_id == 2) {
                return false;
            }
            std::string_view piece = tokenizer.decode(token_id, prev_token);
            print_token(piece);
            std::cout << std::flush;
            prev_token = token_id;
            generated_count++;
            return generated_count < config.max_tokens;
        };
        std::cout << config.prompt;
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.generate(tokens, on_token);
        std::cout << "\n";
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(t1 - t0).count();
        auto tok_per_sec = generated_count / elapsed;

        std::cerr << "\n==================================================\n";
        std::cerr << "Generated Tokens : " << generated_count << "\n";
        std::cerr << "Time Taken       : " << elapsed << " s\n";
        std::cerr << "Throughput       : " << tok_per_sec << " tok/s\n";
        std::cerr << "==================================================\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
