#pragma once
#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

#include "mem.h"
namespace llama {
struct Config {
    std::size_t dim;
    std::size_t head_dim;
    std::size_t kv_dim;
    std::size_t hidden_dim;
    std::size_t n_layers;
    std::size_t n_heads;
    std::size_t n_kv_heads;
    std::size_t vocab_size;
    std::size_t max_seq_len;
    float eps;
};
struct LayerWeights {
    const float* rms_att_weight;
    const float* wq;
    const float* wk;
    const float* wv;
    const float* wo;
    const float* rms_ffn_weight;
    const float* w_up;
    const float* w_down;
    const float* w_gate;
};
struct Weights {
    const float* token_embedding_table;
    const float* rms_final_weight;
    const float* wcls;
    std::vector<LayerWeights> layers;
};
struct RunState {
    float* x;
    float* xb;
    float* q;
    float* k;
    float* v;
    float* h_up;
    float* h_gate;
    float* scores;
    float* logits;
    float* k_cache;
    float* v_cache;
    float* cos;
    float* sin;
};
void forward(const Config& config, const Weights& weights, RunState& state,
             int token_id, std::size_t pos, bool compute_logits = true);
class Engine {
   public:
    Engine(const std::filesystem::path& checkpoint_path);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    void generate(const std::vector<int>& prompt_tokens,
                  std::function<bool(int token_id)> on_token);

   private:
    Config config;
    Weights weights;
    RunState state;
    std::optional<MemoryArena> arena;
    int fd;
    void* mmap_data;
    std::size_t file_size;
    void carve_run_state();
    void mm_weights(const std::filesystem::path& checkpoint_path);
};
}  // namespace llama
