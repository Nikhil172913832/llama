#include "model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ops.h"

namespace llama {
void forward(const Config& config, const Weights& weights, RunState& state,
             int token_id, std::size_t pos, bool compute_logits) {
    llama::embed(weights.token_embedding_table, state.x, token_id, config.dim);
    for (int i = 0; i < config.n_layers; i++) {
        llama::rmsnorm(state.x, weights.layers[i].rms_att_weight, state.xb,
                       config.eps, 1, config.dim);
        llama::matvec(state.xb, weights.layers[i].wk, state.k, config.kv_dim,
                      config.dim);
        llama::matvec(state.xb, weights.layers[i].wq, state.q, config.dim,
                      config.dim);
        llama::matvec(state.xb, weights.layers[i].wv, state.v, config.kv_dim,
                      config.dim);
        llama::rope(state.q, state.k, state.cos, state.sin, 1, config.n_heads,
                    config.n_kv_heads, config.head_dim, pos);
        for (std::size_t j = 0; j < config.n_kv_heads; j++) {
            std::size_t cache_offset =
                i * config.n_kv_heads * config.max_seq_len * config.head_dim +
                j * config.max_seq_len * config.head_dim +
                pos * config.head_dim;
            float* k_cache_vec = state.k_cache + cache_offset;
            float* v_cache_vec = state.v_cache + cache_offset;
            std::size_t offset = j * config.head_dim;
            float* k_vec = state.k + offset;
            float* v_vec = state.v + offset;
            memcpy(k_cache_vec, k_vec, config.head_dim * sizeof(float));
            memcpy(v_cache_vec, v_vec, config.head_dim * sizeof(float));
        }
        std::size_t cache_offset =
            i * config.n_kv_heads * config.max_seq_len * config.head_dim;
        llama::attention(state.k_cache + cache_offset,
                         state.v_cache + cache_offset, state.q, state.scores,
                         state.xb, pos, config.n_heads, config.n_kv_heads,
                         config.head_dim, config.max_seq_len);
        llama::matvec(state.xb, weights.layers[i].wo, state.q, config.dim,
                      config.dim);
        llama::accum(state.x, state.q, config.dim);
        llama::rmsnorm(state.x, weights.layers[i].rms_ffn_weight, state.xb,
                       config.eps, 1, config.dim);
        llama::matvec(state.xb, weights.layers[i].w_up, state.h_up,
                      config.hidden_dim, config.dim);
        llama::matvec(state.xb, weights.layers[i].w_gate, state.h_gate,
                      config.hidden_dim, config.dim);
        llama::swiglu(state.h_up, state.h_gate, config.hidden_dim);
        llama::matvec(state.h_gate, weights.layers[i].w_down, state.xb,
                      config.dim, config.hidden_dim);
        llama::accum(state.x, state.xb, config.dim);
    }
    if (compute_logits) {
        llama::rmsnorm(state.x, weights.rms_final_weight, state.xb, config.eps,
                       1, config.dim);
        llama::matvec(state.xb, weights.wcls, state.logits, config.vocab_size,
                      config.dim);
    }
}
void Engine::mm_weights(const std::string& checkpoint_path) {
    fd = open(checkpoint_path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("file not found");
    }
    struct stat file_info;
    if (fstat(fd, &file_info) == -1) {
        close(fd);
        throw std::runtime_error("fstat failed");
    }
    file_size = file_info.st_size;
    mmap_data = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_data == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }
    const int32_t* data = static_cast<int32_t*>(mmap_data);
    for (int i = 0; i < 7; i++) {
        if (i != 5 && data[i] < 0) {
            throw std::runtime_error("config is corrupted");
        }
    }
    bool shared_classifier = (data[5] > 0);
    config.vocab_size = static_cast<std::size_t>(std::abs(data[5]));
    config.dim = static_cast<std::size_t>(data[0]);
    config.hidden_dim = static_cast<std::size_t>(data[1]);
    config.n_layers = static_cast<std::size_t>(data[2]);
    config.n_heads = static_cast<std::size_t>(data[3]);
    config.n_kv_heads = static_cast<std::size_t>(data[4]);
    config.max_seq_len = static_cast<std::size_t>(data[6]);
    config.head_dim = config.dim / config.n_heads;
    config.kv_dim = config.n_kv_heads * config.head_dim;
    config.eps = 1e-5f;
    const float* weights_p = reinterpret_cast<const float*>(data + 7);
    weights.token_embedding_table = weights_p;
    weights_p += config.vocab_size * config.dim;
    weights.layers = std::vector<LayerWeights>(config.n_layers);
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].rms_att_weight = weights_p;
        weights_p += config.dim;
    }
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].wq = weights_p;
        weights_p += config.dim * config.dim;
    }
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].wk = weights_p;
        weights_p += config.kv_dim * config.dim;
    }
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].wv = weights_p;
        weights_p += config.kv_dim * config.dim;
    }
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].wo = weights_p;
        weights_p += config.dim * config.dim;
    }
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].rms_ffn_weight = weights_p;
        weights_p += config.dim;
    }
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].w_gate = weights_p;
        weights_p += config.hidden_dim * config.dim;
    }
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].w_down = weights_p;
        weights_p += config.hidden_dim * config.dim;
    }
    for (size_t i = 0; i < config.n_layers; i++) {
        weights.layers[i].w_up = weights_p;
        weights_p += config.hidden_dim * config.dim;
    }
    weights.rms_final_weight = weights_p;
    weights_p += config.dim;
    weights_p += config.max_seq_len * config.head_dim;
    if (shared_classifier) {
        weights.wcls = weights.token_embedding_table;
    } else {
        weights.wcls = weights_p;
    }
}
Engine::Engine(const std::string& checkpoint_path) {
    fd = -1;
    mmap_data = nullptr;
    file_size = 0;
    mm_weights(checkpoint_path);
    carve_run_state();
}
Engine::~Engine() {
    if (mmap_data != nullptr) {
        munmap(mmap_data, file_size);
    }
    if (fd >= 0) {
        close(fd);
    }
}
void Engine::carve_run_state() {
    std::size_t total_floats = 3 * config.dim + 2 * config.kv_dim +
                               2 * config.hidden_dim + config.max_seq_len +
                               config.vocab_size +
                               2 * config.n_layers * config.n_kv_heads *
                                   config.max_seq_len * config.head_dim +
                               config.max_seq_len * config.head_dim;
    arena.emplace(total_floats);
    state.x = arena->alloc(config.dim);
    state.xb = arena->alloc(config.dim);
    state.q = arena->alloc(config.dim);
    state.k = arena->alloc(config.kv_dim);
    state.v = arena->alloc(config.kv_dim);
    state.h_up = arena->alloc(config.hidden_dim);
    state.h_gate = arena->alloc(config.hidden_dim);
    state.scores = arena->alloc(config.max_seq_len);
    state.logits = arena->alloc(config.vocab_size);
    state.k_cache = arena->alloc(config.n_layers * config.n_kv_heads *
                                 config.max_seq_len * config.head_dim);
    state.v_cache = arena->alloc(config.n_layers * config.n_kv_heads *
                                 config.max_seq_len * config.head_dim);
    state.cos = arena->alloc(config.max_seq_len * (config.head_dim / 2));
    state.sin = arena->alloc(config.max_seq_len * (config.head_dim / 2));
    precompute_freq_cis(config.head_dim, config.max_seq_len, state.cos,
                        state.sin);
}
void Engine::generate(const std::vector<int>& prompt_tokens,
                      std::function<bool(int)> on_token) {
    if (prompt_tokens.empty()) {
        return;
    }
    if (prompt_tokens.size() > config.max_seq_len) {
        throw std::runtime_error("Prompt exceeds max_seq_len");
    }
    for (size_t pos = 0; pos < prompt_tokens.size() - 1; pos++) {
        forward(config, weights, state, prompt_tokens[pos], pos, false);
    }
    forward(config, weights, state, prompt_tokens[prompt_tokens.size() - 1],
            prompt_tokens.size() - 1);
    int next_token =
        std::max_element(state.logits, state.logits + config.vocab_size) -
        state.logits;
    for (size_t pos = prompt_tokens.size(); pos < config.max_seq_len; pos++) {
        if (!on_token(next_token)) {
            break;
        }
        forward(config, weights, state, next_token, pos);
        next_token =
            std::max_element(state.logits, state.logits + config.vocab_size) -
            state.logits;
    }
}
}  // namespace llama
