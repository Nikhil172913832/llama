#pragma once
#include <cstddef>
namespace llama {
void rmsnorm(const float* x, const float* w, float* out, float eps,
             std::size_t rows, std::size_t dim);
void matvec(const float* x, const float* w, float* out, std::size_t rows,
            std::size_t cols);
void softmax(float* x, std::size_t n);
void apply_rot(float* row, const float* c, const float* s,
               const std::size_t n_heads, const std::size_t head_dim);
void rope(float* q, float* k, const float* c, const float* s,
          const std::size_t seq_len, const std::size_t n_heads,
          const std::size_t n_kv_heads, const std::size_t head_dim,
          const std::size_t pos);
void precompute_freq_cis(const std::size_t head_dim,
                         const std::size_t max_seq_len, float* c, float* s);
void embed(const float* table, float* out, std::size_t token_id,
           std::size_t dim);
void swiglu(const float* up, float* gate, std::size_t n);
void accum(float* a, const float* b, std::size_t n);
void attention(const float* k_cache, const float* v_cache, const float* q,
               float* scores, float* out, const std::size_t pos,
               const std::size_t n_heads, const std::size_t n_kv_heads,
               const std::size_t head_dim, const std::size_t max_seq_len);
}  // namespace llama
