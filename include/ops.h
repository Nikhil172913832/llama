#pragma once
namespace llama {
void rmsnorm(const float* x, const float* w, float* out, float eps, int rows,
             int ed);
void matvec(const float* x, const float* w, float* out, int rows, int cols);
void softmax(float* x, int n);
void apply_rot(float* row, const float* c, const float* s, const int n_heads,
               const int head_dim);
void rope(float* q, float* k, const float* c, const float* s, const int seq_len,
          const int n_heads, const int n_kv_heads, const int head_dim,
          const int pos);
void precompute_freq_cis(const int head_dim, const int max_seq_len, float* c,
                         float* s);
void embed(const float* table, float* out, int token_id, int dim);
void swiglu(const float* up, float* gate, int n);
void accum(float* a, const float* b, int n);
void attention(const float* k_cache, const float* v_cache, const float* q,
               float* scores, float* out, const int pos, const int n_heads,
               const int n_kv_heads, const int head_dim, const int max_seq_len);
}  // namespace ops
