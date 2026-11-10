
void rmsnorm(const float *x, const float *w, float *out, float eps, int rows,
             int ed);
void matvec(const float *x, const float *w, float *out, int rows, int cols);
void softmax(float *x, int n);
void apply_rot(float *row, const float *c, const float *s, const int n_heads,
               const int head_dim);
void rope(float *q, float *k, const float *c, const float *s, const int seq_len,
          const int n_heads, const int n_kv_heads, const int head_dim);
void precompute_freq_cis(const int head_dim, const int max_seq_len, float *c,
                         float *s);