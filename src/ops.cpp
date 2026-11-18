#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>
namespace llama {
void rmsnorm(const float* x, const float* w, float* out, float eps,
             std::size_t rows, std::size_t ed) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (std::size_t)r * ed;
        float* outr = out + (std::size_t)r * ed;
        float ss = 0.f;
        for (int k = 0; k < ed; k++) {
            ss += xr[k] * xr[k];
        }
        float rms = sqrtf(ss / ed + eps);
        for (int k = 0; k < ed; k++) {
            outr[k] = (xr[k] / rms) * w[k];
        }
    }
}

void matvec(const float* x, const float* w, float* out, std::size_t rows,
            std::size_t cols) {
    for (std::size_t i = 0; i < rows; i++) {
        float temp = 0.f;
        const float* row = w + i * cols;
        for (std::size_t j = 0; j < cols; j++) {
            temp += row[j] * x[j];
        }
        out[i] = temp;
    }
}

void softmax(float* x, std::size_t n) {
    float maxVal = x[0];
    for (std::size_t i = 1; i < n; i++) {
        maxVal = std::max(maxVal, x[i]);
    }
    float d = 0.f;
    for (std::size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxVal);
        d += x[i];
    }
    float inv = 1 / d;
    for (std::size_t i = 0; i < n; i++) {
        x[i] *= inv;
    }
}

void apply_rot(float* row, const float* cos, const float* sin,
               const std::size_t n_heads, const std::size_t head_dim) {
    for (std::size_t i = 0; i < n_heads; i++) {
        for (std::size_t j = 0; j < head_dim / 2; j++) {
            std::size_t index = i * head_dim + 2 * j;
            float temp = row[index];
            row[index] = cos[j] * row[index] - sin[j] * row[index + 1];
            row[index + 1] = sin[j] * temp + cos[j] * row[index + 1];
        }
    }
}

void rope(float* q, float* k, const float* cos, const float* sin,
          const std::size_t seq_len, const std::size_t n_heads,
          const std::size_t n_kv_heads, const std::size_t head_dim,
          const std::size_t pos) {
    for (std::size_t i = 0; i < seq_len; i++) {
        float* row = q + i * n_heads * head_dim;
        const float* cosi = cos + (i + pos) * (head_dim / 2);
        const float* sini = sin + (i + pos) * (head_dim / 2);
        apply_rot(row, cosi, sini, n_heads, head_dim);
        row = k + i * n_kv_heads * head_dim;
        apply_rot(row, cosi, sini, n_kv_heads, head_dim);
    }
}

void precompute_freq_cis(const std::size_t head_dim,
                         const std::size_t max_seq_len, float* c, float* s) {
    std::size_t half = head_dim / 2;
    std::size_t n = max_seq_len * half;
    std::vector<float> theta(half);
    for (std::size_t i = 0; i < half; i++) {
        theta[i] = 1 / powf(10000, ((float)i * 2) / head_dim);
    }
    for (std::size_t i = 0; i < max_seq_len; i++) {
        for (std::size_t j = 0; j < head_dim / 2; j++) {
            std::size_t index = i * (head_dim / 2) + j;
            float itheta = (float)i * theta[j];
            c[index] = cosf(itheta);
            s[index] = sinf(itheta);
        }
    }
}

void embed(const float* table, float* out, std::size_t token_id,
           std::size_t dim) {
    const float* offset = table + token_id * dim;
    memcpy(out, offset, dim * sizeof(float));
}

void swiglu(const float* up, float* gate, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        gate[i] = up[i] * (gate[i] / (1 + exp(-gate[i])));
    }
}

void accum(float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

void attention(const float* k_cache, const float* v_cache, const float* q,
               float* scores, float* out, const std::size_t pos,
               const std::size_t n_heads, const std::size_t n_kv_heads,
               const std::size_t head_dim, const std::size_t max_seq_len) {
    std::size_t n_rep = n_heads / n_kv_heads;
    float scaling_factor = sqrtf((float)head_dim);
    for (std::size_t i = 0; i < n_heads; i++) {
        const float* q_vec = q + i * head_dim;
        std::size_t kv_head_index = i / n_rep;
        std::size_t kv_offset = kv_head_index * max_seq_len * head_dim;
        const float* k_head = k_cache + kv_offset;
        const float* v_head = v_cache + kv_offset;
        for (std::size_t j = 0; j < pos + 1; j++) {
            const float* k_vec = k_head + j * head_dim;
            matvec(q_vec, k_vec, scores + j, 1, head_dim);
            scores[j] /= scaling_factor;
        }
        softmax(scores, pos + 1);
        float* out_head = out + i * head_dim;
        for (std::size_t j = 0; j < head_dim; j++) {
            out_head[j] = 0.f;
        }
        for (std::size_t j = 0; j < pos + 1; j++) {
            const float* row = v_head + j * head_dim;
            for (std::size_t k = 0; k < head_dim; k++) {
                out_head[k] += row[k] * scores[j];
            }
        }
    }
}
}  // namespace llama
