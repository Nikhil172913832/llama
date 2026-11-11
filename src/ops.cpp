#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>
namespace llama {
void rmsnorm(const float* x, const float* w, float* out, float eps, int rows,
             int ed) {
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

void matvec(const float* x, const float* w, float* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float temp = 0.f;
        const float* row = w + (std::size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            temp += row[j] * x[j];
        }
        out[i] = temp;
    }
}

void softmax(float* x, int n) {
    float maxVal = x[0];
    for (int i = 1; i < n; i++) {
        maxVal = std::max(maxVal, x[i]);
    }
    float d = 0.f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxVal);
        d += x[i];
    }
    float inv = 1 / d;
    for (int i = 0; i < n; i++) {
        x[i] *= inv;
    }
}

void apply_rot(float* row, const float* c, const float* s, const int n_heads,
               const int head_dim) {
    for (int i = 0; i < n_heads; i++) {
        for (int j = 0; j < head_dim / 2; j++) {
            int index = i * head_dim + 2 * j;
            float temp = row[index];
            row[index] = c[j] * row[index] - s[j] * row[index + 1];
            row[index + 1] = s[j] * temp + c[j] * row[index + 1];
        }
    }
}

void rope(float* q, float* k, const float* c, const float* s, const int seq_len,
          const int n_heads, const int n_kv_heads, const int head_dim,
          const int pos) {
    for (int i = 0; i < seq_len; i++) {
        float* row = q + (size_t)i * n_heads * head_dim;
        const float* ci = c + (size_t)(i + pos) * (head_dim / 2);
        const float* si = s + (size_t)(i + pos) * (head_dim / 2);
        apply_rot(row, ci, si, n_heads, head_dim);
        row = k + (size_t)i * n_kv_heads * head_dim;
        apply_rot(row, ci, si, n_kv_heads, head_dim);
    }
}

void precompute_freq_cis(const int head_dim, const int max_seq_len, float* c,
                         float* s) {
    int half = head_dim / 2;
    size_t n = (size_t)max_seq_len * half;
    std::vector<float> theta(half);
    for (int i = 0; i < half; i++) {
        theta[i] = 1 / powf(10000, ((float)i * 2) / head_dim);
    }
    for (int i = 0; i < max_seq_len; i++) {
        for (int j = 0; j < head_dim / 2; j++) {
            size_t index = (size_t)i * (head_dim / 2) + j;
            float itheta = (float)i * theta[j];
            c[index] = cosf(itheta);
            s[index] = sinf(itheta);
        }
    }
}

void embed(const float* table, float* out, int token_id, int dim) {
    const float* offset = table + token_id * dim;
    memcpy(out, offset, dim * sizeof(float));
}

void swiglu(const float* up, float* gate, int n) {
    for (int i = 0; i < n; i++) {
        gate[i] = up[i] * (gate[i] / (1 + exp(-gate[i])));
    }
}

void accum(const float* b, float* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}
}  // namespace llama
