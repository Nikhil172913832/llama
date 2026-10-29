#include "ops.h"
#include <cmath>
#include <cstddef>
#include <algorithm>
void rmsnorm(const float *x, const float *w, float *out, float eps, int rows,
             int ed) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (std::size_t)r * ed;
        float *outr = out + (std::size_t)r * ed;
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

void matvec(const float *x, const float *w, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float temp = 0.f;
        const float *row = w + (std::size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            temp += row[j] * x[j];
        }
        out[i] = temp;
    }
}

void softmax(float *x, int n) {
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

void apply_rot(float *row, const float *c, const float *s, const int n_heads,
               const int head_dim) {
    for (int j = 0; j < n_heads; j++) {
        for (int l = 0; l < head_dim / 2; l++) {
            int index = j * head_dim + 2 * l;
            float temp = row[index];
            row[index] = c[l] * row[index] - s[l] * row[index + 1];
            row[index + 1] = s[l] * temp + c[l] * row[index + 1];
        }
    }
}

void rope(float *q, float *k, const float *c, const float *s, const int seq_len,
          const int n_heads, const int n_kv_heads, const int head_dim) {
    for (int i = 0; i < seq_len; i++) {
        float *row = q + (std::size_t)i * n_heads * head_dim;
        const float *ci = c + (std::size_t)i * (head_dim / 2);
        const float *si = s + (std::size_t)i * (head_dim / 2);
        apply_rot(row, ci, si, n_heads, head_dim);
        row = k + (std::size_t)i * n_kv_heads * head_dim;
        apply_rot(row, ci, si, n_kv_heads, head_dim);
    }
}
