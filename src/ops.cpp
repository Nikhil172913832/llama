#include "ops.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>
namespace llama {

void rmsnorm(const float* x, const float* w, float* out, float eps,
             std::size_t rows, std::size_t dim) {
    std::size_t vec_dim = dim - (dim % 32);
    for (int i = 0; i < rows; i++) {
        const float* xr = x + (std::size_t)i * dim;
        float* outr = out + (std::size_t)i * dim;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (int j = 0; j < vec_dim; j += 32) {
            __m256 xr_vec = _mm256_loadu_ps(xr + j);
            acc0 = _mm256_fmadd_ps(xr_vec, xr_vec, acc0);
            xr_vec = _mm256_loadu_ps(xr + j + 8);
            acc1 = _mm256_fmadd_ps(xr_vec, xr_vec, acc1);
            xr_vec = _mm256_loadu_ps(xr + j + 16);
            acc2 = _mm256_fmadd_ps(xr_vec, xr_vec, acc2);
            xr_vec = _mm256_loadu_ps(xr + j + 24);
            acc3 = _mm256_fmadd_ps(xr_vec, xr_vec, acc3);
        }
        acc0 =
            _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        float buf[8];
        _mm256_storeu_ps(buf, acc0);
        float ss = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] +
                   buf[6] + buf[7];
        for (int j = vec_dim; j < dim; j++) {
            ss += xr[j] * xr[j];
        }
        float rms = sqrtf(ss / dim + eps);
        __m256 scale_vec = _mm256_set1_ps(1 / rms);
        for (int j = 0; j < vec_dim; j += 32) {
            __m256 xr_vec = _mm256_loadu_ps(xr + j);
            __m256 weight_vec = _mm256_loadu_ps(w + j);
            __m256 result =
                _mm256_mul_ps(_mm256_mul_ps(xr_vec, scale_vec), weight_vec);
            _mm256_storeu_ps(outr + j, result);
            xr_vec = _mm256_loadu_ps(xr + j + 8);
            weight_vec = _mm256_loadu_ps(w + j + 8);
            result =
                _mm256_mul_ps(_mm256_mul_ps(xr_vec, scale_vec), weight_vec);
            _mm256_storeu_ps(outr + j + 8, result);
            xr_vec = _mm256_loadu_ps(xr + j + 16);
            weight_vec = _mm256_loadu_ps(w + j + 16);
            result =
                _mm256_mul_ps(_mm256_mul_ps(xr_vec, scale_vec), weight_vec);
            _mm256_storeu_ps(outr + j + 16, result);
            xr_vec = _mm256_loadu_ps(xr + j + 24);
            weight_vec = _mm256_loadu_ps(w + j + 24);
            result =
                _mm256_mul_ps(_mm256_mul_ps(xr_vec, scale_vec), weight_vec);
            _mm256_storeu_ps(outr + j + 24, result);
        }
        for (int j = vec_dim; j < dim; j++) {
            outr[j] = (xr[j] / rms) * w[j];
        }
    }
}
void matvec(const float* x, const float* w, float* out, std::size_t rows,
            std::size_t cols) {
    std::size_t vec_cols = cols - (cols % 32);
#pragma omp parallel for schedule(static) if (rows >= 32)
    for (std::size_t i = 0; i < rows; i++) {
        const float* row = w + i * cols;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (std::size_t j = 0; j < vec_cols; j += 32) {
            __m256 w_vec = _mm256_loadu_ps(row + j);
            __m256 x_vec = _mm256_loadu_ps(x + j);
            acc0 = _mm256_fmadd_ps(w_vec, x_vec, acc0);
            w_vec = _mm256_loadu_ps(row + j + 8);
            x_vec = _mm256_loadu_ps(x + j + 8);
            acc1 = _mm256_fmadd_ps(w_vec, x_vec, acc1);
            w_vec = _mm256_loadu_ps(row + j + 16);
            x_vec = _mm256_loadu_ps(x + j + 16);
            acc2 = _mm256_fmadd_ps(w_vec, x_vec, acc2);
            w_vec = _mm256_loadu_ps(row + j + 24);
            x_vec = _mm256_loadu_ps(x + j + 24);
            acc3 = _mm256_fmadd_ps(w_vec, x_vec, acc3);
        }
        acc0 =
            _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        float buf[8];
        _mm256_storeu_ps(buf, acc0);
        float sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] +
                    buf[6] + buf[7];
        for (std::size_t j = vec_cols; j < cols; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
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
    std::size_t vec_n = n - (n % 32);
    for (std::size_t i = 0; i < vec_n; i += 32) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 result = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(a + i, result);
        a_vec = _mm256_loadu_ps(a + i + 8);
        b_vec = _mm256_loadu_ps(b + i + 8);
        result = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(a + i + 8, result);
        a_vec = _mm256_loadu_ps(a + i + 16);
        b_vec = _mm256_loadu_ps(b + i + 16);
        result = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(a + i + 16, result);
        a_vec = _mm256_loadu_ps(a + i + 24);
        b_vec = _mm256_loadu_ps(b + i + 24);
        result = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(a + i + 24, result);
    }
    for (std::size_t i = vec_n; i < n; i++) {
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
