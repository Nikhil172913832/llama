#include "ops.h"
#include <cstddef>
#include <math.h>
using namespace std;
void rmsnorm(const float *x, const float *w, float *out, float eps, int rows,
             int ed) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * ed;
        float *outr = out + (size_t)r * ed;
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
        const float *row = w + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            temp += row[j] * x[j];
        }
        out[i] = temp;
    }
}

void softmax(float *x, int n) {
    float maxVal = x[0];
    for (int i = 1; i < n; i++) {
        maxVal = max(maxVal, x[i]);
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
