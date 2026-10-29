void rmsnorm(const float *x, const float *w, float *out, float eps, int rows,
             int ed);
void matvec(const float *x, const float *w, float *out, int rows, int cols);
void softmax(float *x, int n);
