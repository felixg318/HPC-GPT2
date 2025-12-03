#ifdef USE_CUDA

#include "cuda_utils.h"

#include <float.h>
#include <math.h>

// Simple RAII wrapper so cudaFree always runs, even on early exits.
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }
    T* get() const { return ptr; }
    T** addr() { return &ptr; }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer() = default;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
};

template <typename T>
bool cuda_alloc(DeviceBuffer<T>& buffer, size_t bytes) {
    return CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(buffer.addr()), bytes));
}

// --- Block reduction helpers (single CTA) ---
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // up to 32 warps per block
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__inline__ __device__ float block_reduce_max(float val) {
    __shared__ float shared[32];  // up to 32 warps per block
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}


// --- Matmul kernels ---
__global__ void matmul2d_kernel(const float* A,
                                const float* B,
                                float* C,
                                int N,
                                int C1,
                                int C2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < C2) {
        float sum = 0.0f;
        const float* a_ptr = A + row * C1;
        const float* b_ptr = B + col;
        for (int k = 0; k < C1; ++k) {
            sum += a_ptr[k] * b_ptr[k * C2];
        }
        C[row * C2 + col] = sum;
    }
}

__global__ void matmul3d_kernel(const float* A,
                                const float* B,
                                float* C,
                                int Bdim,
                                int T,
                                int C1,
                                int C2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int b = row / T;
    int t = row - b * T;

    if (b < Bdim && t < T && col < C2) {
        int a_offset = (b * T + t) * C1;
        int b_offset = b * C1 * C2;
        float sum = 0.0f;
        for (int k = 0; k < C1; ++k) {
            sum += A[a_offset + k] * B[b_offset + k * C2 + col];
        }
        C[(b * T + t) * C2 + col] = sum;
    }
}

// --- Matmul backward kernels ---
__global__ void matmul2d_backward_dA(const float* grad_C,
                                     const float* B,
                                     float* grad_A,
                                     int N, int C1, int C2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < C1) {
        float sum = 0.0f;
        for (int k = 0; k < C2; ++k) {
            sum += grad_C[row * C2 + k] * B[col * C2 + k];
        }
        grad_A[row * C1 + col] += sum;
    }
}

__global__ void matmul2d_backward_dB(const float* grad_C,
                                     const float* A,
                                     float* grad_B,
                                     int N, int C1, int C2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < C1 && col < C2) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[k * C1 + row] * grad_C[k * C2 + col];
        }
        grad_B[row * C2 + col] += sum;
    }
}

__global__ void matmul3d_backward_dA(const float* grad_C,
                                     const float* B,
                                     float* grad_A,
                                     int Bdim, int T, int C1, int C2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // over B*T
    int col = blockIdx.x * blockDim.x + threadIdx.x; // over C1
    int b = row / T;
    int t = row - b * T;
    if (b < Bdim && t < T && col < C1) {
        float sum = 0.0f;
        for (int k = 0; k < C2; ++k) {
            sum += grad_C[(b * T + t) * C2 + k] * B[(b * C1 + col) * C2 + k];
        }
        grad_A[(b * T + t) * C1 + col] += sum;
    }
}

__global__ void matmul3d_backward_dB(const float* grad_C,
                                     const float* A,
                                     float* grad_B,
                                     int Bdim, int T, int C1, int C2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // over C1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // over C2
    int b = blockIdx.z;
    if (b < Bdim && row < C1 && col < C2) {
        float sum = 0.0f;
        for (int t = 0; t < T; ++t) {
            sum += A[(b * T + t) * C1 + row] * grad_C[(b * T + t) * C2 + col];
        }
        grad_B[(b * C1 + row) * C2 + col] += sum;
    }
}

// --- Softmax kernel (rows x cols layout) ---
__global__ void softmax_rows_kernel(const float* X,
                                    float* Y,
                                    int rows,
                                    int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int offset = row * cols;

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = X[offset + c];
        if (v > local_max) local_max = v;
    }
    float max_val = block_reduce_max(local_max);
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = max_val;
    }
    __syncthreads();
    max_val = shared_max;

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_sum += expf(X[offset + c] - max_val);
    }
    float sum_val = block_reduce_sum(local_sum);
    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = sum_val;
    }
    __syncthreads();
    sum_val = shared_sum;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        Y[offset + c] = expf(X[offset + c] - max_val) / sum_val;
    }
}

__global__ void softmax_rows_backward_kernel(const float* Y,
                                             const float* grad_Y,
                                             float* grad_X,
                                             int rows,
                                             int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int offset = row * cols;

    // dot = sum_c Y * grad_Y
    float local_dot = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_dot += Y[offset + c] * grad_Y[offset + c];
    }
    float dot = block_reduce_sum(local_dot);
    __shared__ float shared_dot;
    if (threadIdx.x == 0) shared_dot = dot;
    __syncthreads();
    dot = shared_dot;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float y_val = Y[offset + c];
        float gy = grad_Y[offset + c];
        grad_X[offset + c] += y_val * (gy - dot);
    }
}

// --- Host wrappers ---
bool cuda_matmul_2d(const float* A, const float* B, float* C,
                    int N, int C1, int C2) {
#if CUDA_USE_MANAGED
    dim3 block(16, 16);
    dim3 grid((C2 + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    matmul2d_kernel<<<grid, block>>>(A, B, C, N, C1, C2);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    DeviceBuffer<float> dA;
    DeviceBuffer<float> dB;
    DeviceBuffer<float> dC;
    bool ok = true;

    size_t bytesA = static_cast<size_t>(N) * C1 * sizeof(float);
    size_t bytesB = static_cast<size_t>(C1) * C2 * sizeof(float);
    size_t bytesC = static_cast<size_t>(N) * C2 * sizeof(float);

    do {
        if (!(ok = cuda_alloc(dA, bytesA))) break;
        if (!(ok = cuda_alloc(dB, bytesB))) break;
        if (!(ok = cuda_alloc(dC, bytesC))) break;

        if (!(ok = CUDA_CHECK(cudaMemcpy(dA.get(), A, bytesA, cudaMemcpyHostToDevice)))) break;
        if (!(ok = CUDA_CHECK(cudaMemcpy(dB.get(), B, bytesB, cudaMemcpyHostToDevice)))) break;

        dim3 block(16, 16);
        dim3 grid((C2 + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y);
        matmul2d_kernel<<<grid, block>>>(dA.get(), dB.get(), dC.get(), N, C1, C2);
        if (!(ok = CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize()))) break;

        ok = CUDA_CHECK(cudaMemcpy(C, dC.get(), bytesC, cudaMemcpyDeviceToHost));
    } while (false);

    return ok;
#endif
}


bool cuda_matmul_3d(const float* A, const float* B, float* C,
                    int Bdim, int T, int C1, int C2) {
#if CUDA_USE_MANAGED
    dim3 block(16, 16);
    dim3 grid((C2 + block.x - 1) / block.x,
              ((Bdim * T) + block.y - 1) / block.y);
    matmul3d_kernel<<<grid, block>>>(A, B, C, Bdim, T, C1, C2);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    DeviceBuffer<float> dA;
    DeviceBuffer<float> dB;
    DeviceBuffer<float> dC;
    bool ok = true;

    size_t bytesA = static_cast<size_t>(Bdim) * T * C1 * sizeof(float);
    size_t bytesB = static_cast<size_t>(Bdim) * C1 * C2 * sizeof(float);
    size_t bytesC = static_cast<size_t>(Bdim) * T * C2 * sizeof(float);

    do {
        if (!(ok = cuda_alloc(dA, bytesA))) break;
        if (!(ok = cuda_alloc(dB, bytesB))) break;
        if (!(ok = cuda_alloc(dC, bytesC))) break;

        if (!(ok = CUDA_CHECK(cudaMemcpy(dA.get(), A, bytesA, cudaMemcpyHostToDevice)))) break;
        if (!(ok = CUDA_CHECK(cudaMemcpy(dB.get(), B, bytesB, cudaMemcpyHostToDevice)))) break;

        dim3 block(16, 16);
        dim3 grid((C2 + block.x - 1) / block.x,
                  ((Bdim * T) + block.y - 1) / block.y);
        matmul3d_kernel<<<grid, block>>>(dA.get(), dB.get(), dC.get(), Bdim, T, C1, C2);
        if (!(ok = CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize()))) break;

        ok = CUDA_CHECK(cudaMemcpy(C, dC.get(), bytesC, cudaMemcpyDeviceToHost));
    } while (false);

    return ok;
#endif
}

bool cuda_matmul_2d_backward(const float* grad_C,
                             const float* A,
                             const float* B,
                             float* grad_A,
                             float* grad_B,
                             int N, int C1, int C2) {
#if CUDA_USE_MANAGED
    dim3 block(16, 16);
    dim3 grid_dA((C1 + block.x - 1) / block.x,
                 (N + block.y - 1) / block.y);
    matmul2d_backward_dA<<<grid_dA, block>>>(grad_C, B, grad_A, N, C1, C2);
    dim3 grid_dB((C2 + block.x - 1) / block.x,
                 (C1 + block.y - 1) / block.y);
    matmul2d_backward_dB<<<grid_dB, block>>>(grad_C, A, grad_B, N, C1, C2);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)grad_C; (void)A; (void)B; (void)grad_A; (void)grad_B; (void)N; (void)C1; (void)C2;
    return false;
#endif
}

bool cuda_matmul_3d_backward(const float* grad_C,
                             const float* A,
                             const float* B,
                             float* grad_A,
                             float* grad_B,
                             int Bdim, int T, int C1, int C2) {
#if CUDA_USE_MANAGED
    dim3 block(16, 16);
    dim3 grid_dA((C1 + block.x - 1) / block.x,
                 ((Bdim * T) + block.y - 1) / block.y);
    matmul3d_backward_dA<<<grid_dA, block>>>(grad_C, B, grad_A, Bdim, T, C1, C2);
    dim3 grid_dB((C2 + block.x - 1) / block.x,
                 (C1 + block.y - 1) / block.y,
                 Bdim);
    matmul3d_backward_dB<<<grid_dB, block>>>(grad_C, A, grad_B, Bdim, T, C1, C2);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)grad_C; (void)A; (void)B; (void)grad_A; (void)grad_B; (void)Bdim; (void)T; (void)C1; (void)C2;
    return false;
#endif
}


bool cuda_softmax_2d(const float* X, float* Y, int N, int C) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid(N);
    softmax_rows_kernel<<<grid, block>>>(X, Y, N, C);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    DeviceBuffer<float> dX;
    DeviceBuffer<float> dY;
    bool ok = true;

    size_t bytes = static_cast<size_t>(N) * C * sizeof(float);
    do {
        if (!(ok = cuda_alloc(dX, bytes))) break;
        if (!(ok = cuda_alloc(dY, bytes))) break;

        if (!(ok = CUDA_CHECK(cudaMemcpy(dX.get(), X, bytes, cudaMemcpyHostToDevice)))) break;

        dim3 block(256);
        dim3 grid(N);
        softmax_rows_kernel<<<grid, block>>>(dX.get(), dY.get(), N, C);
        if (!(ok = CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize()))) break;

        ok = CUDA_CHECK(cudaMemcpy(Y, dY.get(), bytes, cudaMemcpyDeviceToHost));
    } while (false);

    return ok;
#endif
}


bool cuda_softmax_3d(const float* X, float* Y, int Bdim, int T, int C) {
    int rows = Bdim * T;
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid(rows);
    softmax_rows_kernel<<<grid, block>>>(X, Y, rows, C);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    DeviceBuffer<float> dX;
    DeviceBuffer<float> dY;
    bool ok = true;

    size_t bytes = static_cast<size_t>(rows) * C * sizeof(float);
    do {
        if (!(ok = cuda_alloc(dX, bytes))) break;
        if (!(ok = cuda_alloc(dY, bytes))) break;

        if (!(ok = CUDA_CHECK(cudaMemcpy(dX.get(), X, bytes, cudaMemcpyHostToDevice)))) break;

        dim3 block(256);
        dim3 grid(rows);
        softmax_rows_kernel<<<grid, block>>>(dX.get(), dY.get(), rows, C);
        if (!(ok = CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize()))) break;

        ok = CUDA_CHECK(cudaMemcpy(Y, dY.get(), bytes, cudaMemcpyDeviceToHost));
    } while (false);

    return ok;
#endif
}

bool cuda_softmax_2d_backward(const float* Y, const float* grad_Y, float* grad_X, int N, int C) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid(N);
    softmax_rows_backward_kernel<<<grid, block>>>(Y, grad_Y, grad_X, N, C);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)Y; (void)grad_Y; (void)grad_X; (void)N; (void)C;
    return false;
#endif
}

bool cuda_softmax_3d_backward(const float* Y, const float* grad_Y, float* grad_X, int Bdim, int T, int C) {
#if CUDA_USE_MANAGED
    int rows = Bdim * T;
    dim3 block(256);
    dim3 grid(rows);
    softmax_rows_backward_kernel<<<grid, block>>>(Y, grad_Y, grad_X, rows, C);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)Y; (void)grad_Y; (void)grad_X; (void)Bdim; (void)T; (void)C;
    return false;
#endif
}


// --- GELU ---
__global__ void gelu_kernel(const float* X, float* Y, int n) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = X[idx];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
        float t = tanhf(inner);
        Y[idx] = 0.5f * x * (1.0f + t);
    }
}

__global__ void gelu_backward_kernel(const float* X, const float* grad_Y, float* grad_X, int n) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = X[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
        float t = tanhf(inner);
        float sech2 = 1.0f - t * t;
        float inner_deriv = SQRT_2_OVER_PI * (1.0f + 0.044715f * 3.0f * x2);
        float grad = 0.5f * (1.0f + t) + 0.5f * x * sech2 * inner_deriv;
        grad_X[idx] += grad_Y[idx] * grad;
    }
}

bool cuda_gelu(const float* X, float* Y, int n) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    gelu_kernel<<<grid, block>>>(X, Y, n);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    DeviceBuffer<float> dX;
    DeviceBuffer<float> dY;
    bool ok = true;

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    do {
        if (!(ok = cuda_alloc(dX, bytes))) break;
        if (!(ok = cuda_alloc(dY, bytes))) break;

        if (!(ok = CUDA_CHECK(cudaMemcpy(dX.get(), X, bytes, cudaMemcpyHostToDevice)))) break;

        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        gelu_kernel<<<grid, block>>>(dX.get(), dY.get(), n);
        if (!(ok = CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize()))) break;

        ok = CUDA_CHECK(cudaMemcpy(Y, dY.get(), bytes, cudaMemcpyDeviceToHost));
    } while (false);

    return ok;
#endif
}

bool cuda_gelu_backward(const float* X, const float* grad_Y, float* grad_X, int n) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    gelu_backward_kernel<<<grid, block>>>(X, grad_Y, grad_X, n);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)X; (void)grad_Y; (void)grad_X; (void)n;
    return false;
#endif
}


// --- LayerNorm ---
__global__ void layernorm_rows_kernel(const float* X,
                                      float* Y,
                                      const float* gamma,
                                      const float* beta,
                                      int cols,
                                      float eps) {
    int row = blockIdx.x;
    int offset = row * cols;

    // mean
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_sum += X[offset + c];
    }
    float mean = block_reduce_sum(local_sum) / cols;
    __shared__ float shared_mean;
    if (threadIdx.x == 0) shared_mean = mean;
    __syncthreads();
    mean = shared_mean;

    // variance
    float local_var = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float diff = X[offset + c] - mean;
        local_var += diff * diff;
    }
    float var = block_reduce_sum(local_var) / cols;
    __shared__ float shared_inv_std;
    if (threadIdx.x == 0) shared_inv_std = rsqrtf(var + eps);
    __syncthreads();
    float inv_std = shared_inv_std;

    // normalize + affine
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float norm = (X[offset + c] - mean) * inv_std;
        Y[offset + c] = norm * gamma[c] + beta[c];
    }
}

__global__ void layernorm_rows_backward_kernel(const float* X,
                                               const float* grad_Y,
                                               const float* gamma,
                                               float* grad_X,
                                               float* grad_gamma,
                                               float* grad_beta,
                                               int cols,
                                               float eps) {
    int row = blockIdx.x;
    int offset = row * cols;

    // mean
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_sum += X[offset + c];
    }
    float mean = block_reduce_sum(local_sum) / cols;
    __shared__ float shared_mean;
    if (threadIdx.x == 0) shared_mean = mean;
    __syncthreads();
    mean = shared_mean;

    // variance
    float local_var = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float diff = X[offset + c] - mean;
        local_var += diff * diff;
    }
    float var = block_reduce_sum(local_var) / cols;
    __shared__ float shared_inv_std;
    if (threadIdx.x == 0) shared_inv_std = rsqrtf(var + eps);
    __syncthreads();
    float inv_std = shared_inv_std;

    // grad_gamma and grad_beta accumulation (per-column atomic)
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float norm = (X[offset + c] - mean) * inv_std;
        float gy = grad_Y[offset + c];
        atomicAdd(&grad_gamma[c], gy * norm);
        atomicAdd(&grad_beta[c], gy);
    }

    // grad_X
    float dnorm_dx_sum = 0.0f;
    float dnorm_dx_mul_x_minus_mean_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float dx_hat = grad_Y[offset + c] * gamma[c];
        dnorm_dx_sum += dx_hat;
        dnorm_dx_mul_x_minus_mean_sum += dx_hat * (X[offset + c] - mean);
    }
    float sum1 = block_reduce_sum(dnorm_dx_sum);
    float sum2 = block_reduce_sum(dnorm_dx_mul_x_minus_mean_sum);
    __shared__ float shared_sum1;
    __shared__ float shared_sum2;
    if (threadIdx.x == 0) {
        shared_sum1 = sum1;
        shared_sum2 = sum2;
    }
    __syncthreads();
    sum1 = shared_sum1;
    sum2 = shared_sum2;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float dx_hat = grad_Y[offset + c] * gamma[c];
        float term1 = cols * dx_hat;
        float term2 = sum1;
        float term3 = (X[offset + c] - mean) * sum2 * inv_std * inv_std;
        grad_X[offset + c] += (term1 - term2 - term3) * inv_std / cols;
    }
}

// --- Embedding backward ---
__global__ void embedding_backward_2d_kernel(const int* idx,
                                             const float* grad_out,
                                             float* grad_weight,
                                             int B, int T, int dim,
                                             int vocab_size) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T;
    if (pos < total) {
        int token = idx[pos];
        if (token >= 0 && token < vocab_size) {
            int base_out = pos * dim;
            int base_w = token * dim;
            for (int d = 0; d < dim; ++d) {
                atomicAdd(&grad_weight[base_w + d], grad_out[base_out + d]);
            }
        }
    }
}

// Cross-entropy grad: grad_logits = (probs - onehot) / N
__global__ void cross_entropy_backward_3d_kernel(const float* probs,
                                                 const int* targets,
                                                 float* grad_logits,
                                                 int total_positions,
                                                 int V,
                                                 float invN) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < total_positions) {
        int target = targets[pos];
        int base = pos * V;
        for (int v = 0; v < V; ++v) {
            float prob = probs[base + v];
            float grad = prob - (v == target ? 1.0f : 0.0f);
            grad_logits[base + v] += grad * invN;
        }
    }
}

__global__ void adam_step_kernel(float* param, float* grad, float* m, float* v,
                                 int n, float lr_t, float beta1, float beta2, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * g * g;
        m[idx] = m_val;
        v[idx] = v_val;
        param[idx] -= lr_t * m_val / (sqrtf(v_val) + eps);
    }
}

__global__ void causal_mask_kernel(float* att, int B, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // over B*T*T
    int total = B * T * T;
    if (idx < total) {
        int tmp = idx;
        int tk = tmp % T;
        tmp /= T;
        int tq = tmp % T;
        int b = tmp / T;
        if (b < B && tq < T && tk < T && tk > tq) {
            att[idx] = -1e30f;
        }
    }
}

__global__ void concat_head_kernel(const float* head, float* concat,
                                   int B, int T, int head_size,
                                   int head_idx, int n_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // over B*T*head_size
    int total = B * T * head_size;
    if (idx < total) {
        int d = idx % head_size;
        int tmp = idx / head_size;
        int t = tmp % T;
        int b = tmp / T;
        int out_offset = (b * T + t) * (head_size * n_heads) + head_idx * head_size + d;
        concat[out_offset] = head[idx];
    }
}

__global__ void transpose_last2_kernel(const float* X, float* Y,
                                       int B, int T1, int T2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // over B*T1*T2
    int total = B * T1 * T2;
    if (idx < total) {
        int tmp = idx;
        int t2 = tmp % T2;
        tmp /= T2;
        int t1 = tmp % T1;
        int b = tmp / T1;
        int out_idx = (b * T2 + t2) * T1 + t1;
        Y[out_idx] = X[idx];
    }
}

__global__ void zero_buffer_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 0.0f;
}

__global__ void grad_norm_accum_kernel(const float* grad, int n, float* sum) {
    __shared__ float shmem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (idx < n) {
        float g = grad[idx];
        val = g * g;
    }
    int lane = threadIdx.x;
    shmem[lane] = val;
    __syncthreads();
    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shmem[lane] += shmem[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        atomicAdd(sum, shmem[0]);
    }
}

__global__ void grad_scale_kernel(float* grad, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad[idx] *= scale;
}
bool cuda_layernorm_rows(const float* X, float* Y,
                         const float* gamma, const float* beta,
                         int rows, int cols, float eps) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid(rows);
    layernorm_rows_kernel<<<grid, block>>>(X, Y, gamma, beta, cols, eps);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    DeviceBuffer<float> dX;
    DeviceBuffer<float> dY;
    DeviceBuffer<float> dG;
    DeviceBuffer<float> dB;
    bool ok = true;

    size_t bytes_data = static_cast<size_t>(rows) * cols * sizeof(float);
    size_t bytes_params = static_cast<size_t>(cols) * sizeof(float);

    do {
        if (!(ok = cuda_alloc(dX, bytes_data))) break;
        if (!(ok = cuda_alloc(dY, bytes_data))) break;
        if (!(ok = cuda_alloc(dG, bytes_params))) break;
        if (!(ok = cuda_alloc(dB, bytes_params))) break;

        if (!(ok = CUDA_CHECK(cudaMemcpy(dX.get(), X, bytes_data, cudaMemcpyHostToDevice)))) break;
        if (!(ok = CUDA_CHECK(cudaMemcpy(dG.get(), gamma, bytes_params, cudaMemcpyHostToDevice)))) break;
        if (!(ok = CUDA_CHECK(cudaMemcpy(dB.get(), beta, bytes_params, cudaMemcpyHostToDevice)))) break;

        dim3 block(256);
        dim3 grid(rows);
        layernorm_rows_kernel<<<grid, block>>>(dX.get(), dY.get(), dG.get(), dB.get(), cols, eps);
        if (!(ok = CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize()))) break;

        ok = CUDA_CHECK(cudaMemcpy(Y, dY.get(), bytes_data, cudaMemcpyDeviceToHost));
    } while (false);

    return ok;
#endif
}

bool cuda_layernorm_rows_backward(const float* X, const float* grad_Y,
                                  const float* gamma,
                                  float* grad_X, float* grad_gamma, float* grad_beta,
                                  int rows, int cols, float eps) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid(rows);
    layernorm_rows_backward_kernel<<<grid, block>>>(X, grad_Y, gamma, grad_X, grad_gamma, grad_beta, cols, eps);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)X; (void)grad_Y; (void)gamma; (void)grad_X; (void)grad_gamma; (void)grad_beta; (void)rows; (void)cols; (void)eps;
    return false;
#endif
}

bool cuda_layernorm_2d(const float* X, float* Y,
                       const float* gamma, const float* beta,
                       int N, int C, float eps) {
    return cuda_layernorm_rows(X, Y, gamma, beta, N, C, eps);
}

bool cuda_layernorm_3d(const float* X, float* Y,
                       const float* gamma, const float* beta,
                       int Bdim, int T, int C, float eps) {
    int rows = Bdim * T;
    return cuda_layernorm_rows(X, Y, gamma, beta, rows, C, eps);
}

bool cuda_layernorm_2d_backward(const float* X, const float* grad_Y,
                                const float* gamma,
                                float* grad_X, float* grad_gamma, float* grad_beta,
                                int N, int C, float eps) {
    return cuda_layernorm_rows_backward(X, grad_Y, gamma, grad_X, grad_gamma, grad_beta, N, C, eps);
}

bool cuda_layernorm_3d_backward(const float* X, const float* grad_Y,
                                const float* gamma,
                                float* grad_X, float* grad_gamma, float* grad_beta,
                                int Bdim, int T, int C, float eps) {
    int rows = Bdim * T;
    return cuda_layernorm_rows_backward(X, grad_Y, gamma, grad_X, grad_gamma, grad_beta, rows, C, eps);
}

bool cuda_embedding_backward_2d(const int* idx, const float* grad_out,
                                float* grad_weight,
                                int B, int T, int vocab_size, int dim) {
#if CUDA_USE_MANAGED
    int total = B * T;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    embedding_backward_2d_kernel<<<grid, block>>>(idx, grad_out, grad_weight, B, T, dim, vocab_size);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)idx; (void)grad_out; (void)grad_weight; (void)B; (void)T; (void)vocab_size; (void)dim;
    return false;
#endif
}

bool cuda_cross_entropy_backward_3d(const float* probs, const int* targets,
                                    float* grad_logits,
                                    int B, int T, int V) {
#if CUDA_USE_MANAGED
    int total = B * T;
    dim3 block(128);
    dim3 grid((total + block.x - 1) / block.x);
    float invN = 1.0f / (float)(total);
    cross_entropy_backward_3d_kernel<<<grid, block>>>(probs, targets, grad_logits, total, V, invN);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)probs; (void)targets; (void)grad_logits; (void)B; (void)T; (void)V;
    return false;
#endif
}

bool cuda_adam_step(float* param, float* grad, float* m, float* v,
                    int n, float lr_t, float beta1, float beta2, float eps) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    adam_step_kernel<<<grid, block>>>(param, grad, m, v, n, lr_t, beta1, beta2, eps);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)param; (void)grad; (void)m; (void)v; (void)n; (void)lr_t; (void)beta1; (void)beta2; (void)eps;
    return false;
#endif
}

bool cuda_apply_causal_mask(float* att, int B, int T) {
#if CUDA_USE_MANAGED
    int total = B * T * T;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    causal_mask_kernel<<<grid, block>>>(att, B, T);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)att; (void)B; (void)T;
    return false;
#endif
}

bool cuda_concat_head(const float* head, float* concat,
                      int B, int T, int head_size, int head_idx, int n_heads) {
#if CUDA_USE_MANAGED
    int total = B * T * head_size;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    concat_head_kernel<<<grid, block>>>(head, concat, B, T, head_size, head_idx, n_heads);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)head; (void)concat; (void)B; (void)T; (void)head_size; (void)head_idx; (void)n_heads;
    return false;
#endif
}

bool cuda_transpose_last2(const float* X, float* Y, int B, int T1, int T2) {
#if CUDA_USE_MANAGED
    int total = B * T1 * T2;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    transpose_last2_kernel<<<grid, block>>>(X, Y, B, T1, T2);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)X; (void)Y; (void)B; (void)T1; (void)T2;
    return false;
#endif
}

bool cuda_transpose_last2_backward(const float* grad_Y, float* grad_X, int B, int T1, int T2) {
#if CUDA_USE_MANAGED
    int total = B * T1 * T2;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    // backward is transpose of grad; swap T1/T2
    transpose_last2_kernel<<<grid, block>>>(grad_Y, grad_X, B, T2, T1);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)grad_Y; (void)grad_X; (void)B; (void)T1; (void)T2;
    return false;
#endif
}

bool cuda_zero_buffer(float* data, int n) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    zero_buffer_kernel<<<grid, block>>>(data, n);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)data; (void)n;
    return false;
#endif
}

bool cuda_grad_norm_accum(const float* grad, int n, float* sum) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    grad_norm_accum_kernel<<<grid, block>>>(grad, n, sum);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)grad; (void)n; (void)sum;
    return false;
#endif
}

bool cuda_grad_scale(float* grad, int n, float scale) {
#if CUDA_USE_MANAGED
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    grad_scale_kernel<<<grid, block>>>(grad, n, scale);
    return CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)grad; (void)n; (void)scale;
    return false;
#endif
}

#endif  // USE_CUDA
