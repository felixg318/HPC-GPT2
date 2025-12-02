NVCC := nvcc
NVCCFLAGS := -std=c++17 -O2 -Idev -DUSE_CUDA 
# Quadro M1200 variants have shipped as both Maxwell (sm_50/52) and Pascal (sm_61).
# Emit code for the common possibilities plus a PTX fallback so the driver JIT can still run it.
CUDA_ARCH := \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_61,code=[sm_61,compute_61]

TARGETS := train_gpt2

.PHONY: all clean

all: $(TARGETS)

cuda_kernels.o: dev/cuda_kernels.cu dev/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

gpt2: gpt2.cpp cuda_kernels.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $^ -o $@

train_gpt2: train_gpt2.cpp cuda_kernels.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $^ -o $@

test_gpt2: test_gpt2.cpp cuda_kernels.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $^ -o $@

test_tokenizer: test_tokenizer.c
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGETS) cuda_kernels.o
