################################################################################
# MPI + CUDA build for stage3 (hybrid data-parallel training on GPU)
################################################################################

# CUDA toolchain configuration (align with stage2 CUDA setup)
CUDA_PATH      ?= /usr/local/cuda-12.8
MPI_CXX        ?= mpicxx
HOST_COMPILER  ?= /usr/bin/g++-14

# GPU architectures to target (override as needed)
SMS            ?= 50 52 61

# MPI include/link flags (so we can use a specific host compiler)
MPI_CXXFLAGS   := $(shell $(MPI_CXX) --showme:compile)
# Keep MPI link flags simple to avoid nvcc choking on -Wl, options
MPI_LDFLAGS    := -L/usr/lib64/openmpi/lib -lmpi

# NVCC flags
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCCFLAGS      := -O2 -std=c++17 --threads 0 -Xcompiler -fPIE
EXTRA_NVCCFLAGS += -DUSE_CUDA -DUSE_MPI -Wno-deprecated-gpu-targets

# Build gencode flags from SMS list
GENCODE_FLAGS :=
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

INCLUDES  := $(MPI_CXXFLAGS)
LIBRARIES := $(MPI_LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart

TARGETS := train_gpt2 inference

.PHONY: all clean

all: train_gpt2

cuda_kernels.o: cuda_kernels.cu cuda_utils.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

train_gpt2: train_gpt2.cpp cuda_kernels.o
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $^ -o $@ $(LIBRARIES)

inference: inference.cpp cuda_kernels.o
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $^ -o $@ $(LIBRARIES)

clean:
	rm -f $(TARGETS) cuda_kernels.o
