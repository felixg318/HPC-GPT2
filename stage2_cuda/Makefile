################################################################################
# CUDA configuration (adapted from the NVIDIA sample Makefile)
################################################################################

# ---- Overrides ----
CUDA_PATH      ?= /usr/local/cuda-12.8
HOST_COMPILER  := /usr/bin/g++-14

# GPU architectures (Quadro M1200 spans Maxwell/Pascal)
SMS            ?= 50 52 61

# Quiet the “pre-75 will be removed” warning and add project defs
EXTRA_NVCCFLAGS += -Wno-deprecated-gpu-targets -DUSE_CUDA -Idev

# Optional: prefix commands (e.g., EXEC=@ for quiet build)
EXEC ?=#include "mpi_utils.h"


################################################################################
# Host / device flags
################################################################################

HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
TARGET_SIZE := $(shell getconf LONG_BIT)

NVCC        := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS   := -m$(TARGET_SIZE)
CCFLAGS     := -O2 -std=c++17
LDFLAGS     :=

# Build gencode flags from SMS list
GENCODE_FLAGS :=
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

ALL_CCFLAGS  :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += --threads 0 -Xcompiler -fPIE

INCLUDES  :=
LIBRARIES := -L$(CUDA_PATH)/lib64 -lcudart

################################################################################
# Targets
################################################################################

TARGETS := train_gpt2 inference

.PHONY: all clean clobber

all: train_gpt2

cuda_kernels.o: cuda_kernels.cu cuda_utils.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

train_gpt2: train_gpt2.cpp cuda_kernels.o
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@ $(LIBRARIES)

inference: inference.cpp cuda_kernels.o
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@ $(LIBRARIES)

clean:
	rm -f $(TARGETS) cuda_kernels.o
