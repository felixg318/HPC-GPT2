STAGE0 := stage0_serial   # CPU-only build with g++
STAGE1 := stage1_mpi      # MPI-only build with mpicxx
STAGE2 := stage2_cuda     # CUDA build with nvcc
STAGE3 := stage3_mpi-cuda # MPI + CUDA build with nvcc/mpicxx

STAGES := $(STAGE0) $(STAGE1) $(STAGE2) $(STAGE3)
TARGETS := train_gpt2 inference

.PHONY: all $(STAGES) $(TARGETS) clean stage0 stage1 stage2 stage3

# Build both train_gpt2 and inference for every stage
all: $(STAGES)

# Stage-specific builds (use each stage's Makefile/toolchain)
$(STAGES):
	$(MAKE) -C $@

stage0: $(STAGE0)
stage1: $(STAGE1)
stage2: $(STAGE2)
stage3: $(STAGE3)

# Build a specific binary across all stages
train_gpt2 inference:
	for dir in $(STAGES); do \
		$(MAKE) -C $$dir $@; \
	done

# Clean everything
clean:
	for dir in $(STAGES); do \
		$(MAKE) -C $$dir clean; \
	done
