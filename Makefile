STAGES := stage0_serial stage1_mpi stage2_cuda stage3_mpi-cuda
TARGETS := train_gpt2 inference
GROUP_SERIAL := stage0_serial
GROUP_MPI := stage1_mpi
GROUP_CUDA := stage2_cuda
GROUP_MPI_CUDA := stage3_mpi-cuda

.PHONY: all $(STAGES) $(TARGETS) clean serial mpi cuda mpi-cuda

all: $(STAGES)

serial: $(GROUP_SERIAL)
mpi: $(GROUP_MPI)
cuda: $(GROUP_CUDA)
mpi-cuda: $(GROUP_MPI_CUDA)

# Build both train_gpt2 and inference inside each stage directory
$(STAGES):
	$(MAKE) -C $@

# Convenience targets to build a specific binary across all stages
$(TARGETS):
	for dir in $(STAGES); do \
		$(MAKE) -C $$dir $@; \
	done

clean:
	for dir in $(STAGES); do \
		$(MAKE) -C $$dir clean; \
	done
