STAGES := stage0_serial stage1_mpi stage2_cuda stage3_mpi-cuda

.PHONY: all $(STAGES) clean

all: $(STAGES)

$(STAGES):
	$(MAKE) -C $@

clean:
	for dir in $(STAGES); do \
		$(MAKE) -C $$dir clean; \
	done
