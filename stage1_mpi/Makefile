################################################################################
# MPI-only build for stage1 (CPU data-parallel training)
################################################################################

MPI_CXX      ?= mpicxx
CXXFLAGS     := -O3 -std=c++17 -Wall -Wextra -DUSE_MPI

# Pull include/link flags from the MPI wrapper so the underlying compiler/linker
# get the right paths when MPICH/OpenMPI are installed in custom locations.
MPI_CXXFLAGS := $(shell ( $(MPI_CXX) -show 2>/dev/null || echo "" ) | tr ' ' '\n' | grep -E '^-I|^-D|^-pthread' | tr '\n' ' ')
MPI_LDFLAGS  := $(shell ( $(MPI_CXX) -show 2>/dev/null || echo "" ) | tr ' ' '\n' | grep -E '^-L|^-l' | tr '\n' ' ')

INCLUDES  := $(MPI_CXXFLAGS)
LIBRARIES := $(MPI_LDFLAGS)

TARGETS := train_gpt2 inference

.PHONY: all clean

all: $(TARGETS)

train_gpt2: train_gpt2.cpp
	$(MPI_CXX) $(INCLUDES) $(CXXFLAGS) $< -o $@ $(LIBRARIES)

inference: inference.cpp
	$(MPI_CXX) $(INCLUDES) $(CXXFLAGS) $< -o $@ $(LIBRARIES)

clean:
	rm -f $(TARGETS)
