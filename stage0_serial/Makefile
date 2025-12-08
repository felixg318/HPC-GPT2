CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
TARGETS := train_gpt2 inference

.PHONY: all clean

all: $(TARGETS)

train_gpt2: train_gpt2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

inference: inference.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
