CC := gcc
CXX := g++

MPIC := mpicc
MPICXX := mpic++

# Compiler flags
CFLAGS := -Wall -Wextra -O2 -g -fPIC -I.
CXXFLAGS := -Wall -Wextra -O2 -g -std=c++11 -fPIC -I.

# Link flags
LDFLAGS := -lm

TARGET := llm

C_SOURCES := embedding.c attention.c debug.c
CPP_SOURCES := main.cpp naive_tokenizor.cpp
HEADERS := tokenizor.hpp

OBJECTS := $(C_SOURCES:.c=.o) $(CPP_SOURCES:.cpp=.o)

all: $(TARGET)

# Link object files to create executable - USE MPICXX HERE!
$(TARGET): $(OBJECTS)
	$(MPICXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile C++ source files
%.o: %.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
