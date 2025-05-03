# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++11 -O2

# Executable name
TARGET = profile_transpose

# Source files
SRC = profile_transpose.cu copy_kernel.cu naive_transpose_kernel.cu coalesced_transpose_kernel.cu coalesced_transpose_no_bank_conflicts_kernel.cu
LIBS = -lnvToolsExt

# Output files for profiling
PROFILE_OUTPUT = $(TARGET)
STATS_OUTPUT = $(TARGET)_stats.txt

# Default target: compile the program
all: $(TARGET)


# Compile the program
$(TARGET): $(SRC)
		$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

# Run profiling with nsys
profile: $(TARGET)
		nsys profile --stats=true --output=$(PROFILE_OUTPUT) ./$(TARGET)

# Export profiling stats to a .txt file
stats: $(PROFILE_OUTPUT).nsys-rep
		nsys stats --force-export=true $(PROFILE_OUTPUT).nsys-rep > $(STATS_OUTPUT)

# Clean up generated files
clean:
		rm -f $(TARGET) $(PROFILE_OUTPUT).nsys-rep $(PROFILE_OUTPUT).sqlite $(STATS_OUTPUT)

# Phony targets
.PHONY: all profile stats clean