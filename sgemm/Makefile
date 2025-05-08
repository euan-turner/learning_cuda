# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++17 -O2

# Executable name
TARGET = profile_sgemm

# Source files
SRC = naive_sgemm.cu coalesced_sgemm.cu
LIBS = -lnvToolsExt

# Output files for profiling
PROFILE_OUTPUT = $(TARGET)

# Default target: compile the program
all: $(TARGET)


# Compile the program
$(TARGET): profile_sgemm.cu $(SRC)
		$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

# Check kernel implementations
check: check_sgemm.cu $(SRC)
		$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)
		./check

# Run profiling with nsys
profile: $(TARGET)
		../nsys_easy -o $(PROFILE_OUTPUT) $(TARGET)

profile_single_sgemm: profile_single_sgemm.cu $(SRC)
		$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

profile-%: profile_single_sgemm
		../nsys_easy -o $(PROFILE_OUTPUT)-$* ./profile_single_sgemm $*

# Clean up generated files
clean:
		rm -f $(TARGET) $(PROFILE_OUTPUT)*.nsys-rep $(PROFILE_OUTPUT)*.sqlite check profile_single_sgemm

# Phony targets
.PHONY: all profile stats clean