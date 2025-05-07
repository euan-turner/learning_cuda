# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++11 -O2

TARGET_ALL = profile_all_transpose
TARGET_SINGLE = profile_single_transpose

# Shared src files
KERNEL_SRC = copy_kernel.cu naive_transpose_kernel.cu coalesced_transpose_kernel.cu coalesced_transpose_no_bank_conflicts_kernel.cu
LIBS = -lnvToolsExt

# Output files for profiling
PROFILE_OUTPUT = profile_results

# Default: compile both
all: $(TARGET_ALL) $(TARGET_SINGLE)

$(TARGET_ALL): profile_all_transpose.cu $(KERNEL_SRC)
		$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

$(TARGET_SINGLE): profile_single_transpose.cu $(KERNEL_SRC)
		$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

# Profiling targests
profile-all-kernels: $(TARGET_ALL)
		../nsys_easy -o $(PROFILE_OUTPUT) $(TARGET_ALL)

# Options: profile-copy, profile-naive, profile-coalesced, profile-coalesced_no_bank_conflicts
profile-%: $(TARGET_SINGLE)
		../nsys_easy -o $(PROFILE_OUTPUT)-$* $(TARGET_SINGLE) $*

# Clean up generated files
clean:
		rm -f $(TARGET_ALL) $(TARGET_SINGLE) $(PROFILE_OUTPUT)*.nsys-rep $(PROFILE_OUTPUT)*.sqlite

# Phony targets
.PHONY: all profile stats clean