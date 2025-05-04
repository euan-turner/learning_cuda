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
STATS_OUTPUT = profile_stats.txt

# Default: compile both
all: $(TARGET_ALL) $(TARGET_SINGLE)

$(TARGET_ALL): profile_all_transpose.cu $(KERNEL_SRC)
		$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

$(TARGET_SINGLE): profile_single_transpose.cu $(KERNEL_SRC)
		$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

# Profiling targests
profile-all-kernels: $(TARGET_ALL)
		nsys profile --output=$(PROFILE_OUTPUT)_all $(TARGET_ALL)

# Options: profile-copy, profile-naive, profile-coalesced, profile-coalesced_no_bank_conflicts
profile-%: $(TARGET_SINGLE)
		nsys profile --output=$(PROFILE_OUTPUT)_$* $(TARGET_SINGLE) $*

# Export profiling stats to .txt files
# Options: 
stats-%: $(PROFILE_OUTPUT)_%.nsys-rep
		nsys stats $(PROFILE_OUTPUT)_$*.nsys-rep > $*_$(STATS_OUTPUT)

# Clean up generated files
clean:
		rm -f $(TARGET_ALL) $(TARGET_SINGLE) $(PROFILE_OUTPUT)*.nsys-rep $(PROFILE_OUTPUT)*.sqlite *$(STATS_OUTPUT)

# Phony targets
.PHONY: all profile stats clean