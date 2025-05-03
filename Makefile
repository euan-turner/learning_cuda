TRANSPOSE_DIR = matrix_transpose
SGEMM_DIR = sgemm

# TODO: Restructure Makefiles, commands defined at top level, targets below
# Targets for matrix transpose
transpose:
		$(MAKE) -C $(TRANSPOSE_DIR)

transpose_profile:
		$(MAKE) -C $(TRANSPOSE_DIR) profile

transpose_stats:
		$(MAKE) -C $(TRANSPOSE_DIR) stats

transpose_clean:
		$(MAKE) -C $(TRANSPOSE_DIR) clean

# Targets for SGEMM
sgemm:
		$(MAKE) -C $(SGEMM_DIR)
sgemm_profile:
		$(MAKE) -C $(SGEMM_DIR) profile
sgemm_stats:
		$(MAKE) -C $(SGEMM_DIR) stats
sgemm_clean:
		$(MAKE) -C $(SGEMM_DIR) clean
# Phony targets
.PHONY: transpose transpose_profile transpose_stats transpose_clean