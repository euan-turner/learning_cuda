# Subdirectory for the matrix transpose project
TRANSPOSE_DIR = matrix_transpose

# Targets for the matrix transpose project
transpose:
		$(MAKE) -C $(TRANSPOSE_DIR)

transpose_profile:
		$(MAKE) -C $(TRANSPOSE_DIR) profile

transpose_stats:
		$(MAKE) -C $(TRANSPOSE_DIR) stats

transpose_clean:
		$(MAKE) -C $(TRANSPOSE_DIR) clean

# Phony targets
.PHONY: transpose transpose_profile transpose_stats transpose_clean