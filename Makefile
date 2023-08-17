# Makefile

# Compiler
NVCC := nvcc

# Directories
SRC_DIR := lib
BUILD_DIR := build

# CUDA Compiler Flags
CUDA_FLAGS := # -arch=sm_61
CUDA_INCLUDE := -I$(SRC_DIR)/gen_spike_vector_from_dense -I$(SRC_DIR)/spike_vector_matmul
COMMON_FLAGS := -Xcompiler -fPIC

# Files
SRC_FILES := $(wildcard $(SRC_DIR)/spike_vector_matmul/*.cu) $(wildcard $(SRC_DIR)/gen_spike_vector_from_dense/*.cu)
HEADER_FILES := $(SRC_DIR)/spike_vector_matmul/spike_vector_matmul_gpu.h $(SRC_DIR)/gen_spike_vector_from_dense/spike_vector_from_dense_gpu.h
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_FILES))

# Shared Object
SHARED_OBJECT := $(BUILD_DIR)/ssax_shared_object.so

# Target
all: $(SHARED_OBJECT)

$(SHARED_OBJECT): $(OBJ_FILES)
	$(NVCC) -shared -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADER_FILES)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(COMMON_FLAGS) -o $@ -c $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
