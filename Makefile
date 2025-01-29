# Compiler
NVCC = nvcc

# Directories
INCLUDE_DIR = include
SRC_DIR = src
CPU_DIR = $(INCLUDE_DIR)/CPU

# Output executable
TARGET = cacachavecuda

# Source files
CPU_SOURCES = $(CPU_DIR)/Int.cpp $(CPU_DIR)/IntMod.cpp $(CPU_DIR)/Point.cpp $(CPU_DIR)/SECP256K1.cpp
CUDA_SOURCES = $(SRC_DIR)/GPUSecp.cu $(SRC_DIR)/sha256_cuda.cu $(SRC_DIR)/ripemd160_cuda.cu $(SRC_DIR)/bitcoin_address.cu

# All source files
SOURCES = $(CPU_SOURCES) $(CUDA_SOURCES)

# Compilation flags
CFLAGS = -I$(INCLUDE_DIR) -rdc=true
LDFLAGS = -lsecp256k1

# Default target
all: $(TARGET)

# Link and build the executable
$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)