# Compiler
NVCC = nvcc

# Directories
INCLUDE_DIR = include
SRC_DIR = src

# Output executable
TARGET = bitcoin_address_cuda

# Source files
SOURCES = $(SRC_DIR)/sha256_cuda.cu $(SRC_DIR)/ripemd160_cuda.cu $(SRC_DIR)/bitcoin_address.cu

# Compilation flags
CFLAGS = -I$(INCLUDE_DIR)

# Default target
all: $(TARGET)

# Link and build the executable
$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCES)

# Clean target
clean:
	rm -f $(TARGET)
