# nom du binaire
TARGET = device

# fichiers source
SRC = device.cu

# compilateur CUDA
NVCC = nvcc

# options de compilation
NVCC_FLAGS = -O2

# règle par défaut
all: $(TARGET)

# compilation du binaire
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

# nettoyer les fichiers compilés
clean:
	rm -f $(TARGET)
