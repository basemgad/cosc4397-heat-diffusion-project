NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17

TARGET := build/heat
SRC := src/main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -rf build

.PHONY: all run clean

