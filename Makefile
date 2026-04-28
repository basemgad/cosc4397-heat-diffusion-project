.RECIPEPREFIX := >

NVCC := nvcc
TARGET := build/heat
SRC := src/main.cu

NVCCFLAGS := -O3 -std=c++17 -m64 -lineinfo

all: $(TARGET)

$(TARGET): $(SRC)
>mkdir -p build
>$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: all
>./$(TARGET) 512 500

bench: all
>./scripts/run_benchmarks.sh

clean:
>rm -rf build

.PHONY: all run bench clean
