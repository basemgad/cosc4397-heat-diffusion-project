.RECIPEPREFIX := >

NVCC := nvcc
TARGET := build/heat
SRC := src/main.cu

NVCCFLAGS := -O3 -std=c++17 -m64 -lineinfo

all: $(TARGET)

$(TARGET): $(SRC)
>@mkdir -p build
>@$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: all
>@mkdir -p results
>@./$(TARGET) 2048 1000 all results/heat_2048.ppm

bench: all
>@mkdir -p results
>@./scripts/run_benchmarks.sh

clean:
>@rm -rf build

.PHONY: all run bench clean
