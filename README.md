# COSC 4397 Final Project: CUDA 2D Heat Diffusion

## Project Summary

This project implements a 2D heat diffusion simulation using the Jacobi stencil method. The main goal is to compare a CPU reference implementation against a student-written CUDA GPU implementation.

The grid has fixed boundary values. The top boundary is hot, and the other boundaries are cold. Each interior cell is updated using the average of its four direct neighbors.

## Proposal

We will implement a 2D heat diffusion Jacobi stencil kernel in CUDA, validate against a CPU implementation, compare performance against the CPU baseline on multiple grid sizes, and fall back to a simpler single-kernel stencil version if needed.

## Platform

## to run the program
make clean
make
make run
./build/heat 2048 1000 all "" csv

- CUDA
- NVIDIA GPU server
- `nvcc`
- GNU Make

## Build