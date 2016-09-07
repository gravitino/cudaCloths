all: single openmp gpu

single:  pbd.cu
	nvcc pbd.cu -lGL -lGLU -lglut -lGLEW -std=c++11 -O3 -arch=sm_30 -Xcompiler="-march=native" -o single

openmp: pbd.cu
	nvcc pbd.cu -lGL -lGLU -lglut -lGLEW -std=c++11 -O3 -arch=sm_30 -Xcompiler="-fopenmp  -march=native" -o openmp

gpu: pbd.cu
	nvcc pbd.cu -lGL -lGLU -lglut -lGLEW -std=c++11 -O3 -arch=sm_30 -Xcompiler="-march=native" -o gpu -DGPU
