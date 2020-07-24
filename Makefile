CUDA_PATH ?= /usr/local/cuda-10.2
HOST_ARCH   := $(shell uname -m)

CC=g++
NVCC=$(CUDA_PATH)/bin/nvcc
CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS= -std=c++11 -c 
LDFLAGS= -lpthread -lcuda -lcublas -lcurand -lcudart -lSDL2

# Common includes and paths for CUDA/SDL2
INCLUDES  := -I/usr/include/SDL2 -I$(CUDA_PATH)/include
LIBRARIES := -L$(CUDA_PATH)/lib64

################################################################################


# Target rules
all: colors

kernel.o:kernel.cu kernel.cuh
	$(NVCC) $(INCLUDES) $(CUDAFLAGS) -o $@ -c $<

board.o:board.cpp board.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

initializer.o:initializer.cpp initializer.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

main.o:main.cpp
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

RuleGenerator.o:RuleGenerator.cpp RuleGenerator.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

screen.o:screen.cpp screen.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

colors: kernel.o board.o initializer.o main.o RuleGenerator.o screen.o 
	$(NVCC) $(LIBRARIES) $(LDFLAGS) -o $@ $+ 

clean:
	rm -f colors *.o
