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
all: ../rulesets.a

lifelike.o:lifelike.cpp lifelike.h ruleset.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

../rulesets.a: lifelike.o 
	$(NVCC) $(LIBRARIES) $(LDFLAGS) --lib -o $@ $+ 

clean:
	rm -f ../rulesets.a *.o
