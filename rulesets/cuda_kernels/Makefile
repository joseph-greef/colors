CUDA_PATH ?= /usr/local/cuda-10.2
HOST_ARCH   := $(shell uname -m)

CC=$(CUDA_PATH)/bin/nvcc
CXXFLAGS= -O3 -std=c++11 -DUSE_GPU
CUDAFLAGS= -std=c++11 -dc 
LDFLAGS= -lpthread -lcuda -lcublas -lcurand -lcudart -lSDL2

# Common includes and paths for CUDA/SDL2
INCLUDES  := -I/usr/include/SDL2 -I$(CUDA_PATH)/include -I../
LIBRARIES := -L$(CUDA_PATH)/lib64


all:cuda_kernels.a

#common_cu.o:common.cu common.cuh
#	$(CC) $(INCLUDES) $(CUDAFLAGS) -o $@ -c $<

lifelike_cu.o:lifelike.cu lifelike.cuh
	$(CC) $(INCLUDES) $(CUDAFLAGS) -o $@ -c $<

cuda_kernels.a: lifelike_cu.o #
	$(CC) $(LIBRARIES) $(LDFLAGS) --lib -o $@ $+ 

clean:
	rm *.o cuda_kernels.a
