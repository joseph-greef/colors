CUDA_PATH ?= /usr/local/cuda-10.2
HOST_ARCH   := $(shell uname -m)

CC=$(CUDA_PATH)/bin/nvcc
CUDAFLAGS= -std=c++11 -dc 
LDFLAGS= -lpthread -lcuda -lcublas -lcurand -lcudart -lSDL2

# Common includes and paths for CUDA/SDL2
INCLUDES  := -I/usr/include/SDL2 -I$(CUDA_PATH)/include -I../
LIBRARIES := -L$(CUDA_PATH)/lib64


all:cuda_kernels.a

#common_cu.o:common.cu common.cuh
#	$(CC) $(INCLUDES) $(CUDAFLAGS) -o $@ -c $<

hodge_cu.o:hodge.cu hodge.cuh
	$(CC) $(INCLUDES) $(CUDAFLAGS) -o $@ -c $<

lifelike_cu.o:lifelike.cu lifelike.cuh
	$(CC) $(INCLUDES) $(CUDAFLAGS) -o $@ -c $<

cuda_kernels.a: hodge_cu.o lifelike_cu.o #
	$(CC) $(LIBRARIES) $(LDFLAGS) --lib -o $@ $+ 

clean:
	rm -f *.o cuda_kernels.a
