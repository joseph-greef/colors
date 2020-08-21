CUDA_PATH ?= /usr/local/cuda-10.2
HOST_ARCH   := $(shell uname -m)

CC=$(CUDA_PATH)/bin/nvcc
CXXFLAGS= -O3 -std=c++11 -DUSE_GPU
#CUDAFLAGS= -std=c++11 -c 
LDFLAGS= -lpthread -lcuda -lcublas -lcurand -lcudart -lSDL2

# Common includes and paths for CUDA/SDL2
INCLUDES  := -I/usr/include/SDL2 -I$(CUDA_PATH)/include -I../ -I cuda_kernels
LIBRARIES := -L$(CUDA_PATH)/lib64


################################################################################


# Target rules
all: rulesets.a cuda_kernels/cuda_kernels.a

.PHONY: all cuda_kernels/cuda_kernels.a

cuda_kernels/cuda_kernels.a:
	$(MAKE) -C cuda_kernels/

lifelike.o:lifelike.cpp lifelike.h ruleset.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

rainbows.o:rainbows.cpp rainbows.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

ruleset.o:ruleset.cpp ruleset.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

rulesets.a: lifelike.o ruleset.o rainbows.o 
	$(CC) $(LIBRARIES) $(LDFLAGS) --lib -o $@ $+ 

clean:
	rm -f ../rulesets.a *.o
	$(MAKE) -C cuda_kernels/ clean
