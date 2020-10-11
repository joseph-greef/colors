CUDA_PATH ?= /usr/local/cuda-10.2
HOST_ARCH   := $(shell uname -m)

LD=$(CUDA_PATH)/bin/nvcc
CC=gcc
CXXFLAGS = -g -DUSE_GPU -std=c++11 -Wall -Werror -Wpedantic -Wextra -Wno-unused-parameter
LDFLAGS = -lpthread -lcuda -lcublas -lcurand -lcudart -lSDL2 -lSDL2_image

# Common includes and paths for CUDA/SDL2
INCLUDES  := -I/usr/include/SDL2 -I$(CUDA_PATH)/include -I../ -I../cuda_kernels
LIBRARIES := -L$(CUDA_PATH)/lib64


################################################################################

# Target rules
all: rulesets.a

hodge.o:hodge.cpp hodge.h ruleset.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

lifelike.o:lifelike.cpp lifelike.h ruleset.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

rainbows.o:rainbows.cpp rainbows.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

ruleset.o:ruleset.cpp ruleset.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

rulesets.a: hodge.o lifelike.o ruleset.o rainbows.o 
	$(LD) $(LIBRARIES) $(LDFLAGS) --lib -o $@ $+

clean:
	rm -f ../rulesets.a *.o
