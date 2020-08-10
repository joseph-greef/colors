CUDA_PATH ?= /usr/local/cuda-10.2
HOST_ARCH   := $(shell uname -m)

CC=$(CUDA_PATH)/bin/nvcc
CXXFLAGS= -g -std=c++11
#CUDAFLAGS= -std=c++11 -c 
LDFLAGS= -lpthread -lcuda -lcublas -lcurand -lcudart -lSDL2
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

# Common includes and paths for CUDA/SDL2
INCLUDES  := -I/usr/include/SDL2 -I$(CUDA_PATH)/include -I$(CURRENT_DIR)
LIBRARIES := -L$(CUDA_PATH)/lib64

SUBDIRS := $(wildcard */.)

################################################################################


# Target rules
all: colors rulesets/rulesets.a


rulesets/rulesets.a:
	$(MAKE) -C rulesets/

.PHONY: all rulesets/rulesets.a

#kernel.o:kernel.cu kernel.cuh
#	$(NVCC) $(INCLUDES) $(CUDAFLAGS) -o $@ -c $<

#board.o:board.cpp board.h
#	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

game.o:game.cpp game.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

initializer.o:initializer.cpp initializer.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

main.o:main.cpp
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

#RuleGenerator.o:RuleGenerator.cpp RuleGenerator.h
#	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

#screen.o:screen.cpp screen.h
#	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

colors: game.o initializer.o main.o rulesets/rulesets.a #kernel.o board.o screen.o RuleGenerator.o
	$(CC) $(LIBRARIES) $(LDFLAGS) -o $@ $+ 

clean:
	rm -f colors *.o
	$(MAKE) -C rulesets/ clean
