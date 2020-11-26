CUDA_PATH ?= /usr/local/cuda-10.2
HOST_ARCH   := $(shell uname -m)

LD=$(CUDA_PATH)/bin/nvcc
CC=gcc
CXXFLAGS = -g -DUSE_GPU -std=c++11 -Wall -Werror -Wpedantic -Wextra -Wno-unused-parameter
LDFLAGS = -lpthread -lcuda -lcublas -lcurand -lcudart -lSDL2 -lSDL2_image
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

# Common includes and paths for CUDA/SDL2
INCLUDES  := -I/usr/include/SDL2 -I$(CUDA_PATH)/include -I$(CURRENT_DIR)
LIBRARIES := -L$(CUDA_PATH)/lib64

SUBDIRS := $(wildcard */.)

################################################################################


# Target rules
all: colors rulesets/rulesets.a cuda_kernels/cuda_kernels.a

cuda_kernels/cuda_kernels.a:
	$(MAKE) -C cuda_kernels/

rulesets/rulesets.a:
	$(MAKE) -C rulesets/

.PHONY: all cuda_kernels/cuda_kernels.a

.PHONY: all rulesets/rulesets.a

game.o:game.cpp game.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

input_manager.o:input_manager.cpp input_manager.h
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

main.o:main.cpp
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

colors: game.o input_manager.o main.o rulesets/rulesets.a cuda_kernels/cuda_kernels.a 
	$(LD) $(LIBRARIES) $(LDFLAGS) -o $@ $+

pdf:
	{ grip README.md 8421 & echo $$! > grip.PID; }
	sleep 2
	wkhtmltopdf http://localhost:8421 readme.pdf
	kill `cat grip.PID`
	rm grip.PID

clean:
	rm -f colors readme.pdf *.o
	$(MAKE) -C rulesets/ clean
	$(MAKE) -C cuda_kernels/ clean

cleanmedia:
	rm -f *.png *.gif *.mp4
