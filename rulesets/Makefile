CUDA_PATH ?= /usr/local/cuda-10.2
HOST_ARCH   := $(shell uname -m)

LD=$(CUDA_PATH)/bin/nvcc
CXX=g++
CXXFLAGS = -g -DUSE_GPU -std=c++11 -Wall -Werror -Wpedantic -Wextra -Wno-unused-parameter
LDFLAGS = -lpthread -lcuda -lcublas -lcurand -lcudart -lSDL2 -lSDL2_image

# Common includes and paths for CUDA/SDL2
INCLUDES  := -I/usr/include/SDL2 -I$(CUDA_PATH)/include -I../ -I../cuda_kernels
LIBRARIES := -L$(CUDA_PATH)/lib64


################################################################################

# Target rules
all: rulesets.a

ants.o:ants/ants.cpp ants/ants.h ruleset.h ants/ant_location.h
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

colony.o:ants/colony.cpp ants/colony.h ants/ant_location.h
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

hodge.o:hodge.cpp hodge.h ruleset.h
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

initializer.o:initializer.cpp initializer.h
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

lifelike.o:lifelike.cpp lifelike.h ruleset.h
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

rainbows.o:rainbows.cpp rainbows.h
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

ruleset.o:ruleset.cpp ruleset.h
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ -c $<

rulesets.a: ants.o colony.o hodge.o initializer.o lifelike.o ruleset.o rainbows.o 
	$(LD) $(LIBRARIES) $(LDFLAGS) --lib -o $@ $+

clean:
	rm -f rulesets.a *.o
