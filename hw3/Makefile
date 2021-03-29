CXX = g++
CXXFLAGS = -fopenmp -std=c++11 -O3 -march=native

RM = rm -f
MKDIRS = mkdir -p

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGETS)

