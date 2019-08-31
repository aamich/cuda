# Wayne
# toy code for cuda, matrix multiplication
# 2019-08-29

#seven “core” automatic variables:
#
#    $@: The filename representing the target.
#    $%: The filename element of an archive member specification.
#    $<: The filename of the first prerequisite.
#    $?: The names of all prerequisites that are newer than the target, separated by spaces.
#    $^: The filenames of all the prerequisites, separated by spaces. This list has duplicate filenames removed since for most uses, such as compiling, copying, etc., duplicates are not wanted.
#    $+: Similar to $^, this is the names of all the prerequisites separated by spaces, except that $+ includes duplicates. This variable was created for specific situations such as arguments to linkers where duplicate values have meaning.
#    $*: The stem of the target filename. A stem is typically a filename without its suffix. Its use outside of pattern rules is discouraged.
# Location of the CUDA Toolkit
#
CUDA_PATH ?= /usr/local/cuda-10.1

# architecture
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Common includes and paths for CUDA
INCLUDES  := -I../common/inc
LIBRARIES :=
################################################################################

# Target rules
all: build

build: matrixMul

matrixMul.o:matrixMul.cu
	$(NVCC) $(INCLUDES) -o $@ -std=c++11 -c $<

matrixMul: matrixMul.o
	$(NVCC) -o $@ $+ 
	
run: build
	./matrixMul

clean:
	rm -f matrixMul matrixMul.o

clobber: clean

