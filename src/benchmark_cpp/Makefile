CUDNN_PATH := cudnn
SPUNIT_DIR := /usr/local/sputnik
INCLUDES := `python3 -m pybind11 --includes` -I$(CUDNN_PATH)/include 
NVCC := /usr/local/cuda/bin/nvcc
CUDA_TOOLKIT := /usr/local/cuda/
NVCC_INC := -I/usr/local/cuda/include
LIBS         := -lcudart -lcusparse -lcublas
CXX_FLAGS := -std=c++11 -g -O2
LIBRARY_DIR := cpp_lib.cu
SPUNIT_LIB :=  -L$(SPUNIT_DIR)/lib -lsputnik
SPUNIT_INC := -I$(SPUNIT_DIR)/include
all:
	${NVCC} ${NVCC_INC}  $(CXX_FLAGS) $(LIBRARY_DIR) -shared -Xcompiler -fPIC $(INCLUDES) $(SPUNIT_INC) -o cpp_lib`python3-config --extension-suffix` $(LIBS) $(SPUNIT_LIB)

clean:
	rm *.so
	rm *.o