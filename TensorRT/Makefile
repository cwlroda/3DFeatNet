# This assumes that the TensorRT git folder is in the same parent directory as the 3DFeatNet directory.
TENSORRT := ../../TensorRT/include \
		../../TensorRT/quickstart/common \
		/usr/local/cuda-11.3/include \
		./ops/grouping ./ops/sign ./
#$(shell find ../../TensorRT/plugin/ -maxdepth 2 -type d) \

SRCFILES := $(wildcard ../../TensorRT/quickstart/common/*.cpp) \
			$(wildcard ./ops/*/*.cpp)

CUSTOM_OPS := $(shell pwd)/ops/grouping/build $(shell pwd)/ops/sign/build

CXXFLAGS := -std=c++17 -Wno-deprecated-declarations

LIBPATHS := /usr/local/cuda-11.3/lib64/ /usr/local/cuda-11.3/lib/ \
			 /usr/lib/x86_64-linux-gnu/ \
			../../TensorRT/build/out \
			$(CUSTOM_OPS)

# Above link paths are needed for TensorRT's basic operation.
# The other two link paths are the custom ops that are registered.
LINKPATHS := cudart cuda cublas nvinfer nvinfer_plugin \
			PointNetGroupingOps SignOps 

.PHONY: all clean

all: inference_trt ld_path

inference_trt: inference_trt.cpp inference_3dfn.cpp
	g++ $(CXXFLAGS) -o inference_trt \
	inference_trt.cpp inference_3dfn.cpp \
	$(SRCFILES) \
	$(addprefix -I, $(TENSORRT)) \
	$(addprefix -L, $(LIBPATHS)) \
	$(addprefix -l, $(LINKPATHS))

# This is necessary for the linker to call the custom ops at runtime.
ld_path:
	@echo "Adding custom libs to LD_LIBRARY_PATH..."
	$(shell LD_LIBRARY_PATH="$(addsuffix :, $(CUSTOM_OPS)):$LD_LIBRARY_PATH")

clean:
	@echo "I am cleaning..."
	rm -f inference_trt