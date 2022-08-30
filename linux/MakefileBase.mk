include ../ComputeCapability.mk

### output binary
PLATFORM      = $(shell arch)
TARGET_DIR    = ../../bin/gcc_$(PLATFORM)_$(CONFIGURATION)
TARGET        = $(TARGET_DIR)/$(TARGET_NAME)
TARGET_BENCH  = $(TARGET)_benchmark

$(eval TARGET_LOW_NAME := $(shell echo KROS_$(TARGET_NAME) | tr A-Z a-z))
LIB_NAME      = lib$(TARGET_LOW_NAME).so
LIB_TARGET    = $(TARGET_DIR)/$(LIB_NAME)

### cpp source codes (.cpp)
SRCDIR         = ../../module
MAIN_SRC       = $(SRCDIR)/$(TARGET_NAME)_Main/$(TARGET_NAME)_main.cpp
MAIN_BENCH_SRC = $(SRCDIR)/$(TARGET_NAME)_Main/$(TARGET_NAME)_benchmark.cpp
LIB_SRCS       = $(wildcard $(SRCDIR)/$(TARGET_NAME)/*.cpp)
LIB_SRCS      += $(wildcard $(SRCDIR)/TRT/*.cpp)
LIB_SRCS      += $(wildcard $(SRCDIR)/Common/*.cpp)

### cuda source codes (.cu)
LIB_CUSRCS  = $(wildcard $(SRCDIR)/$(TARGET_NAME)/*.cu)
LIB_CUSRCS += $(wildcard $(SRCDIR)/Common/*.cu)

### object files
MAIN_OBJ       = $(MAIN_SRC:.cpp=.o)
MAIN_BENCH_OBJ = $(MAIN_BENCH_SRC:.cpp=.o)
LIB_OBJS       = $(LIB_SRCS:.cpp=.o)
LIB_OBJS      += $(LIB_CUSRCS:.cu=.o)
OBJS           = $(LIB_OBJS) $(MAIN_OBJ) $(MAIN_BENCH_OBJ)

### root user availability
SUDO           = $(shell which sudo)

### common flags
OPTFLAGS     = -O2
ifeq ($(CONFIGURATION), Debug)
	OPTFLAGS = -O0 -g
endif

### compiled by gcc 
CXX         = g++-9
CXXFLAGS    = $(OPTFLAGS) -Wall -fPIC -std=c++1z -mtune=native -march=native -fopenmp -D_REENTRANT 
INCLUDE     = -I"/usr/local/cuda/include" -I"../../include"

### compiled by nvcc
NVCC        = nvcc
NVCCFLAGS   = $(OPTFLAGS) -std=c++11 $(COMPUTE_CAPABILITY) -Xcompiler "$(OPTFLAGS) -Wall -fPIC" 
NVCCINCLUDE = -I"/usr/local/cuda/include" -I"../../include"

### linked by nvcc
LDFLAGS     = -lnvparsers -lnvonnxparser -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lgomp -l$(TARGET_LOW_NAME) -Xcompiler \"-Wl,-rpath,.,-rpath,$(TARGET_DIR)\"
LDPATH      = -L"/usr/local/cuda/lib64" -L"$(TARGET_DIR)"

### suffix rules
.SUFFIXES : .cpp .cu .o

.PHONY : all
all : $(TARGET) $(TARGET_BENCH)

.PHONY : clean
clean :
	rm -f $(TARGET) $(OBJS) $(LIB_TARGET) $(TARGET_BENCH)

.PHONY : install
install :
	$(SUDO) mkdir -p /usr/local/include/kros/trt/TRT
	$(SUDO) mkdir -p /usr/local/include/kros/trt/$(TARGET_NAME)
	$(SUDO) cp ../../include/kros/trt.h /usr/local/include/kros
	$(SUDO) cp -r ../../include/kros/trt/TRT /usr/local/include/kros/trt
	$(SUDO) cp -r ../../include/kros/trt/$(TARGET_NAME) /usr/local/include/kros/trt
	$(SUDO) cp $(LIB_TARGET) /usr/local/lib/
	$(SUDO) ldconfig

.PHONY : uninstall
uninstall :
	$(SUDO) rm -f /usr/local/include/kros/trt.h
	$(SUDO) rm -rf /usr/local/include/kros/trt/TRT
	$(SUDO) rm -rf /usr/local/include/kros/trt/$(TARGET_NAME)
	$(SUDO) rm -f /usr/local/lib/$(LIB_NAME)
	$(SUDO) ldconfig

$(TARGET): $(MAIN_OBJ) $(LIB_TARGET)
	$(NVCC) $(LDFLAGS) $(LDPATH) -o $@ $<
	
$(TARGET_BENCH): $(MAIN_BENCH_OBJ) $(LIB_TARGET)
	$(NVCC) $(LDFLAGS) $(LDPATH) -o $@ $<
	
$(LIB_TARGET): $(LIB_OBJS) 
	@if [ ! -d $(TARGET_DIR) ]; \
		then mkdir -p $(TARGET_DIR); \
	fi
	$(NVCC) -shared -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDE) -o $@ -c $<
