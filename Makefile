default: build
	echo "Start Build"

# Accelerator architecture
ifeq ($(ACC),CUDA)

CXX = nvc++
CXXFLAGS = -g -O3 -acc -gpu=cc80 -Minfo=accel
LDFLAGS += -acc -gpu=cc80
EXE = bessel

else ifeq ($(CUDA),1)

CXX = nvcc
CXXDEFS = -DHAVE_CUDA
CXXFLAGS = -g -O3 -x=cu -extended-lambda -gencode=arch=compute_80,code=sm_80
EXE = bessel

else ifeq ($(HIP),CUDA)

CXX = hipcc
CXXDEFS = -DHAVE_HIP
CXXFLAGS = -g -O3 -x=cu -extended-lambda -gencode=arch=compute_80,code=sm_80
EXE = bessel

else ifeq ($(HIP),ROCM)

CXX = hipcc
CXXDEFS = -DHAVE_HIP
CXXFLAGS = -g -O3 -x hip --offload-arch=gfx90a -I/opt/rocm/hiprand/include/ -I/opt/rocm/rocrand/include/
EXE = bessel

else ifeq ($(KOKKOS),CUDA)

# Inputs for Makefile.kokkos
KOKKOS_PATH = $(shell pwd)/kokkos
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -g -O3 -extended-lambda
KOKKOS_DEVICES = "CUDA"
KOKKOS_ARCH = "AMPERE80"
include $(KOKKOS_PATH)/Makefile.kokkos
# Remove any ldl flags that do not start with -L
KOKKOS_LDFLAGS := $(filter -L%,$(KOKKOS_LDFLAGS))
# Other
CLEAN = kokkos-clean
CXXDEFS = -DHAVE_KOKKOS
EXE = bessel

else ifeq ($(KOKKOS),ROCM)

# Inputs for Makefile.kokkos
KOKKOS_PATH = $(shell pwd)/kokkos
CXX = hipcc
CXXFLAGS = -g -O3
KOKKOS_DEVICES = "HIP"
KOKKOS_ARCH = "VEGA90A"
include $(KOKKOS_PATH)/Makefile.kokkos
# Remove any ldl flags that do not start with -L
KOKKOS_LDFLAGS := $(filter -L%,$(KOKKOS_LDFLAGS))
# Other
CLEAN = kokkos-clean
CXXDEFS = -DHAVE_KOKKOS
EXE = bessel

else ifeq ($(KOKKOS),OMP)

# Inputs for Makefile.kokkos
KOKKOS_PATH = $(shell pwd)/kokkos
CXX = g++
CXXFLAGS = -g -O3
KOKKOS_DEVICES = "OPENMP"
include $(KOKKOS_PATH)/Makefile.kokkos
# Other
CLEAN = kokkos-clean
CXXDEFS = -DHAVE_KOKKOS
EXE = bessel

else ifeq ($(OMP),CUDA)

CXX = nvc++
CXXFLAGS = -g -O3 -mp=gpu -gpu=cc80 -Minfo=mp
# Huge performance loss without linker flags
LDFLAGS += -mp=gpu -gpu=cc80
EXE = bessel

else ifeq ($(SYCL),CUDA)

CXX = icpx
CXXDEFS = -DHAVE_SYCL
CXXFLAGS = -g -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80
LDFLAGS += -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80
LIBS += -lsycl
EXE = bessel

else ifeq ($(SYCL),HOST)

CXX = icpx
CXXDEFS = -DHAVE_SYCL
CXXFLAGS = -g -O3 -fsycl
LDFLAGS += -fsycl
LIBS += -lsycl
EXE = bessel

else

CXX = g++
CXXFLAGS = -g -O3
EXE = bessel

endif

# Plot results with Matplot
ifeq ($(MATPLOT),1)

CXXDEFS += -DHAVE_MATPLOT
CXXFLAGS += -std=c++17 -I$(shell pwd)/matplot/include/
LDFLAGS += -L$(shell pwd)/matplot/lib64/ -L$(shell pwd)/matplot/lib64/Matplot++/
LIBS += -lmatplot -ljpeg -ltiff -lz -lpng -lnodesoup

endif

# Message passing protocol
ifeq ($(MPI),OMPI)

# On Mahti
MPICXX = mpicxx
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(CXX) -DHAVE_MPI $(CXXDEFS) $(CXXFLAGS)'
LDFLAGS += -L${CUDA_PATH}/lib64
LIBS += -lcudart

else ifeq ($(MPI),CRAY)

# On Lumi
ifeq ($(CXX),g++)
CXX = CC
endif
MPICXX = CC
MPICXXFLAGS = $(CXXDEFS) -DHAVE_MPI -g -O3
LDFLAGS += -L${ROCM_PATH}/lib
LIBS += -lamdhip64

else

MPICXX = $(CXX)
MPICXXFLAGS = $(CXXDEFS) $(CXXFLAGS)

endif

SRC_PATH = src/
SOURCES = $(wildcard $(SRC_PATH)*.cpp)
HEADERS = $(wildcard $(SRC_PATH)*.h) $(wildcard $(SRC_PATH)arch/*.h)

OBJ_PATH = $(SRC_PATH)
OBJECTS = $(SOURCES:$(SRC_PATH)%.cpp=$(OBJ_PATH)%.o)


build: $(EXE)

depend:
	makedepend $(CXXDEFS) -m $(SOURCES)

test: $(EXE)
	./$(EXE)

# KOKKOS_DEFINITIONS are outputs from Makefile.kokkos 
$(EXE): $(OBJECTS) $(KOKKOS_LINK_DEPENDS)
	$(MPICXX) $(OBJECTS) $(LDFLAGS) $(LIBS) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -o $(EXE)

# Type 'make clean KOKKOS=CUDA' to clean Kokkos stuff as well
clean: $(CLEAN)
	rm -rf $(OBJECTS) $(EXE) *.o *.tmp desul

# Rule for compiling comms.cpp with $(MPICXX)
$(OBJ_PATH)comms.o: $(SRC_PATH)comms.cpp $(HEADERS)
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c $< -o $@

# Rule for compiling other bessel.cpp with $(CXX)
$(OBJ_PATH)bessel.o: $(SRC_PATH)bessel.cpp $(HEADERS) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(CXXDEFS) $(CXXFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c $< -o $@
