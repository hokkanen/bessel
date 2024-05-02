default: build
	echo "Start Build"

# Accelerator architecture
ifeq ($(CUDA),1)

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
CXXFLAGS = -g -O3 --offload-arch=gfx90a -I/opt/rocm/hiprand/include/ -I/opt/rocm/rocrand/include/
EXE = bessel

else ifeq ($(KOKKOS),CUDA)

# Inputs for Makefile.kokkos
KOKKOS_PATH = $(shell pwd)/kokkos
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -g -O3 -extended-lambda
KOKKOS_DEVICES = "CUDA"
KOKKOS_ARCH = "AMPERE80"
include $(KOKKOS_PATH)/Makefile.kokkos
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
CXXFLAGS = -g -O3 -mp=gpu -gpu=cc80
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
ifeq ($(MPI),MAHTI)

# On Mahti
MPICXX = mpicxx
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(CXX) -DHAVE_MPI $(CXXDEFS) $(CXXFLAGS)'
LDFLAGS += -L${CUDA_PATH}/lib64
LIBS += -lcudart

else ifeq ($(MPI),LUMI)

# On Lumi
MPICXX = CC
MPICXXFLAGS = $(CXXDEFS) -DHAVE_MPI $(CXXFLAGS) -std=c++11 -x hip
LDFLAGS += -L${ROCM_PATH}/lib
LIBS += -lamdhip64

else

MPICXX = $(CXX)
MPICXXFLAGS = $(CXXDEFS) $(CXXFLAGS)

endif

SRC_PATH = src/
SOURCES = $(shell ls src/*.cpp)

OBJ_PATH = src/
OBJECTS = $(shell for file in $(SOURCES);\
		do echo -n $$file | sed -e "s/\(.*\)\.cpp/\1\.o/";echo -n " ";\
		done)

build: $(EXE)

depend:
	makedepend $(CXXDEFS) -m $(SOURCES)

test: $(EXE)
	./$(EXE)

# KOKKOS_DEFINITIONS are outputs from Makefile.kokkos 
$(EXE): $(OBJECTS) $(KOKKOS_LINK_DEPENDS)
	$(MPICXX) $(LDFLAGS) $(OBJECTS) $(LIBS) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -o $(EXE)

# Type 'make clean KOKKOS=CUDA' to clean Kokkos stuff as well
clean: $(CLEAN)
	rm -rf $(OBJECTS) $(EXE) *.tmp desul

# Compilation rules
$(OBJ_PATH)%.o: $(SRC_PATH)%.cpp $(SRC_PATH)arch/%.h $(KOKKOS_CPP_DEPENDS)
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c $< -o $(OBJ_PATH)$(notdir $@)
