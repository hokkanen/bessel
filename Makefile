default: build
	echo "Start Build"

# Accelerator architecture
ifeq ($(CUDA),1)

CXX = nvcc
CXXDEFS = -DHAVE_CUDA
CXXFLAGS = -g -O3 --x=cu --extended-lambda -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80
EXE = bessel

else ifeq ($(HIP),CUDA)

CXX = hipcc
CXXDEFS = -DHAVE_HIP
CXXFLAGS = -g -O3 --x=cu --extended-lambda -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -I$(shell pwd)/hiprand
EXE = bessel

else ifeq ($(HIP),ROCM)

CXX = hipcc
CXXDEFS = -DHAVE_HIP
CXXFLAGS = -g -O3 --offload-arch=gfx90a -I/opt/rocm/hiprand/include/ -I/opt/rocm/rocrand/include/
FILETYPE = .cpp
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
ifeq ($(MPI),1)

# On Puhti
MPICXX = mpicxx
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(CXX) -DHAVE_MPI $(CXXDEFS) $(CXXFLAGS)'
LDFLAGS += -L${CUDA_PATH}/lib64
LIBS += -lcudart

# On Lumi
# MPICXX = CC
# MPICXXFLAGS = $(CXXDEFS) -DHAVE_MPI $(CXXFLAGS) -std=c++11 -x hip
# LDFLAGS += -L${ROCM_PATH}/lib
# LIBS += -lamdhip64

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

$(EXE): $(OBJECTS)
	$(MPICXX) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $(EXE)

clean: $(CLEAN)
	rm -f $(OBJECTS) $(EXE)

# Compilation rules
$(OBJ_PATH)%.o: $(SRC_PATH)%.cpp
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c $< -o $(SRC_PATH)$(notdir $@)
