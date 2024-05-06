default: build
	echo "Start Build"

# Accelerator architecture
ifeq ($(CUDA),1)

CXX = nvcc
CXXDEFS = -DHAVE_CUDA
CXXFLAGS = -g -O3 -x=cu -extended-lambda -gencode=arch=compute_80,code=sm_80
FILETYPE = .c
EXE = bessel

else ifeq ($(HIP),CUDA)

CXX = hipcc
CXXDEFS = -DHAVE_HIP
CXXFLAGS = -g -O3 -x=cu -extended-lambda -gencode=arch=compute_80,code=sm_80
FILETYPE = .cpp
EXE = bessel

else ifeq ($(HIP),ROCM)

CXX = hipcc
CXXDEFS = -DHAVE_HIP 
CXXFLAGS = -g -O3 --offload-arch=gfx90a -I/opt/rocm/hiprand/include/ -I/opt/rocm/rocrand/include/
FILETYPE = .cpp
EXE = bessel

else ifeq ($(OMP),CUDA)

CXX = nvc
CXXFLAGS = -g -O3 -mp=gpu -gpu=cc80
FILETYPE = .c
EXE = bessel

else

CXX = gcc
CXXFLAGS = -g -O3
FILETYPE = .c
EXE = bessel

endif

# Memory manager
ifeq ($(UMPIRE),1)

CXXDEFS += -DHAVE_UMPIRE
CXXFLAGS += -I$(shell pwd)/umpire/include/
LDFLAGS += -L$(shell pwd)/umpire/lib/
LIBS += -lcamp -lumpire

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
MPICXX = CC
MPICXXFLAGS = $(CXXDEFS) -DHAVE_MPI $(CXXFLAGS) -std=c++17 -x hip
LDFLAGS += -L${ROCM_PATH}/lib
LIBS += -lamdhip64

else

MPICXX = $(CXX)
MPICXXFLAGS = $(CXXDEFS) $(CXXFLAGS)

endif

# Create temporary .cpp files if needed (for HIP only)
ifeq ($(FILETYPE),.cpp)
$(shell for file in `ls src/*.c`;\
		do cp -- "$$file" "$${file%.c}.cpp";\
		done)
endif

# Identify sources and objects
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

$(EXE): $(OBJECTS)
	$(MPICXX) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $(EXE)

clean: $(CLEAN)
	rm -f $(OBJECTS) $(EXE) src/*.cpp

# Compilation rules
$(OBJ_PATH)%.o: $(SRC_PATH)%$(FILETYPE)
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c $< -o $(SRC_PATH)$(notdir $@)
	
