default: build
	echo "Start Build"

# Accelerator architecture
ifeq ($(ACC),CUDA)

# Use nvc++ instead of nvc because of curand_kernel.h
CXX = nvc++
CXXFLAGS = -g -O3 -acc -gpu=cc80 -Minfo=accel
LDFLAGS += -acc -gpu=cc80
FILETYPE = .c
EXE = bessel

else ifeq ($(CUDA),1)

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
CXXFLAGS = -g -O3 -x hip --offload-arch=gfx90a -I/opt/rocm/hiprand/include/ -I/opt/rocm/rocrand/include/
FILETYPE = .cpp
EXE = bessel

else ifeq ($(OMP),CUDA)

# Use nvc++ instead of nvc because of curand_kernel.h
CXX = nvc++
CXXFLAGS = -g -O3 -mp=gpu -gpu=cc80
# Huge performance loss without linker flags
LDFLAGS += -mp=gpu -gpu=cc80
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
LDFLAGS += -L$(shell pwd)/umpire/lib/ -L$(shell pwd)/umpire/lib64/
LIBS += -lcamp -lumpire

endif

# Message passing protocol
ifeq ($(MPI),OMPI)

# On Mahti
MPICXX = mpicxx
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(CXX) -DHAVE_MPI $(CXXDEFS) $(CXXFLAGS)'
LDFLAGS += -L${CUDA_PATH}/lib64
LIBS += -lm -lcudart

else ifeq ($(MPI),CRAY)

# On Lumi
ifeq ($(CXX),gcc)
CXX = CC
endif
MPICXX = CC
MPICXXFLAGS = $(CXXDEFS) -DHAVE_MPI -g -O3
LDFLAGS += -L${ROCM_PATH}/lib
LIBS += -lm -lamdhip64

else

MPICXX = $(CXX)
MPICXXFLAGS = $(CXXDEFS) $(CXXFLAGS)
LIBS += -lm

endif

# Create temporary .cpp files if needed (for HIP only)
ifeq ($(FILETYPE),.cpp)
$(shell for file in `ls src/*.c`;\
		do cp -- "$$file" "$${file%.c}.cpp";\
		done)
endif

# Identify sources and objects
SRC_PATH = src/
SOURCES = $(wildcard $(SRC_PATH)*$(FILETYPE))
HEADERS = $(wildcard $(SRC_PATH)*.h) $(wildcard $(SRC_PATH)arch/*.h)

OBJ_PATH = $(SRC_PATH)
OBJECTS = $(SOURCES:$(SRC_PATH)%$(FILETYPE)=$(OBJ_PATH)%.o)


build: $(EXE)

depend:
	makedepend $(CXXDEFS) -m $(SOURCES)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJECTS)
	$(MPICXX) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $(EXE)

clean: $(CLEAN)
	rm -f $(OBJECTS) $(EXE) src/*.cpp

# Rule for compiling comms.cpp with $(MPICXX)
$(OBJ_PATH)comms.o: $(SRC_PATH)comms$(FILETYPE) $(HEADERS)
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c $< -o $@

# Rule for compiling other bessel.cpp with $(CXX)
$(OBJ_PATH)bessel.o: $(SRC_PATH)bessel$(FILETYPE) $(HEADERS)
	$(CXX) $(CXXDEFS) $(CXXFLAGS) -c $< -o $@
	
