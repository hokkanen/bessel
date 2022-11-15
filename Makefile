default: build
	echo "Start Build"

# Accelerator architecture
ifeq ($(CUDA),1)

CXX = nvcc
CXXDEFS = -DHAVE_CUDA
CXXFLAGS = -g -O3 --x=cu --extended-lambda -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80
FILETYPE = .c
EXE = bessel

else ifeq ($(HIP),CUDA)

CXX = hipcc
CXXDEFS = -DHAVE_HIP
CXXFLAGS = -g -O3 --x=cu --extended-lambda -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -I$(shell pwd)/hiprand
FILETYPE = .cpp
EXE = bessel

else ifeq ($(HIP),ROCM)

CXX = hipcc
CXXDEFS = -DHAVE_HIP 
CXXFLAGS = -g -O3 --offload-arch=gfx90a -I/opt/rocm/hiprand/include/ -I/opt/rocm/rocrand/include/
FILETYPE = .cpp
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
ifeq ($(MPI),1)

# On Puhti
#MPICXX = mpicxx
#MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(CXX) -DHAVE_MPI $(CXXDEFS) $(CXXFLAGS)'
#LDFLAGS += -L/appl/spack/install-tree/gcc-9.1.0/openmpi-4.1.1-vonyow/lib
#LIBS += -lmpi -lm

# On Lumi
MPICXX = $(CXX)
MPICXXFLAGS = $(CXXDEFS) -DHAVE_MPI $(CXXFLAGS) -I${MPICH_DIR}/include
LDFLAGS += -L${MPICH_DIR}/lib -L${CRAY_MPICH_ROOTDIR}/gtl/lib
LIBS += -lmpi -lmpi_gtl_hsa -lm

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
SOURCES = $(shell ls src/*$(FILETYPE))

OBJ_PATH = src/
OBJECTS = $(shell for file in $(SOURCES);\
		do echo -n $$file | sed -e "s/\(.*\)\$(FILETYPE)/\1\.o/";echo -n " ";\
		done)

build: $(EXE)

depend:
	makedepend $(CXXDEFS) -m $(SOURCES)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $(EXE)

clean: $(CLEAN)
	rm -f $(OBJECTS) $(EXE) src/*.cpp

# Compilation rules
$(OBJ_PATH)%.o: $(SRC_PATH)%$(FILETYPE)
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c $< -o $(SRC_PATH)$(notdir $@)
