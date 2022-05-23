CC = nvcc
CCFLAGS = -D=HAVE_CUDA=1 --x cu --extended-lambda -gencode=arch=compute_70,code=sm_70
MPICXX = mpicxx
MPICXXFLAGS = -g -O2
LD = $(CC)
INCLUDES =
SRC = bessel.cpp
OBJS = $(SRC:.cpp=.cpp.o)
EXE = bessel

# Mahti
MPICXXENV = OMPI_CXXFLAGS='' OMPI_CXX='$(CC) $(CCFLAGS)'
LDFLAGS = -L/appl/spack/install-tree/gcc-9.1.0/openmpi-4.1.1-vonyow/lib
LIBS = -lmpi

.SUFFIXES:

all: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBS)

bessel.cpp.o: bessel.cpp
	$(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) $(INCLUDES) -c -o $@ $<

.PHONY: clean

clean:
	rm -f $(OBJS) *~ $(EXE) *.o *.out *.err