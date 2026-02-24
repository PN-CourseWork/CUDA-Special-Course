TARGET= reduction
OBJS	= 

OPT	= -g -O3
ISA	= 
PARA	= -fopenmp

CXX	= nvc++ -cuda -mp=gpu -Minfo --diag_suppress integer_sign_change,cuda_compile
CXXFLAGS= $(OPT) $(ISA) $(PARA)

NVCC	= nvcc
CUFLAGS= --compiler-options "$(OPT) $(PARA)" -Xptxas=-v -arch=sm_90 -lineinfo -diag-suppress 68

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

.SUFFIXES: .cu .c
.cu.o:
	$(NVCC) -o $*.o -c $*.cu $(CUFLAGS) $(SOFLAGS) $(INCLUDES)

.cpp.o:
	$(CXX) -o $*.o -c $*.cpp $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@rm -f $(TARGET) cpu omp_offload cuda

# Individual targets
cpu: cpu.o
	$(CXX) $(CXXFLAGS) -o cpu cpu.o

offload: offload.o
	$(CXX) $(CXXFLAGS) -o omp_offload omp_offload.o

cuda: cuda.o
	$(NVCC) $(CUFLAGS) -o cuda cuda.o
