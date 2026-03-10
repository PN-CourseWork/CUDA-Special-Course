NVCC	= nvcc
CUFLAGS	= --compiler-options "-g -O3" -Xptxas=-v -arch=sm_90 -lineinfo -diag-suppress 68

all: ex1 ex2 compare

ex1: ex1.cu
	$(NVCC) $(CUFLAGS) -o ex1 ex1.cu

ex2: ex2.cu
	$(NVCC) $(CUFLAGS) -o ex2 ex2.cu

compare: compare.cu
	$(NVCC) $(CUFLAGS) -o compare compare.cu

clean:
	rm -f ex1 ex2 compare
