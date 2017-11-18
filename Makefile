CC		= nvcc 
EXEC	= out

SOURCES = \
cuda_mmult_kernels.cu \
cuda_mmult.cu

OBJS = $(SOURCES:.cu=.o)

%.o: %.cu
	$(CC) -c -O3 -o $@ $<

all: $(OBJS)
	$(CC) -link -L/usr/local/cuda/lib64/ -O3 $(OBJS) -o $(EXEC)

clean:
	@rm -f $(OBJS) $(EXEC)
