CFLAGS = -std=c99 -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math -DGRAPH_GENERATOR_OMP # -g -pg
# CFLAGS = -g -Wall -Drestrict= -DGRAPH_GENERATOR_MPI
LDFLAGS = -fopenmp -g # -g -pg
#NWARPS=1
CUDAFLAGS = -arch=$(CUDA_ARCH) -Xcompiler -fopenmp -O3 -Drestrict=__restrict__ -rdc=true #-DNWARPS=$(NWARPS)

CUDA_ARCH=sm_35
CUDA_LINK= -Xcompiler -fopenmp -arch=$(CUDA_ARCH)

all: merrill 

GENERATOR_SOURCES = ../generator/graph_generator.o ../generator/make_graph.o \
		../generator/splittable_mrg.o ../generator/utils.o
GRAPH500_SOURCES = ../graph500.o ../options.o ../rmat.o ../kronecker.o ../verify.o \
		../prng.o ../xalloc.o ../timer.o

base: $(GENERATOR_SOURCES) $(GRAPH500_SOURCES) bfs_base.o 
	nvcc -o $@ $^ $(CUDA_LINK)

merrill: $(GENERATOR_SOURCES) $(GRAPH500_SOURCES) bfs_merrill.o 
	nvcc -o $@ $^ $(CUDA_LINK)

%.o: %.cu 
	nvcc $(CUDAFLAGS) -c -o $@ $<

%.o: %.c 
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	-rm -f base merrill *.o *.a ../*.o ../generator/*.o
