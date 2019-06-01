#include "../compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <thrust/sort.h>
#include <thrust/version.h>
#include <thrust/unique.h>


#include <cuda.h>

extern "C"{
#include "../graph500.h"
}

#include "../xalloc.h"
#include "../generator/graph_generator.h"

// Set by Makefile 
//#define NWARPS 1

//#define CHUNK_ELEM 4096
//#define CHUNK_SIZE (sizeof(int32_t)*CHUNK_ELEM) // Size of a chunk in byte

#define BITMAP_TYPE uint32_t
#define BITMAP_WORD 32

/* Global variables */
static int64_t maxvtx; // total number of vertices 
static int64_t nv; // number of vertices 
static int64_t maxedg;
static int32_t nwords; 

/* Host pointers */
static int32_t * h_CSR_R;
static int32_t * h_CSR_C;
static BITMAP_TYPE * h_in_queue;
static BITMAP_TYPE * h_out_queue;
static int32_t * h_predecessors;
static int32_t h_n_in_queue;
static int32_t h_n_out_queue;

/* Device pointers */
static int32_t * d_CSR_R;
static int32_t * d_CSR_C;
static BITMAP_TYPE * d_in_queue;
static BITMAP_TYPE * d_out_queue;
static int32_t * d_predecessors; 
static int32_t * d_n_in_queue; 
static int32_t * d_n_out_queue;
__constant__ int32_t d_nwords;

static cudaEvent_t start, stop;

static void HandleError(cudaError_t err, 
			const char *file,
			int line)
{
	if(err != cudaSuccess)
	{
		printf("%s in %s at line %d \n",cudaGetErrorString(err),file,line);
		exit(EXIT_FAILURE); 
	}
}

#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))

/* "Private functions" */
/* Compute the total number of vertices in the generated graph */
static void find_nv(const struct packed_edge * restrict IJ, const int64_t nedge)
{
	maxvtx = -1;
	// Here use the 40 cores to compute   
	#pragma omp parallel 
	{
		int64_t k;
		#pragma omp for reduction(max:maxvtx)
		for(k = 0 ; k < nedge ; ++k)
		{
			if(get_v0_from_edge(&IJ[k]) > maxvtx)
				maxvtx = get_v0_from_edge(&IJ[k]);
			if(get_v1_from_edge(&IJ[k]) > maxvtx)
				maxvtx = get_v1_from_edge(&IJ[k]);
		}
	} 
	nv = maxvtx+1;
}

static void 
omp_prefix_sum(int32_t * x, int N)
{
	int32_t * suma; 
	#pragma omp parallel 
	{
		const int ithread = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();
		#pragma omp single
		{	
			suma = (int32_t*)malloc(sizeof(int32_t)*nthreads+1);
			suma[0] = 0;
		}
		int32_t sum = 0;
		#pragma omp for schedule(static)
		for(unsigned int i = 0 ; i < N ; ++i)
		{
			sum += x[i];
			x[i] = sum; 
		}
		suma[ithread+1] = sum; 
		#pragma omp barrier
		float offset = 0;
		for(unsigned int i = 0 ; i < (ithread+1) ; ++i)
			offset += suma[i];
		#pragma omp for schedule(static)
		for(unsigned int i = 0 ; i < N ; ++i)
			x[i] += offset;
	}
	
	for(unsigned int i = N ; i > 0 ; --i)
		x[i] = x[i-1];
	x[0] = 0;

	free(suma);

}

static void 
edgelist_to_CSR(const struct packed_edge * restrict IJ, const int64_t nedge)
{
	//int32_t *h_chunk_v0, *h_chunk_v1; 
	//int32_t *d_chunk_v0, *d_chunk_v1;
 
	//int nchunk = (2*nedge*sizeof(int32_t))/CHUNK_SIZE;

	printf("MAXVTX(%" PRId64 ")\n",maxvtx);
	printf("NV(%" PRId64 ")\n",nv);

	cudaSetDevice(0);
	/* Init CSR arrays on GPU */
	HANDLE_ERROR(cudaMalloc((void**)&d_CSR_R,sizeof(int32_t)*(nv+1)));
	h_CSR_R = (int32_t*)malloc(sizeof(int32_t)*(nv+1));
	assert(h_CSR_R); 
	memset(h_CSR_R,0,sizeof(int32_t)*(nv+1));

	/* Step one, count the CSR_R and CSR_C size */
	maxedg = 0;

	#pragma omp parallel for reduction(+:maxedg)
	for(unsigned int i = 0 ; i < nedge ; ++i)
	{
		// No self loop 
		if(get_v0_from_edge(&IJ[i]) != get_v1_from_edge(&IJ[i]))
		{
			__sync_fetch_and_add(&h_CSR_R[get_v0_from_edge(&IJ[i])],1);
			__sync_fetch_and_add(&h_CSR_R[get_v1_from_edge(&IJ[i])],1);
			maxedg+=2;
		}
	}
	
	printf("MAXEDG(%" PRId64 ")\n",maxedg);

	int32_t tot = 0;
	for(unsigned int i = 0 ; i < nv+1 ; ++i)
		tot += h_CSR_R[i];
	printf("tot(%d)\n",tot);
	
	// Malloc CRC array 
	h_CSR_C = (int32_t*)malloc(sizeof(int32_t)*maxedg);
	assert(h_CSR_C);
	HANDLE_ERROR(cudaMalloc((void**)&d_CSR_C,sizeof(int32_t)*maxedg));

	//omp_prefix_sum(h_CSR_R,nv);
	int32_t tmp = h_CSR_R[0];
	for(unsigned int i = 1 ; i < nv+1 ; ++i)
	{
		int32_t tmp2 = h_CSR_R[i];
		h_CSR_R[i] = tmp;
		tmp += tmp2;
	}
	h_CSR_R[0] = 0;

	printf("last(%d)\n",h_CSR_R[nv]);
	assert(h_CSR_R[nv] == maxedg);
	//printf("\nCSR_R list");
	//for(unsigned int i = 0 ; i < nv-1 ; ++i)
	//	printf(" %d(%d)",h_CSR_R[i],h_CSR_R[i+1] - h_CSR_R[i]);
	//printf("\n");

	int32_t * CSR_R_counter = (int32_t*)malloc(sizeof(int32_t)*nv); 
	assert(CSR_R_counter);
	memset(CSR_R_counter,0,sizeof(int32_t)*nv);

	//printf("CSR_C generiation\n");

	/* Step two generate CSC array */
	#pragma omp parallel for 
	for(unsigned int i = 0 ; i < nedge ; ++i)

	{
		int32_t v0 = (int32_t)get_v0_from_edge(&IJ[i]);
		int32_t v1 = (int32_t)get_v1_from_edge(&IJ[i]);
		if(v0 != v1)
		{
			int counter_v0 = __sync_fetch_and_add(&(CSR_R_counter[v0]),1);
			int counter_v1 = __sync_fetch_and_add(&(CSR_R_counter[v1]),1);
			//printf("Edge(%d,%d) added in %d(%d) and %d(%d)\n",v0,v1,v0,counter_v0,v1,counter_v1);
			h_CSR_C[h_CSR_R[v0]+counter_v0] = v1;	
			h_CSR_C[h_CSR_R[v1]+counter_v1] = v0;
		 }
	}
	free(CSR_R_counter);
	//printf("ENDED \n");

	// Unique the vertices for one vertex
	// With or without sorting dirst
	//for(unsigned int i = 0 ; i < nv ; ++i)
	//	thrust::sort(&(h_CSR_C[h_CSR_R[i]]),&(h_CSR_C[h_CSR_R[i+1]]));	

	int32_t * previous = &(h_CSR_C[h_CSR_R[0]]);
	for(unsigned int i = 0  ; i < nv ; ++i)
	{
		int32_t * new_last = thrust::unique(&(h_CSR_C[h_CSR_R[i]]),&(h_CSR_C[h_CSR_R[i+1]]));
		int32_t nElem = (new_last - &(h_CSR_C[h_CSR_R[i]]));
		//printf("%d Old(%d) New(%d) previous(%p) first(%p) last(%p) new_last(%p)\n",i,h_CSR_R[i+1] - h_CSR_R[i],nElem,previous,&(h_CSR_C[h_CSR_R[i]]),&(h_CSR_C[h_CSR_R[i+1]]),new_last);
		// Move part 
		memmove(&(h_CSR_C[i]),previous,h_CSR_R[i]*sizeof(int32_t));
		previous = new_last;
		h_CSR_R[i] = nElem;


	}

	// New prefix sum 
	tmp = h_CSR_R[0];
	for(unsigned int i = 1 ; i < nv+1 ; ++i)
	{
		int32_t tmp2 = h_CSR_R[i];
		h_CSR_R[i] = tmp;
		tmp += tmp2;
	}
	h_CSR_R[0] = 0;

	printf("NEWEDGES(%d)\n",h_CSR_R[nv]);

	//printf("\nMalloc\n");
	// Copy CSR and CSC on GPU 
	HANDLE_ERROR(cudaMemcpy(d_CSR_R,h_CSR_R,sizeof(int32_t)*(nv+1),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_CSR_C,h_CSR_C,sizeof(int32_t)*maxedg,cudaMemcpyHostToDevice));

	// Prepare in and ou queues as bitmap 
	nwords = (nv + (BITMAP_WORD / 2)) / BITMAP_WORD; 
	HANDLE_ERROR(cudaMemcpyToSymbol(d_nwords,&nwords,sizeof(int32_t)));
	HANDLE_ERROR(cudaMalloc((void**)&d_in_queue,sizeof(BITMAP_TYPE)*nwords));
	h_in_queue = (BITMAP_TYPE*)malloc(sizeof(BITMAP_TYPE)*nwords);
	assert(h_in_queue);
	cudaMalloc((void**)&d_out_queue,sizeof(BITMAP_TYPE)*nwords);
	h_out_queue = (BITMAP_TYPE*)malloc(sizeof(BITMAP_TYPE)*nwords);
	assert(h_out_queue);
	cudaMalloc((void**)&d_n_in_queue,sizeof(int32_t));
	cudaMalloc((void**)&d_n_out_queue,sizeof(int32_t));	

	cudaMalloc((void**)&d_predecessors,sizeof(int32_t)*nv);
	h_predecessors = (int32_t*)malloc(sizeof(int32_t)*nv);
	assert(h_predecessors);

	cudaEventCreate(&start);
	cudaEventCreate(&stop); 

	printf("End make_CSR\n");

}

__inline__ __device__ int warpReduceSum(int val)
{
	for(int offset = warpSize/2 ; offset > 0 ; offset /= 2)
		val += __shfl_down(val,offset);
	return val;
} 

__global__ void explore_CSC(	int32_t * CSC_C, int32_t * CSC_R,
				BITMAP_TYPE * in_queue, int32_t * n_in_queue, 
				BITMAP_TYPE * out_queue, int32_t * n_out_queue, 
				int32_t * predecessors)
{
	int lane_id = threadIdx.x & 31; 
	int warp_id = threadIdx.x >> 5; 

	int vtx = threadIdx.x+blockIdx.x*blockDim.x;
	int32_t value_vtx = predecessors[vtx]; 
	int active = 0; 
	if(value_vtx == -1)
		active = 1;

	// No more active thread in this warp 
	if(!__any(active))
		return;

	int32_t range[3] = {0,0,0};

	if(active)
	{
		range[0] = CSC_C[vtx];
		range[1] = CSC_C[vtx+1]; 
		range[2] = range[1] - range[0];
	}

	volatile __shared__ int comm[NWARPS][3]; 
	volatile __shared__ int shared_vtx[NWARPS]; 
	volatile __shared__ int sum[NWARPS]; 
	volatile __shared__ int end[NWARPS]; 

	if(lane_id == 0)
		sum[warp_id] = 0;

	while(__any(range[2]))
	{
		int neighbor = -1; 

		if(range[2])
			comm[warp_id][0] = lane_id;
		if(comm[warp_id][0] == lane_id)
		{
			comm[warp_id][1] = range[0]; 
			comm[warp_id][2] = range[1]; 
			range[2] = 0; 
			shared_vtx[warp_id] = vtx; 
		}
		
		int r_gather = comm[warp_id][1] + lane_id;
		int r_gather_end = comm[warp_id][2]; 

		if(lane_id == 0)
			end[warp_id] = 0;

		while((r_gather < r_gather_end) && !end[warp_id])
		{
			neighbor = CSC_R[r_gather];
		
			BITMAP_TYPE value_in_queue = in_queue[neighbor/BITMAP_WORD];
			if(value_in_queue & (1<<(neighbor%BITMAP_WORD)))
			{
				// Can be changed because just this warp access to this vertex
				int prev = atomicCAS(&(predecessors[shared_vtx[warp_id]]),-1,neighbor);
				if(prev == -1)
				{
					int value_out_queue = 1 << (shared_vtx[warp_id]%BITMAP_WORD);
					atomicOr(&(out_queue[shared_vtx[warp_id]/BITMAP_WORD]),value_out_queue);
					++sum[warp_id];
					end[warp_id] = 1;
				}
			}			
			r_gather += 32; 
		}
	}
	if(lane_id == 0 && sum[warp_id])
		atomicAdd(n_out_queue,sum[warp_id]);
}

/* Global functions required by graph500.c */
/* declared extern "C" for CUDA to C compatibility */
__global__ void explore_CSR(	int32_t * CSR_R, int32_t * CSR_C, 
				BITMAP_TYPE * in_queue, int32_t * n_in_queue,
				BITMAP_TYPE * out_queue, int32_t * n_out_queue,
				int32_t * predecessors)
{
	int lane_id = threadIdx.x & 31; 
	int warp_id = threadIdx.x >> 5;
	int word = blockIdx.x*NWARPS+warp_id;
	BITMAP_TYPE val_in_queue = in_queue[word]; 

	if(val_in_queue == 0)
		return; 

	uint32_t range[3] = {0,0,0};
	int32_t id_vertex = -1;

	if(val_in_queue & 1 << lane_id)
	{
		id_vertex = word*BITMAP_WORD+lane_id;
		range[0] = CSR_R[id_vertex];
		range[1] = CSR_R[id_vertex+1];
		range[2] = range[1] - range[0]; 
	}

	//if(blockIdx.x == 20)
		//printf("val_in_queue(%u) thread(%d) lane_id(%d) warp_id(%d) id_vertex(%d) range(%d,%d,%d)\n",val_in_queue,threadIdx.x,lane_id,warp_id,id_vertex,range[0],range[1],range[2]);

	volatile __shared__ int comm[NWARPS][3];
	volatile __shared__ int shared_vertex[NWARPS];
	uint32_t sum; 

	while(__any(range[2]))
	{
		int32_t neighbor = -1; 
	
		if(range[2])
			comm[warp_id][0] = lane_id; 
	
		if(comm[warp_id][0] == lane_id)
		{
			comm[warp_id][1] = range[0];
			comm[warp_id][2] = range[1]; 
			range[2] = 0; 
			shared_vertex[warp_id] = id_vertex; 
		}

		//if(blockIdx.x == 20)
			//printf("thread(%d) lane_id(%d) warp_id(%d) id_vertex(%d) shared_vertex(%d)(%d,%d)\n",threadIdx.x,lane_id,warp_id,id_vertex,shared_vertex,comm[warp_id][0],comm[warp_id][1]);


		int r_gather = comm[warp_id][1] + lane_id; 
		int r_gather_end = comm[warp_id][2]; 
		while(r_gather < r_gather_end)	
		{
			sum = 0; 
			neighbor = CSR_C[r_gather];
 
			int32_t prev = atomicCAS(&(predecessors[neighbor]),-1,shared_vertex[warp_id]);
			if(prev == -1){
				uint32_t val_out_queue = 1 << neighbor%BITMAP_WORD;
				atomicOr(&(out_queue[neighbor/BITMAP_WORD]),val_out_queue);
				sum = 1;
			} 

			// Check visited 
			if(__any(sum))
			{
				sum = warpReduceSum(sum); 
				if(lane_id == 0)
					atomicAdd(n_out_queue,sum); 
			}		
			r_gather += 32; 
		}
	}
}

__global__ void setup_GPU(int32_t * predecessors, int64_t srcvtx)
{
	//int id = threadIdx.x + blockIdx.x*blockDim.x;
	//predecessors[id] = -1;
	//if(id == srcvtx)
	predecessors[srcvtx] = (int32_t)srcvtx;
}

/* Create the graph structure on the GPUs */
extern "C"
int create_graph_from_edgelist(struct packed_edge * IJ, int64_t nedge)
{
	printf("create_graph_from_edgelist nedge(%" PRId64 ")\n",nedge);
	#pragma omp parallel 
	#pragma omp single 
	printf("%d threads\n", omp_get_num_threads());
	/* Each thread handle a GPU */		
	find_nv(IJ,nedge);

	/* Compute CSR representation */
	edgelist_to_CSR(IJ,nedge);

	return 0; 
}

extern "C"
int make_bfs_tree( int64_t *bfs_tree_out, int64_t *max_vtx_out, int64_t srcvtx)
{
	// TODO check this nv != maxvtx
	*max_vtx_out = maxvtx;
	memset(h_in_queue,0,sizeof(BITMAP_TYPE)*nwords);
	h_in_queue[srcvtx >> 5] = 1 << (srcvtx & 31);
	h_n_in_queue = 1; 
	printf("\nsrcvtx(%" PRId64 ") in_queue(%u) n_in_queue(%d)\n",srcvtx,h_in_queue[0],h_n_in_queue);
	HANDLE_ERROR(cudaMemcpy(d_n_in_queue,&h_n_in_queue,sizeof(int32_t),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_in_queue,h_in_queue,sizeof(BITMAP_TYPE)*nwords,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(d_predecessors,-1,sizeof(int32_t)*nv));
	setup_GPU<<<1,1>>>(d_predecessors,srcvtx);
	//setup_GPU<<<(nv + (NWARPS/2))/NWARPS,NWARPS>>>(d_predecessors,srcvtx);
	
	int32_t iteration = 0;

	while(1)
	{
		if(iteration++ > 1 << 20)
		{
			fprintf(stderr,"Too many iterations(%d)\n",iteration);
			return -1;
		}
		printf("iteration(%2d) n_in_queue(%10d) nblocks(%d) nthreads(%d) ",
			iteration,
			h_n_in_queue,
			(nwords+(NWARPS/2))/NWARPS,
			32*NWARPS);
		fflush(stdout);

		HANDLE_ERROR(cudaMemset(d_n_out_queue,0,sizeof(int32_t)));
		HANDLE_ERROR(cudaMemset(d_out_queue,0,sizeof(BITMAP_TYPE)*nwords));
		
		cudaEventRecord(start);
		if(iteration < 3)
		{
			printf(" CSR ");
			explore_CSR<<< (nwords+(NWARPS/2))/NWARPS ,32*NWARPS >>>(	
						d_CSR_R,d_CSR_C,d_in_queue,d_n_in_queue,
						d_out_queue,d_n_out_queue,d_predecessors);
		}else{
			printf(" CSC ");
			explore_CSC<<< (nwords + (NWARPS/2))/NWARPS, 32*NWARPS >>>(
						d_CSR_R,d_CSR_C,d_in_queue,d_n_in_queue,
						d_out_queue,d_n_out_queue,d_predecessors);
		}
		cudaEventRecord(stop);
		HANDLE_ERROR(cudaMemcpy(&h_n_out_queue,d_n_out_queue,sizeof(int32_t),cudaMemcpyDeviceToHost));
		cudaEventSynchronize(stop);
		float milliseconds = 0; 
		cudaEventElapsedTime(&milliseconds,start,stop);
		printf("out_queue(%10d) time(%.4f)s \n",h_n_out_queue,milliseconds/1000);	
		if(h_n_out_queue == 0)
		{
			printf("BFS ended\n");
			break;
		}	
		/* Switch queues */
		HANDLE_ERROR(cudaMemcpy(d_in_queue,d_out_queue,sizeof(BITMAP_TYPE)*nwords,cudaMemcpyDeviceToDevice));
		/* For checking */
		HANDLE_ERROR(cudaMemcpy(h_in_queue,d_out_queue,sizeof(BITMAP_TYPE)*nwords,cudaMemcpyDeviceToHost));
		h_n_in_queue = h_n_out_queue;
	}

	HANDLE_ERROR(cudaMemcpy(h_predecessors,d_predecessors,sizeof(int32_t)*nv,cudaMemcpyDeviceToHost));
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < nv ; ++i)
	{
		bfs_tree_out[i] = (int64_t)h_predecessors[i];		
		assert(bfs_tree_out[i] < nv);
		assert(bfs_tree_out[i] > -2);
	}

	return 0;
}

extern "C"
void destroy_graph()
{
	free(h_CSR_R);
	free(h_CSR_C);
	cudaFree(d_CSR_R);
	cudaFree(d_CSR_C);
	free(h_in_queue);
	free(h_out_queue);
	cudaFree(d_in_queue);
	cudaFree(d_out_queue);
}
