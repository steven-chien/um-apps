#include "../compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

extern "C"{
#include "../graph500.h"
}

#include "../xalloc.h"
#include "../generator/graph_generator.h"

#ifndef NWARPS
// Set by Makefile 
#define NWARPS 8
#endif

#ifndef NGPU
// Set by Makefile 
#define NGPU 4
#endif

//#define CHUNK_ELEM 4096
//#define CHUNK_SIZE (sizeof(int32_t)*CHUNK_ELEM) // Size of a chunk in byte

#define BITMAP_TYPE uint32_t
#define BITMAP_WORD 32

/* Global variables */
static int64_t maxvtx; // total number of vertices 
static int64_t nv; // number of vertices 
static int64_t maxedg;
static int32_t nwords; 
static int32_t edg_per_gpu; // number of edges per GPU 
static int32_t edg_last_gpu; // Number of edges for the last GPU 

/* Host pointers */
static int32_t * h_CSR_R;
static int32_t * h_CSR_C;
static int32_t * h_predecessors;
static int32_t h_n_in_queue;
static int32_t h_n_out_queue;

/* Device pointers */
static int32_t * d_CSR_R[NGPU];
static int32_t * d_CSR_C[NGPU];
static BITMAP_TYPE * d_in_queue[NGPU];
static BITMAP_TYPE * d_out_queue[NGPU];
static int32_t * d_predecessors[NGPU]; 
static int32_t * d_n_in_queue[NGPU]; 
static int32_t * d_n_out_queue[NGPU];
static BITMAP_TYPE * d_visited_tex[NGPU];
__constant__ int32_t d_nwords;
static int32_t * d_tmp_predecessors; 

texture<BITMAP_TYPE, 1, cudaReadModeElementType> tex_visited; 
texture<BITMAP_TYPE, 1, cudaReadModeElementType> tex_in_queue; 

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

void 
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

	// Check GPU number 
	int nDevices; 
	HANDLE_ERROR(cudaGetDeviceCount(&nDevices)); 
	if(nDevices < NGPU)
	{
		fprintf(stderr,"Error, less GPU (%d) than requested (%d)\n ",nDevices,NGPU); 
		exit(EXIT_FAILURE); 
	}

	//printf("MAXVTX(%" PRId64 ")\n",maxvtx);
	//printf("NV(%" PRId64 ")\n",nv);

	cudaSetDevice(0);
	/* Init CSR arrays on GPU */
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
	
	//printf("MAXEDG(%" PRId64 ")\n",maxedg);

	int32_t tot = 0;
	for(unsigned int i = 0 ; i < nv+1 ; ++i)
		tot += h_CSR_R[i];
	printf("tot(%d)\n",tot);
	
	// Malloc CRC array 
	h_CSR_C = (int32_t*)malloc(sizeof(int32_t)*maxedg);
	assert(h_CSR_C);

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

	// Divide the list between the GPUs
	edg_per_gpu = maxedg / NGPU; 
	edg_last_gpu = maxedg - edg_per_gpu * (NGPU-1);

	//printf("edges(%lld) edg_per_gpu(%d) edg_last_gpu(%d) total(%d)\n",maxedg,edg_per_gpu,edg_last_gpu,edg_per_gpu*(NGPU-1)+edg_last_gpu);	

	nwords = (nv + (BITMAP_WORD / 2)) / BITMAP_WORD; 

#pragma omp parallel
{
	int32_t * tmpCSR_R = (int32_t*)malloc(sizeof(int32_t)*(nv+1));

#pragma omp for
	for(unsigned int i = 0 ; i < NGPU ; ++i)
	{
		//printf("\nGPU(%d)\n",i);
		HANDLE_ERROR(cudaSetDevice(i));
		for(unsigned int j = 0 ; j < nv+1 ; ++j)
		{
			tmpCSR_R[j] = h_CSR_R[j] - i*edg_per_gpu;
			if(tmpCSR_R[j] < 0) tmpCSR_R[j] = 0; 
			// Last GPU case
			if(i == NGPU-1){
				if(tmpCSR_R[j] > edg_last_gpu) tmpCSR_R[j] = edg_last_gpu; 
			}else{
				if(tmpCSR_R[j] > edg_per_gpu) tmpCSR_R[j] = edg_per_gpu; 
			}
			//printf(" %2d ",tmpCSR_R[j]);
		}
	
		int p2p_enable; 
		//printf("\n");
		for(unsigned int j = 0 ; j < NGPU ; ++j)
		{
			if(i!=j)
			{
				HANDLE_ERROR(cudaDeviceCanAccessPeer(&p2p_enable,i,j)); 
				printf("Connecting GPU(%d) <-> GPU(%d) enable(%d)\n",i,j,p2p_enable);
				if(p2p_enable)
					HANDLE_ERROR(cudaDeviceEnablePeerAccess(j,0)); 
			}
		}		

		// Same size for the CSR array 
		HANDLE_ERROR(cudaMalloc((void**)&d_CSR_R[i],sizeof(int32_t)*(nv+1)));
		HANDLE_ERROR(cudaMemcpy(d_CSR_R[i],tmpCSR_R,sizeof(int32_t)*(nv+1),cudaMemcpyHostToDevice));
		if(i!=NGPU-1){
			HANDLE_ERROR(cudaMalloc((void**)&d_CSR_C[i],sizeof(int32_t)*edg_per_gpu));
			HANDLE_ERROR(cudaMemcpy(d_CSR_C[i],&h_CSR_C[edg_per_gpu*i],sizeof(int32_t)*edg_per_gpu,cudaMemcpyHostToDevice)); 

		}else{
			HANDLE_ERROR(cudaMalloc((void**)&d_CSR_C[i],sizeof(int32_t)*edg_last_gpu));
			HANDLE_ERROR(cudaMemcpy(d_CSR_C[i],&h_CSR_C[edg_per_gpu*i],sizeof(int32_t)*edg_last_gpu,cudaMemcpyHostToDevice)); 
		}
	       	HANDLE_ERROR(cudaMemcpyToSymbol(d_nwords,&nwords,sizeof(int32_t)));
		HANDLE_ERROR(cudaMalloc((void**)&d_in_queue[i],sizeof(BITMAP_TYPE)*nwords));
		HANDLE_ERROR(cudaMalloc((void**)&d_out_queue[i],sizeof(BITMAP_TYPE)*nwords));
		HANDLE_ERROR(cudaMalloc((void**)&d_n_in_queue[i],sizeof(int32_t)));
		HANDLE_ERROR(cudaMalloc((void**)&d_n_out_queue[i],sizeof(int32_t)));
		HANDLE_ERROR(cudaMalloc((void**)&d_visited_tex[i],sizeof(BITMAP_TYPE)*nwords));	
		
		if(i == 0)
			HANDLE_ERROR(cudaMalloc((void**)&d_tmp_predecessors,sizeof(int32_t)*nv)); 
		HANDLE_ERROR(cudaMalloc((void**)&d_predecessors[i],sizeof(int32_t)*nv));
	}

	free(tmpCSR_R); 
}

	h_predecessors = (int32_t*)malloc(sizeof(int32_t)*nv);
	assert(h_predecessors);

	cudaEventCreate(&start);
	cudaEventCreate(&stop); 

	printf("End make_CSR\n");

}

__inline__ __device__ int warpScanSumDown(int val)
{
	int lane_id = threadIdx.x & 31; 
	for(int offset = 1 ; offset < 32 ; offset <<= 1)
	{
		int y = __shfl_down(val,offset); 
		if(lane_id <= 31 - offset)
			val += y;
	}
	return val; 
}

__inline__ __device__ int warpScanSum(int val)
{
	int lane_id = threadIdx.x & 31; 
	for(int offset = 1 ; offset < 32 ; offset <<= 1)
	{
		int y = __shfl_up(val,offset); 	
		if(lane_id >= offset)
			val += y;
	}
	return val; 
}


__inline__ __device__ int warpReduceSum(int val)
{
	for(int offset = warpSize/2 ; offset > 0 ; offset /= 2)
		val += __shfl_down(val,offset);
	return val;
} 

__global__ void explore_frontier_CSR( BITMAP_TYPE * out_queue,  int32_t * visited_label, BITMAP_TYPE * visited_tex,  int32_t * n_out_queue, int32_t * R, int32_t * C)
{
	int lane_id = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;
	
	int ligne = threadIdx.x+blockIdx.x*blockDim.x;
	int32_t value_visited = visited_label[ligne];			//GLOBAL
	int actif = 0;
	if(value_visited == -1)
		actif = 1;
		
	if(!__any(actif))
		return;
		
	unsigned int word = ligne/BITMAP_WORD;
	unsigned int range[3] = {0,0,0};
	
	if(value_visited == -1)
	{
		range[0] = R[ligne];
		range[1] = R[ligne+1];
		range[2] = range[1] - range[0];
	}
	
	// On va explorer chaque ligne successivement 
	volatile __shared__ int comm[NWARPS][3];
	volatile __shared__ int shared_ligne[NWARPS];
	volatile __shared__ int sum[NWARPS];
	volatile __shared__ int fin[NWARPS];
	
	if(lane_id == 0)
		sum[warp_id] = 0;
	
	while( __any(range[2]) )
	{
		int voisin = -1;
	
		if(range[2])
			comm[warp_id][0] = lane_id;
	
		if(comm[warp_id][0] == lane_id)
		{
			comm[warp_id][1] = range[0];
			comm[warp_id][2] = range[1];
			range[2] = 0;
			shared_ligne[warp_id] = ligne;
		}
		
		int r_gather = comm[warp_id][1] + lane_id;
		int r_gather_end = comm[warp_id][2];
		
		if(lane_id==0)
			fin[warp_id] = 0;
		
		while(r_gather < r_gather_end && !fin[warp_id])
		{
			voisin = C[r_gather];
	
			// Vérifier voisin dans in_queue
			unsigned int position = voisin / BITMAP_WORD;
			BITMAP_TYPE mask = tex1Dfetch(tex_in_queue,position);
			BITMAP_TYPE mask_bit = 1 << (voisin % BITMAP_WORD);
			if(mask & mask_bit)
			{
				// Ajout direct du voisin dans visited et passer à la suite 
				//visited_label[shared_ligne[warp_id]] = voisin+d_offset;
				//int old = atomicCAS(&visited_label[shared_ligne[warp_id]],-1,voisin+d_offset);
				//if(old == -1)

				visited_label[shared_ligne[warp_id]] =  voisin;
				if(visited_label[shared_ligne[warp_id]] == voisin)
				{
					visited_tex[word] |= 1 << shared_ligne[warp_id]%BITMAP_WORD;
					out_queue[word] |= 1 << shared_ligne[warp_id]%BITMAP_WORD;
					++sum[warp_id];
					fin[warp_id] = 1;
				}
			}
			r_gather+=32;
		}
	}
	
	if(lane_id == 0 && sum[warp_id])
		atomicAdd(n_out_queue,sum[warp_id]);
}

//__launch_bounds__(NWARPS*32, MIN_BLOCKS_PER_SMX)
__global__ void explore_frontier_CSC( restrict BITMAP_TYPE * in_queue, restrict BITMAP_TYPE * out_queue,  int32_t * visited_label, BITMAP_TYPE * visited_tex , int32_t * n_out_queue, int32_t * R, int32_t * C)
{
	int lane_id = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5; 

	int word = blockIdx.x*NWARPS+warp_id;
	int val_in_queue = in_queue[word];								// GLOBAL
	if(val_in_queue == 0)
		return;

	int id_sommet = -1;
	unsigned int range[3] = {0,0,0};
	
	if(val_in_queue & 1 << lane_id)
	{
		id_sommet = word*32+lane_id;
		range[0] = C[id_sommet];									//GLOBAL
		range[1] = C[id_sommet+1];								//GLOBAL
		range[2] = range[1] - range[0];
	}

	volatile __shared__ int comm[NWARPS][3];
	volatile __shared__ int shared_sommet[NWARPS];
	uint32_t sum;

	while( __any(range[2]) )
	{

		int voisin = -1;

		if(range[2])
			comm[warp_id][0] = lane_id;							// SHARED

		if(comm[warp_id][0] == lane_id)
		{
			comm[warp_id][1] = range[0];							// SHARED
			comm[warp_id][2] = range[1];							// SHARED
			range[2] = 0;
			shared_sommet[warp_id] = id_sommet;					// SHARED
		}

		int r_gather = comm[warp_id][1] + lane_id;
		int r_gather_end = comm[warp_id][2];
		while(r_gather < r_gather_end)
		{
			sum = 0;
			voisin = R[r_gather];								// GLOBAL

			unsigned int position = voisin / BITMAP_WORD;
			BITMAP_TYPE mask = tex1Dfetch(tex_visited,position);
			BITMAP_TYPE mask_bit = 1 << (voisin % BITMAP_WORD);
			if(!(mask & mask_bit))
			{
				visited_tex[position] |= mask_bit;
				//int32_t value = atomicCAS(&visited_label[voisin],-1,shared_sommet[warp_id]+d_offset);
				if(visited_label[voisin] == -1)
					visited_label[voisin] = shared_sommet[warp_id];

				if(visited_label[voisin] == shared_sommet[warp_id])
				{
					unsigned int val_out_queue = 1 << voisin%32;  
					atomicOr(&out_queue[voisin/32],val_out_queue);
					sum = 1;
				}
			}

			// TODO faire à la fin 
			if(__any(sum))
			{
				sum = warpReduceSum(sum);
				if(lane_id == 0)
					atomicAdd(n_out_queue,sum);
			}
			r_gather+=32;
		}
	}
}

__global__ void add_queues(BITMAP_TYPE * in_queue, BITMAP_TYPE * out_queue, BITMAP_TYPE * visited_tex)
{
	int thx = threadIdx.x+blockIdx.x*blockDim.x; 
	out_queue[thx] |= in_queue[thx]; 
	visited_tex[thx] |= out_queue[thx]; 
}

__global__ void add_predecessors(int32_t * received, int32_t * predecessors, int32_t nv)
{
	int thx = threadIdx.x+blockDim.x*blockIdx.x; 
	if(thx < nv)
	{
		int valueRecv = received[thx]; 
		int valueOld = predecessors[thx]; 
		if(valueRecv != -1 && valueOld == -1)
			predecessors[thx] = valueRecv; 
	}
}

__global__ void setup_GPU(int32_t * predecessors, int64_t srcvtx, BITMAP_TYPE * in_queue, BITMAP_TYPE * visited_tex)
{
	predecessors[srcvtx] = (int32_t)srcvtx;
	in_queue[srcvtx/BITMAP_WORD] = 1 << srcvtx%BITMAP_WORD;
	visited_tex[srcvtx/BITMAP_WORD] = 1 << srcvtx%BITMAP_WORD;
}

/* Create the graph structure on the GPUs */
extern "C"
int create_graph_from_edgelist(struct packed_edge * IJ, int64_t nedge)
{
	//printf("create_graph_from_edgelist nedge(%" PRId64 ")\n",nedge);
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
	printf("\n");
	// TODO check this nv != maxvtx
	*max_vtx_out = maxvtx;
	h_n_in_queue = 1; 

	#pragma omp parallel for
	for(unsigned int i = 0 ; i < NGPU ; ++i)
	{
		HANDLE_ERROR(cudaSetDevice(i)); 
		HANDLE_ERROR(cudaMemcpy(d_n_in_queue[i],&h_n_in_queue,sizeof(int32_t),cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemset(d_in_queue[i],0,sizeof(BITMAP_TYPE)*nwords));
		HANDLE_ERROR(cudaMemset(d_visited_tex[i],0,sizeof(BITMAP_TYPE)*nwords));
		HANDLE_ERROR(cudaMemset(d_predecessors[i],-1,sizeof(int32_t)*nv));
		setup_GPU<<<1,1>>>(d_predecessors[i],srcvtx,d_in_queue[i],d_visited_tex[i]);	
	}
	
	int32_t iteration = 0;

	while(1)
	{
		if(iteration++ > 1 << 20)
		{
			fprintf(stderr,"Too many iterations(%d)\n",iteration);
			return -1;
		}

		dim3 dimGrid(nwords/NWARPS,0,0);		
		dim3 dimBlock(32*NWARPS ,0,0);
	
		printf("iter(%2d) n_in_queue(%10d) nblocks(%4d) nthreads(%d) ",
			iteration,
			h_n_in_queue,
			dimGrid.x,
			dimBlock.x);
		fflush(stdout);
	
		#pragma omp parallel for 
		for(unsigned int i = 0 ; i < NGPU ; ++i)
		{
			HANDLE_ERROR(cudaSetDevice(i));
			HANDLE_ERROR(cudaMemset(d_n_out_queue[i],0,sizeof(int32_t)));
			HANDLE_ERROR(cudaMemset(d_out_queue[i],0,sizeof(BITMAP_TYPE)*nwords));
		}
		cudaEventRecord(start);
		
		if(iteration < 3)		
		{
			printf(" CSC ");
			#pragma omp parallel for 
			for(unsigned int i = 0 ; i < NGPU ; ++i)
			{
				HANDLE_ERROR(cudaSetDevice(i)); 
				HANDLE_ERROR(cudaBindTexture(0, tex_visited, d_visited_tex[i],sizeof(BITMAP_TYPE)*nwords));
				explore_frontier_CSC<<< dimGrid.x , dimBlock.x >>>( 	d_in_queue[i], d_out_queue[i], 
											d_predecessors[i],d_visited_tex[i], 
											d_n_out_queue[i], d_CSR_C[i], d_CSR_R[i]);
				HANDLE_ERROR(cudaUnbindTexture(tex_visited));
			}
		}else{
			printf(" CSR ");
			#pragma omp parallel for 
			for(unsigned int i = 0 ; i < NGPU ; ++i)
			{
				HANDLE_ERROR(cudaSetDevice(i)); 
				HANDLE_ERROR(cudaBindTexture(0, tex_in_queue, d_in_queue[i],sizeof(BITMAP_TYPE)*nwords));
				explore_frontier_CSR<<< dimGrid.x , dimBlock.x >>>(	d_out_queue[i], d_predecessors[i],
											d_visited_tex[i], d_n_out_queue[i],
											d_CSR_R[i], d_CSR_C[i]);
				HANDLE_ERROR(cudaUnbindTexture(tex_in_queue));
			}
		}

		cudaEventRecord(stop);
		int32_t total_n_out_queue = 0; 

		//#pragma omp parallel for reduction(+:total_n_out_queue)
		for(unsigned int i = 0 ; i < NGPU ; ++i)
		{
			HANDLE_ERROR(cudaSetDevice(i)); 
			HANDLE_ERROR(cudaMemcpy(&h_n_out_queue,d_n_out_queue[i],sizeof(int32_t),cudaMemcpyDeviceToHost)); 
			total_n_out_queue += h_n_out_queue; 
		}
		cudaEventSynchronize(stop);
		float milliseconds = 0; 
		cudaEventElapsedTime(&milliseconds,start,stop);
		printf("out_queue(%10d) time(%.4f)s \n",total_n_out_queue,milliseconds/1000);	
		if(total_n_out_queue == 0)
		{
			printf("BFS ended\n");
			break;
		}	
		/* Generate in_queue from all the GPUs */
		HANDLE_ERROR(cudaSetDevice(0)); 
		for(unsigned int i = 1 ; i < NGPU ; ++i)
		{
			// Copy in in_queue of gpu 0
			HANDLE_ERROR(cudaMemcpyPeer(d_in_queue[0],0,d_out_queue[i],i,sizeof(BITMAP_TYPE)*nwords));
		       	HANDLE_ERROR(cudaMemcpyPeer(d_tmp_predecessors,0,d_predecessors[i],i,sizeof(int32_t)*nv)); 	
			//HANDLE_ERROR(cudaMemcpy(d_in_queue[0],d_out_queue[i],sizeof(BITMAP_TYPE)*nwords,cudaMemcpyDeviceToDevice));
		       	//HANDLE_ERROR(cudaMemcpy(d_tmp_predecessors,d_predecessors[i],sizeof(int32_t)*nv,cudaMemcpyDeviceToDevice)); 	

			// Call GPU 0 kernel to apply or 
			add_queues <<< (nwords + ((32*NWARPS)/2) )/ (32*NWARPS), 32*NWARPS >>>(d_in_queue[0],d_out_queue[0],d_visited_tex[0]);  
			add_predecessors <<< (nv + (32*NWARPS)/2) / (32*NWARPS), 32*NWARPS >>>(d_tmp_predecessors,d_predecessors[0],nv); 
		}

		#pragma omp parallel for 
		for(unsigned int i = 0 ; i < NGPU ; ++i)
		{
			HANDLE_ERROR(cudaSetDevice(i));
			if(i != 0)
			{
				HANDLE_ERROR(cudaMemcpyPeer(d_predecessors[i],i,d_predecessors[0],0,sizeof(int32_t)*nv));
		       		HANDLE_ERROR(cudaMemcpyPeer(d_visited_tex[i],i,d_visited_tex[0],0,sizeof(BITMAP_TYPE)*nwords));	
				//HANDLE_ERROR(cudaMemcpy(d_predecessors[i],d_predecessors[0],sizeof(int32_t)*nv,cudaMemcpyDeviceToDevice));
		       		//HANDLE_ERROR(cudaMemcpy(d_visited_tex[i],d_visited_tex[0],sizeof(BITMAP_TYPE)*nwords,cudaMemcpyDeviceToDevice));  
			}	
			HANDLE_ERROR(cudaMemcpyPeer(d_in_queue[i],i,d_out_queue[0],0,sizeof(BITMAP_TYPE)*nwords));
			//HANDLE_ERROR(cudaMemcpy(d_in_queue[i],d_out_queue[0],sizeof(BITMAP_TYPE)*nwords,cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy(d_n_in_queue[i],&total_n_out_queue,sizeof(int32_t),cudaMemcpyHostToDevice));
		}
		h_n_in_queue = total_n_out_queue;
	}

	HANDLE_ERROR(cudaSetDevice(0));
	HANDLE_ERROR(cudaMemcpy(h_predecessors,d_predecessors[0],sizeof(int32_t)*nv,cudaMemcpyDeviceToHost));
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
	cudaFree(d_in_queue);
	cudaFree(d_out_queue);
}
