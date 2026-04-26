#include "graph.h"
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <inttypes.h>
#include <stdbool.h>
#include <time.h>

// Returns 1 if line contained a valid edge, 0 otherwise.
// On success, writes the two endpoints into *in_node and *out_node.
int parse_edge(const char* line, uint32_t* in_node, uint32_t* out_node)
{
    // Skip comments
    if (line[0] == '#' || line[0] == '/') return 0;

    // Advance to next number
    int i = 0;
    for (; line[i] != '\0' && !isdigit(line[i]); i++);
    if (line[i] == '\0') return 0;

    char* endptr;
    *in_node = strtoull(line + i, &endptr, 10);

    // Advance to next number again
    for (; *endptr != '\0' && !isdigit(*endptr); endptr++);
    if (*endptr == '\0') return 0;

    *out_node = strtoull(endptr, NULL, 10);
    return 1;
}

// Parses the input file and determines how many uniques vertices and edges are in the graph
static void set_edge_and_vertex_counts(Graph* g, FILE* file)
{
    char* line = NULL;
    size_t n = 1;
    uint32_t max = 0, edges = 0, in_node, out_node;
    while (getline(&line, &n, file) != -1) {
        if (!parse_edge(line, &in_node, &out_node)) continue;
        edges++;
        if (in_node > max) max = in_node;
        if (out_node > max) max = out_node;
    }
    
    g->edge_count = edges;
    g->vertex_count = max + 1;
    free(line);
}

// For now, this is assuming no dictionary is needed.
static void build_edges_and_offsets(Graph* g, FILE* file)
{
    char* line = NULL;
    size_t n = 1;
    uint32_t edges = 0, in_node = 0, out_node = 0, prev_node = -1;
    while (getline(&line, &n, file) != -1) {
        if (!parse_edge(line, &in_node, &out_node)) continue;
        /* CSR logic: we make a list of all outgoing edges, and if we are starting a new node, 
         * record in `offsets` the index of where its neighbors begin in `edges`. */
        g->edges[edges] = out_node;
        if (in_node != prev_node) {
            /* Since the list is in sorted order, everything we didn't see in between
             * `in_node` and `prev_node` has the same offset as `in_node`; they have no edges. 
             * Also note that unsigned overflow is defined behavior in C, so this is fine. */
            for (uint32_t i = prev_node + 1; i < in_node; i++) {
                g->offsets[i] = edges;
            }
            g->offsets[in_node] = edges;
        }
        prev_node = in_node;
        edges++;
    }
    // Fill any trailing nodes that had no outgoing edges
    for (uint32_t i = prev_node + 1; i < g->vertex_count; i++) {
        g->offsets[i] = g->edge_count;
    }
    // Sentinel to allow `g->offsets[in_node + 1]` to always work
    g->offsets[g->vertex_count] = g->edge_count;
    free(line);
}

// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
// switching from xorshift64 to xorshift32 since we need very fast 32-bit random integers
// and curand is too expensive.
__device__ uint32_t xorshift32(uint32_t x) {
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
    return x;
}

// d_threshold is precomputed on CPU as (uint32_t)(D * UINT32_MAX) to avoid floating point division at each step
__global__ void monte_carlo_kernel(const uint32_t* __restrict__ d_offsets, const uint32_t* __restrict__ d_edges, uint32_t* d_counts, 
    const uint32_t vertex_count, const uint32_t K, const uint32_t d_threshold, uint32_t seed)
{
    uint32_t cur_node = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_node >= vertex_count) return;

    uint32_t rng = seed ^ (cur_node * (uint32_t)6700417); // just need to seed with a big prime
    if (rng == 0) rng = 1; // 0 will cause rng to get stuck. 1 in 4 billion chance though
    for(uint32_t step = 0; step < K; step++) {
        // __ldg works because this is treated as read-only; gives speedup
        uint32_t start = __ldg(&d_offsets[cur_node]);
        uint32_t num_neighbors = __ldg(&d_offsets[cur_node + 1]) - start;
        rng = xorshift32(rng);
        if (num_neighbors < 1 || rng < d_threshold) {
            rng = xorshift32(rng);
            cur_node = rng % vertex_count;
        } else {
            rng = xorshift32(rng);
            cur_node = __ldg(&d_edges[cur_node + rng % num_neighbors]);
        }
        
        atomicAdd(&d_counts[cur_node], 1u);
    }
}

// We take in counts and not pageranks to save memory on GPU
void calculate_pageranks(Graph* G, Pagerank* pageranks, const double D, const uint32_t K, int num_to_show) 
{
    uint32_t* d_offsets, *d_edges, *d_counts;
    cudaError_t err;
    err = cudaMalloc(&d_offsets, (G->vertex_count + 1) * sizeof(uint32_t));
    if (err) {
        fprintf(stderr, "Error allocating offsets array on device\n");
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_offsets, G->offsets, (G->vertex_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);  

    cudaMalloc(&d_edges, (G->edge_count) * sizeof(uint32_t));
    if (err) {
        fprintf(stderr, "Error allocating offsets array on device\n");
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_edges, G->edges, (G->edge_count) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_counts, (G->vertex_count) * sizeof(uint32_t));
    if (err) {
        fprintf(stderr, "Error allocating offsets array on device\n");
        exit(EXIT_FAILURE);
    }
    cudaMemset(d_counts, 0, (G->vertex_count) * sizeof(uint32_t));
    
    // now we launch the kernel, simulate the walks, etc.
    int block_size = 256;
    int grid_size = (G->vertex_count + block_size - 1) / block_size;
    
    srand((unsigned int)time(NULL));
    uint32_t seed = (uint32_t)rand();
    
    monte_carlo_kernel<<<grid_size, block_size>>>(d_offsets, d_edges, d_counts, 
        G->vertex_count, K, (uint32_t)(D * UINT32_MAX), seed);
    
    // Copy counts from GPU to host. We didn't send the GPU a Pagerank array to save memory (allows better caching)
    uint32_t* counts = (uint32_t*)malloc(G->vertex_count * sizeof(uint32_t));
    if (!counts) {
        fprintf(stderr, "Failed to allocate counts array\n");
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(counts, d_counts, G->vertex_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for(uint32_t i = 0; i < G->vertex_count; i++) {
        pageranks[i].hits = counts[i];
        if (DEBUG) printf("Node %" PRIu32 ": %" PRIu32 "\n", i, counts[i]);
    }

    free(counts);
    cudaFree(d_offsets);
    cudaFree(d_edges);
    cudaFree(d_counts);
}

Graph* init_graph(FILE* file) 
{
    Graph* g = (Graph*)calloc(1, sizeof(Graph));
    if (!g) exit(EXIT_FAILURE);
    set_edge_and_vertex_counts(g, file);
    rewind(file);
    if (g->edge_count > UINT32_MAX || g->vertex_count > UINT32_MAX) {
        uint32_t larger = (g->edge_count > g->vertex_count) ? g->edge_count : g->vertex_count;
        fprintf(stderr, "error: too many vertices or edges (%" PRIu32 ") for size_t (%zu)", larger, SIZE_MAX);
        exit(EXIT_FAILURE);
    }
    
    g->edges = (uint32_t*)calloc((size_t)g->edge_count, sizeof(uint32_t));
    g->offsets = (uint32_t*)calloc((size_t)g->vertex_count + 1, sizeof(uint32_t)); // includes sentinel at end
    if (!g->edges || !g->offsets) exit(EXIT_FAILURE);
    build_edges_and_offsets(g, file);
    rewind(file);
    return g;
}

Pagerank* init_pageranks(uint64_t vertex_count)
{
    Pagerank* pageranks = (Pagerank*)malloc(vertex_count * sizeof(Pagerank));
    if (!pageranks) return NULL;
    for(uint64_t i = 0; i < vertex_count; i++) {
        pageranks[i].hits = 0;
        pageranks[i].idx = i;
    }

    return pageranks;
}

int pagerank_cmp(const void* a, const void* b)
{
    Pagerank* pr_a = (Pagerank*)a, *pr_b = (Pagerank*)b;
    return (pr_a->hits > pr_b->hits) - (pr_a->hits < pr_b->hits);
}


void free_graph(Graph* g)
{
    free(g->edges);
    free(g->offsets);
    free(g);
}