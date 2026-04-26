#ifndef GRAPH
#define GRAPH
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct graph {
    uint32_t* edges;
    uint32_t* offsets;
    uint32_t vertex_count;
    uint32_t edge_count;
} Graph;

typedef struct pagerank {
    uint32_t hits;
    uint32_t idx;
} Pagerank;

int parse_edge(const char* line, uint32_t* in_node, uint32_t* out_node);
Graph* init_graph(FILE* file);
void free_graph(Graph* g);
Pagerank* init_pageranks(uint64_t vertex_count);
int pagerank_cmp(const void* a, const void* b);
void calculate_pageranks(Graph* G, Pagerank* pageranks, const double D, const uint32_t K, int num_to_show); 

extern const bool DEBUG;

/* 
 * Instead of making a list of nodes, where each node has an adjacency list,
 * we will use compressed sparse row format.
 * 
 * Because the only property we care about is a node's number/ID, we don't even need to make a Node struct.
 * We will only examine the edges between nodes, so that is what we will store.

 * To do this, we'll need two arrays: `edges` and `offsets`.
 * `edges` will store every single node's edges in order.
 * Suppose a graph has this structure:
 * Node 0 -> 2,5
 * Node 1 -> 0
 * Node 2 -> nothing
 * Node 3 -> 1,3,4
 * Node 4 -> nothing
 * This results in `edges = {2, 5, 0, 1, 3, 4}`
 * But notice we have no idea when node 0's neighbors stop and node 1's neighbors begin.
 * To determine this, we will use `offsets`. 
 * offsets[i] = index of node i's first neighbor in `edges`.
 *
 * So, for the above graph:
 * offsets[] = {
 * 0, // node 0 neighbors start at edges[0]
 * 2, // node 1 neighbors start at edges[2]
 * 3, // node 2 has no neighbors
 * 3, // node 3 neighbors start at edges[3]
 * 6, // node 4 neighbors start at edges[6]
 * 6  // sentinel: total number of edges (we're done)
 * };
 *
 * So to process all the neighbors of node i, you would use:
 * size_t start = offsets[i], end = offsets[i+1];
 * Then iterate like `for (size_t j = start; j < end; j++)`
 * and each neighbor will be `edges[j]`.
 * 
 * This requires 2 passes over the input file. One to find both the number of nodes and edges and malloc accordingly
 * (which results in only two mallocs instead of millions), and another to actually build the edges and offsets. 
 * The size of `edges` is of course the number of edges, while `offsets` is the number of nodes.
 * 
 * Note that if the node numbers are extremely large (e.g. random 64-bit integers), this would require
 * associating them with a dictionary to put them into a dense range from 0 to [number of nodes] - 1.
 * Once they are "compressed" into that range, and the edges/offsets are built accordingly, the algorithm
 * proceeds as normal - this is just an additional upfront cost.
 */
#endif
