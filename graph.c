#include "graph.h"
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "xorshift.h"
#include <stdbool.h>

// Returns 1 if line contained a valid edge, 0 otherwise.
// On success, writes the two endpoints into *in_node and *out_node.
int parse_edge(const char* line, uint64_t* in_node, uint64_t* out_node)
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
    uint64_t max = 0, edges = 0, in_node, out_node;
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
    uint64_t edges = 0, in_node, out_node, prev_node = -1;
    while (getline(&line, &n, file) != -1) {
        if (!parse_edge(line, &in_node, &out_node)) continue;
        /* CSR logic: we make a list of all outgoing edges, and if we are starting a new node, 
         * record in `offsets` the index of where its neighbors begin in `edges`. */
        g->edges[edges] = out_node;
        if (in_node != prev_node) {
            /* Since the list is in sorted order, everything we didn't see in between
             * `in_node` and `prev_node` has the same offset as `in_node`; they have no edges. 
             * Also note that unsigned overflow is defined behavior in C, so this is fine. */
            for (uint64_t i = prev_node + 1; i < in_node; i++) {
                g->offsets[i] = edges;
            }
            g->offsets[in_node] = edges;
        }
        prev_node = in_node;
        edges++;
    }
    // Fill any trailing nodes that had no outgoing edges
    for (uint64_t i = prev_node + 1; i < g->vertex_count; i++) {
        g->offsets[i] = g->edge_count;
    }
    // Sentinel to allow `g->offsets[in_node + 1]` to always work
    g->offsets[g->vertex_count] = g->edge_count;
    free(line);
}

Pagerank* init_pageranks(uint64_t vertex_count)
{
    Pagerank* pageranks = malloc(vertex_count * sizeof(Pagerank));
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

void calculate_pageranks(Graph* G, Pagerank* pageranks,  double d, uint64_t k, int p) {
    omp_set_num_threads(p);
    
    // Each thread should get a chunk of the array. We know how long the array is, and all chunks are equal work, so just use static scheduling
    #pragma omp parallel
    {     
        XorshiftState state;
        seed_state(&state, omp_get_thread_num());

        #pragma omp for schedule(static)
        for(uint64_t i = 0; i < G->vertex_count; i++){
            uint64_t curnode = i;

            // Repeats for the number of steps in the walk
            for(uint64_t walk_len = 0; walk_len < k; walk_len++){
                uint64_t num_neighbors = G->offsets[curnode + 1] - G->offsets[curnode];
                if(num_neighbors < 1) {
                    // this functionality needs to be checked with ananth. for now, we jump
                    // we could also do coin flip here and if tails stay and NOT increment pagerank
                    // and if heads jump.

                    curnode = random_u64(&state) % G->vertex_count;
                    continue;
                }

                // If we land on tails, then we look to neighbors. Otherwise, Grab a random node
                bool tails =  random_double(&state) > d;

                if(tails){
                    uint64_t neighbor = random_u64(&state) % num_neighbors;
                    curnode = G->edges[G->offsets[curnode] + neighbor];
                } else {
                    curnode = random_u64(&state) % G->vertex_count;
                }

                #pragma omp atomic update
                pageranks[curnode].hits++;
            }
        }
    }
}

Graph* init_graph(FILE* file) 
{
    Graph* g = calloc(1, sizeof(Graph));
    if (!g) exit(EXIT_FAILURE);
    set_edge_and_vertex_counts(g, file);
    rewind(file);
    if (g->edge_count > UINT64_MAX || g->vertex_count > UINT64_MAX) {
        uint64_t larger = (g->edge_count > g->vertex_count) ? g->edge_count : g->vertex_count;
        fprintf(stderr, "error: too many vertices or edges (%"PRIu64") for size_t (%zu)", larger, SIZE_MAX);
        exit(EXIT_FAILURE);
    }
    
    g->edges = calloc((size_t)g->edge_count, sizeof(uint64_t));
    g->offsets = calloc((size_t)g->vertex_count + 1, sizeof(uint64_t)); // includes sentinel at end
    if (!g->edges || !g->offsets) exit(EXIT_FAILURE);
    build_edges_and_offsets(g, file);
    rewind(file);
    return g;
}

void free_graph(Graph* g)
{
    free(g->edges);
    free(g->offsets);
    free(g);
}