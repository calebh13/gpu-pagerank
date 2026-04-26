#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <getopt.h>
// #include <omp.h>
#include <cuda.h>

#include "graph.h"
#include "test_graph.h"
#include "heap.h"

// const int SEC_TO_US = 1000000;
const bool DEBUG = true;

int main(int argc, char* argv[])
{
    uint32_t K = 0;
    double D = 0.0;
    char* filename = NULL;
    int opt;

    // parse command line options
    while ((opt = getopt(argc, argv, "k:d:")) != -1) {
        switch (opt) {
            case 'k':
                K = (uint64_t)strtoull(optarg, NULL, 10);
                break;
            case 'd':
                D = strtod(optarg, NULL);
                break;
            default: 
                printf("Usage: %s -k <K> -d <D> -p <p> filename\n", argv[0]); 
                printf("\tK: length of random walk (max val. 2^64 - 1)\n"); 
                printf("\tD: damping ratio - probability of jumping to random node (from 0 to 1)\n"); 
                printf("\tFilename: filename of input graph to simulate walks on\n"); 
                exit(EXIT_FAILURE);
        }
    }
    
    if (K == 0) {
        fprintf(stderr, "%s: missing -k argument\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    if (D == 0.0) {
        fprintf(stderr, "%s: missing -d argument\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // after options, the remaining argument should be the filename
    if (optind >= argc) {
        fprintf(stderr, "%s: expected filename after options\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    filename = argv[optind];

    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    Graph* G = init_graph(file);
    if (DEBUG) test_graph(G, file); 
    Pagerank* pageranks = init_pageranks(G->vertex_count);

    const int num_to_show = 5;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    calculate_pageranks(G, pageranks, D, K, num_to_show);
    clock_gettime(CLOCK_MONOTONIC, &end);
    long long us = (long long)(end.tv_sec - start.tv_sec) * 1e6 + (long long)((end.tv_nsec - start.tv_nsec) / 1000);
    printf("D,K,p,time (us),\n");
    printf("%.1f,%" PRIu32 ",%lld\n", D, K, us);

    // time this too, to see how long the heap stuff actually takes
    clock_gettime(CLOCK_MONOTONIC, &start);
    MinHeap* heap = MinHeap_init(num_to_show);
    for(uint64_t i = 0; i < G->vertex_count; i++) {
        if (heap->cur_size == heap->max_size) {
            // If we're at max size, only insert if we're adding a new better value.
            if (pageranks[i].hits > ((Pagerank*)MinHeap_peek(heap))->hits) {
                MinHeap_pop(heap, pagerank_cmp);
                MinHeap_insert(heap, &pageranks[i], pagerank_cmp);
            }
        } else {
            // if we're not at max size, get there
            MinHeap_insert(heap, &pageranks[i], pagerank_cmp);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    us = (long long)(end.tv_sec - start.tv_sec) * 1e6 + (long long)((end.tv_nsec - start.tv_nsec) / 1000);
    printf("Heap time: %lld us\n", us);
    
    printf("Top %d nodes: \n", num_to_show);
    for(int i = 0; i < num_to_show; i++) {
        Pagerank* max = (Pagerank*)MinHeap_pop(heap, pagerank_cmp);
        // compute fraction of visits that occurred at this node
        double rank = ((double)max->hits / K) / G->vertex_count;
        printf("%d. Node %" PRIu32 ": %.7f\n", (num_to_show - i), max->idx, rank);
    }
    printf("\n");    

    free(pageranks);
    free_graph(G);
    fclose(file);
    return 0;
}