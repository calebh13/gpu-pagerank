#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <getopt.h>
#include <omp.h>

#include "graph.h"
#include "heap.h"

const int SEC_TO_US = 1000000;
const bool DEBUG = false;

int main(int argc, char* argv[])
{
    uint64_t K = 0;
    double D = 0.0;
    int p = 0;
    char* filename = NULL;
    int opt;

    // parse command line options
    while ((opt = getopt(argc, argv, "k:d:p:")) != -1) {
        switch (opt) {
            case 'k':
                K = (uint64_t)strtoull(optarg, NULL, 10);
                break;
            case 'd':
                D = strtod(optarg, NULL);
                break;
            case 'p':
                p = (int)strtol(optarg, NULL, 10);
                break;
            default: 
                printf("Usage: %s -k <K> -d <D> -p <p> filename\n", argv[0]); 
                printf("\tK: length of random walk (max val. 2^64 - 1)\n"); 
                printf("\tD: damping ratio - probability of jumping to random node (from 0 to 1)\n"); 
                printf("\np: number of processes to spawn\n");
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
    if (p == 0) {
        p = omp_get_num_procs();
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

    Pagerank* pageranks = init_pageranks(G->vertex_count);
    double start = omp_get_wtime();
    calculate_pageranks(G, pageranks, D, K, p);
    double end = omp_get_wtime();
    printf("D,K,p,time (us),\n");
    printf("%.1f,%"PRIu64",%d,%lld\n", D, K, p, (long long)((end - start) * SEC_TO_US));

    // now add to heap and keep size limited
    const int num_to_show = 5;
    
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
    
    printf("Top %d nodes: \n", num_to_show);
    for(int i = 0; i < num_to_show; i++) {
        Pagerank* max = (Pagerank*)MinHeap_pop(heap, pagerank_cmp);
        // compute fraction of visits that occurred at this node
        double rank = ((double)max->hits / K) / G->vertex_count;
        printf("%d. Node %"PRIu64": %.7f\n", (num_to_show - i), max->idx, rank);
    }
    printf("\n");    

    free(pageranks);
    free(heap->arr);
    free(heap);
    free_graph(G);
    fclose(file);
    return 0;
}