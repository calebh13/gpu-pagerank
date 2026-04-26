#include "test_graph.h"
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

static int edge_exists(Graph* g, uint32_t in_node, uint32_t out_node)
{
    uint32_t start = g->offsets[in_node];
    uint32_t end = g->offsets[in_node + 1];
    for (uint32_t i = start; i < end; i++) {
        if (g->edges[i] == out_node) return 1;
    }
    return 0;
}

void test_graph(Graph* g, FILE* file)
{
    char* line = NULL;
    size_t n = 1;
    uint32_t in_node, out_node, edges = 0, failures = 0;

    while (getline(&line, &n, file) != -1) {
        if (!parse_edge(line, &in_node, &out_node)) continue;
        edges++;
        if (!edge_exists(g, in_node, out_node)) {
            fprintf(stderr, "FAIL: edge %" PRIu32 " -> %" PRIu32 " not found\n", in_node, out_node);
            failures++;
        }
    }

    free(line);
    printf("%" PRIu32 "/%" PRIu32 " edges verified OK\n", edges - failures, edges);
    if (!failures) printf("Tests passed!\n"); else exit(EXIT_FAILURE);
}