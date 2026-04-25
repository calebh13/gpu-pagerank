#ifndef HEAP
#define HEAP
#include <stdlib.h>
#include <stdbool.h>

typedef struct MaxHeap {
    void** arr;
    size_t cur_size;
    size_t max_size;
} MinHeap;

// If max_size < 0, then it will be unbounded
MinHeap* MinHeap_init(size_t max_size);
void MinHeap_insert(MinHeap* heap, void* x, int (*cmp)(const void *, const void *));
void* MinHeap_pop(MinHeap* heap, int (*compar)(const void *, const void *));
void* MinHeap_peek(MinHeap* heap);
bool MinHeap_empty(MinHeap* heap);

#endif