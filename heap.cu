#include "heap.h"

/* This heap is 1-indexed! */

// If initial_capacity is not positive, then it is ignored
MinHeap* MinHeap_init(size_t initial_capacity)
{
    MinHeap* heap = (MinHeap*)malloc(sizeof(MinHeap));
    if (!heap) exit(EXIT_FAILURE);
    heap->cur_size = 0;
    heap->max_size = (initial_capacity > 0) ? initial_capacity : 8; // default val
    heap->arr = (void**)calloc(heap->max_size + 1, sizeof(void*));
    if (!heap->arr) exit(EXIT_FAILURE);
    return heap;
}

void MinHeap_insert(MinHeap* heap, void* x, int (*cmp)(const void *, const void *))
{
    if (heap->cur_size == heap->max_size) {
        heap->max_size *= 2;
        heap->arr = (void**)realloc(heap->arr, (heap->max_size + 1) * sizeof(void*));        
        if (!heap->arr) exit(EXIT_FAILURE);
    }

    size_t pos = ++heap->cur_size;

    // bubble up smaller values
    for (; pos > 1 && cmp(x, heap->arr[pos / 2]) < 0; pos /= 2) {
        heap->arr[pos] = heap->arr[pos / 2];
    }

    heap->arr[pos] = x;
}

void* MinHeap_pop(MinHeap* heap, int (*cmp)(const void *, const void *))
{
    if (heap->cur_size == 0) exit(EXIT_FAILURE);

    void* max = heap->arr[1];
    void* last_elem = heap->arr[heap->cur_size--];
    size_t pos = 1;
    size_t child;

    // percolate down
    while (pos * 2 <= heap->cur_size) {
        child = pos * 2;

        // pick smaller child
        if (child < heap->cur_size && cmp(heap->arr[child + 1], heap->arr[child]) < 0) {
            child++;
        }
        if (cmp(last_elem, heap->arr[child]) <= 0) break;
        heap->arr[pos] = heap->arr[child];
        pos = child;
    }

    heap->arr[pos] = last_elem;
    return max;
}

void* MinHeap_peek(MinHeap* heap)
{
    return heap->arr[1];
}

bool MinHeap_empty(MinHeap* heap)
{
    return heap->cur_size == 0;
}