#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#include "xorshift.h"

void seed_state(XorshiftState* state, int tid)
{
    unsigned int seed = (unsigned int)time(NULL) ^ tid;
    state->x = rand_r(&seed);
    state->y = rand_r(&seed);
}

/* Source: https://journals.tubitak.gov.tr/cgi/viewcontent.cgi?article=2086&context=elektrik */
uint64_t random_u64(XorshiftState* state)
{
    uint64_t x = state->x;
    uint64_t const y = state->y;
    state->x = y;
    x ^= x << 23; // a, shift & xor
    x ^= x >> 17; // b, shift & xor
    x ^= y;
    state->y = x + y;
    return x;
}

double random_double(XorshiftState* state)
{
    return (double)random_u64(state) / UINT64_MAX;
}