#include <inttypes.h>

typedef struct XorshiftState {
    uint64_t x;
    uint64_t y;
} XorshiftState;

void seed_state(XorshiftState* state, int pid);
uint64_t random_u64(XorshiftState* state);
double random_double(XorshiftState* state);