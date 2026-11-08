// The gate from hostgate.c, on the GPU and in whole numbers
#pragma once

// The climate salts from hostgate.c
__constant__ uint64_t GATE_CLIMATE_SALTS[3][2] = {
    {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL},  // Humidity
    {0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL},  // Erosion
    {0xefc8ef4d36102b34ULL, 0x1beeeb324a0f24eaULL}}; // Weirdness

// The octave salts from hostgate.c, already stepped once (a step is just xors and shifts). uploadGpuGate fills this in
__constant__ uint64_t GATE_STEPPED_SALTS[3][2][2];

// The weights from hostgate.c, [climate][half][octave]
static const double GATE_WEIGHTS[3][2][2] = {
    {{1.000, 0.281}, {0.984, 0.202}},  // Humidity
    {{0.464, 0.245}, {0.486, 0.193}},  // Erosion
    {{0.197, 0.182}, {0.209, 0.198}}}; // Weirdness

// Seeds with a humidity part over this don't get erosion and weirdness added (hostgate.c uses 0.5). At 0.45 a quarter fewer seeds
// need the second part, and out of the seeds that pass the gate at 2.5% it loses about 1 in 130 (and 1 in 400 of the good ones)
#define GATE_HUMIDITY_LIMIT 0.45
#ifndef GATE_THREADS
#define GATE_THREADS 128
#endif
#ifndef GATE_ITEMS
#define GATE_ITEMS 8              // How many indexes in a row a thread checks humidity for
#endif
#define GATE_QUEUE (GATE_THREADS * GATE_ITEMS / 2) // About 30% pass humidity, so half of the block's indexes is lots of room

// A xoroshiro step without making a number
DEV void xoroshiroStep(uint64_t &low, uint64_t &high) {
    high ^= low;
    low = rotateLeft(low, 49) ^ high ^ (high << 21);
    high = rotateLeft(high, 28);
}

// The gate in whole numbers. A term is |f - 1/2| in steps of 2^-16 times its weight in steps of 2^-10, so the totals are in steps
// of 2^-26. uploadGpuGate fills these in
__constant__ uint32_t GATE_INT_WEIGHTS[3][2][2];
// The climate salts stepped 0 to 3 times. The generator of a climate starts from the seed's two numbers xored with its salt, and a
// step only shifts and xors, so step^k(seed ^ salt) = step^k(seed) ^ step^k(salt). The climates share the seed's steps that way
__constant__ uint64_t GATE_CLIMATE_STEPS[3][4][2];

// A xoroshiro output (xNextLong without the step)
DEV uint64_t xoroshiroOutput(uint64_t low, uint64_t high) {
    return rotateLeft(low + high, 17) + low;
}

// The seed's two numbers stepped 0 to 3 times
struct SeedSteps {
    uint64_t low[4];
    uint64_t high[4];
};

DEV void fillSeedSteps(uint64_t seedLow, uint64_t seedHigh, SeedSteps &steps) {
    steps.low[0] = seedLow;
    steps.high[0] = seedHigh;
#pragma unroll
    for (int k = 1; k < 4; k++) {
        steps.low[k] = steps.low[k - 1];
        steps.high[k] = steps.high[k - 1];
        xoroshiroStep(steps.low[k], steps.high[k]);
    }
}

// One gate term from an octave's stepped generator: |f - 1/2| in steps of 2^-16 for the y offset's fraction f. It only works out the
// top word of the random number, without the carry from the bottom word (that moves f by 2^-24 at most, and only rarely reaches the
// bits we use)
DEV uint32_t gateTerm(uint64_t low, uint64_t high) {
    uint64_t sum = low + high;
    uint32_t top = __funnelshift_l((uint32_t) sum, (uint32_t) (sum >> 32), 17) + (uint32_t) (low >> 32);
    uint32_t fraction = (top >> 8) & 0xFFFFu;     // the top 16 bits of the 24 bit fraction
    uint32_t below = ((top >> 23) & 1u) - 1u;      // all ones when the fraction is under a half
    return ((fraction ^ below) & 0x7FFFu) - below; // |fraction - 0x8000|
}

/**
 * @brief Returns the gate total for a climate value (in steps of 2^-26), like addClimate in hostgate.c
 *
 * @param climate 0, 1 or 2 (humidity, erosion or weirdness)
 * @param steps The seed's two numbers stepped 0 to 3 times
 */
DEV uint32_t gateClimate(int climate, const SeedSteps &steps) {
    uint32_t total = 0;
#pragma unroll
    for (int half = 0; half < 2; half++) {
        // The half's two numbers are the climate generator's outputs 2 * half and 2 * half + 1
        uint64_t low = xoroshiroOutput(steps.low[2 * half] ^ GATE_CLIMATE_STEPS[climate][2 * half][0],
                                       steps.high[2 * half] ^ GATE_CLIMATE_STEPS[climate][2 * half][1]);
        uint64_t high = xoroshiroOutput(steps.low[2 * half + 1] ^ GATE_CLIMATE_STEPS[climate][2 * half + 1][0],
                                        steps.high[2 * half + 1] ^ GATE_CLIMATE_STEPS[climate][2 * half + 1][1]);
        xoroshiroStep(low, high); // skip the x offset (the octave salts are stepped too)
#pragma unroll
        for (int octave = 0; octave < 2; octave++) {
            uint32_t term = gateTerm(low ^ GATE_STEPPED_SALTS[climate][octave][0], high ^ GATE_STEPPED_SALTS[climate][octave][1]);
            total += (uint32_t) (uint16_t) term * (uint32_t) (uint16_t) GATE_INT_WEIGHTS[climate][half][octave];
        }
    }
    return total;
}

// A seed that passed humidity, waiting for the erosion and weirdness part
struct GateQueueEntry {
    uint64_t seedLow;
    uint64_t seedHigh;
    uint32_t item;  // index - first
    uint32_t total;
};

/**
 * @brief Checks erosion and weirdness for a seed that passed humidity, returns true if it passes the whole gate
 */
DEV bool finishGate(const GateQueueEntry &entry, uint32_t threshold) {
    SeedSteps steps;
    fillSeedSteps(entry.seedLow, entry.seedHigh, steps);
    uint32_t total = entry.total + gateClimate(1, steps);
    if (total <= threshold) {
        total += gateClimate(2, steps);
    }
    return total <= threshold;
}

/**
 * @brief This kernel runs the gate on a batch of stream indexes, the indexes that pass get added to output (not in order).
 * The seeds that pass humidity go in a queue, then the block does erosion and weirdness for the queue.
 * A thread takes GATE_ITEMS indexes in a row. In the fast stream, seed i's xoroshiro starts from mix(i * GOLDEN) and
 * mix((i + 1) * GOLDEN), so the next seed's low half is this seed's high half and the thread only mixes once per seed.
 *
 * @param first The stream index to start at (with the custom seed offset added)
 * @param count The number of indexes to check
 * @param threshold The gate threshold (in steps of 2^-26)
 * @param humidityThreshold The humidity cut (in steps of 2^-26)
 * @param output Where the indexes that pass go
 * @param outputCount The number of indexes that passed (it keeps going up past capacity)
 * @param capacity Room in output
 */
__global__ void __launch_bounds__(GATE_THREADS)
gateKernel(uint64_t first, uint32_t count, uint32_t threshold, uint32_t humidityThreshold, uint64_t *output, uint32_t *outputCount, uint32_t capacity) {
    __shared__ GateQueueEntry queue[GATE_QUEUE];
    __shared__ int queued;
    if (threadIdx.x == 0) {
        queued = 0;
    }
    __syncthreads();

    // Humidity for all of the indexes
    uint32_t threadFirst = blockIdx.x * (GATE_THREADS * GATE_ITEMS) + threadIdx.x * GATE_ITEMS;
    const bool fastStream = STREAM_KIND == STREAM_FAST;
    uint64_t weyl = (first + threadFirst) * GOLDEN_RATIO_64;
    uint64_t nextMix = fastStream ? mix64(weyl) : 0;
#pragma unroll 2
    for (int item = 0; item < GATE_ITEMS; item++) {
        uint32_t i = threadFirst + item;
        if (i >= count) {
            break;
        }
        XoroshiroState random;
        if (fastStream) {
            weyl += GOLDEN_RATIO_64;
            random.low = nextMix;
            nextMix = mix64(weyl);
            random.high = nextMix;
        } else {
            xSetSeed(&random, mix64(weyl));
            weyl += GOLDEN_RATIO_64;
        }
        uint64_t seedLow = xNextLong(&random);
        uint64_t seedHigh = xoroshiroOutput(random.low, random.high);
        SeedSteps steps;
        fillSeedSteps(seedLow, seedHigh, steps);
        uint32_t total = gateClimate(0, steps);
        if (total <= humidityThreshold) {
            GateQueueEntry entry;
            entry.seedLow = seedLow;
            entry.seedHigh = seedHigh;
            entry.item = i;
            entry.total = total;
            // A shared atomic for each seed (one for the warp with a ballot was slower)
            int slot = atomicAdd(&queued, 1);
            if (slot < GATE_QUEUE) {
                queue[slot] = entry;
            } else if (finishGate(entry, threshold)) {
                // The queue is full (that shouldn't really happen), so this one gets done right away
                uint32_t outputSlot = atomicAdd(outputCount, 1u);
                if (outputSlot < capacity) {
                    output[outputSlot] = first + i;
                }
            }
        }
    }
    __syncthreads();

    // Then erosion and weirdness for the queue
    int queueLength = min(queued, GATE_QUEUE);
    for (int base = 0; base < queueLength; base += GATE_THREADS) {
        int queueIndex = base + threadIdx.x;
        bool passes = false;
        uint64_t index = 0;
        if (queueIndex < queueLength) {
            GateQueueEntry entry = queue[queueIndex];
            passes = finishGate(entry, threshold);
            index = first + entry.item;
        }

        // Lane zero gets room in output for the warp, then the threads that passed write their indexes
        unsigned passedLanes = __ballot_sync(0xFFFFFFFFu, passes);
        if (passedLanes) {
            int lane = threadIdx.x & 31;
            uint32_t warpFirst = 0;
            if (lane == 0) {
                warpFirst = atomicAdd(outputCount, (uint32_t) __popc(passedLanes));
            }
            warpFirst = __shfl_sync(0xFFFFFFFFu, warpFirst, 0);
            uint32_t slot = warpFirst + __popc(passedLanes & ((1u << lane) - 1));
            if (passes && slot < capacity) {
                output[slot] = index;
            }
        }
    }
}

/**
 * @brief This method does a xoroshiro step on the octave salts and copies the results to the GPU (for GATE_STEPPED_SALTS)
 */
static void uploadGpuGate() {
    const uint64_t climateSalts[3][2] = {
        {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL},  // Humidity
        {0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL},  // Erosion
        {0xefc8ef4d36102b34ULL, 0x1beeeb324a0f24eaULL}}; // Weirdness
    const uint64_t octaveSalts[3][2][2] = {
        {{0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}, {0xf11268128982754fULL, 0x257a1d670430b0aaULL}},  // Humidity
        {{0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL}, {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}},  // Erosion
        {{0xf11268128982754fULL, 0x257a1d670430b0aaULL}, {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL}}}; // Weirdness
    auto step = [](uint64_t &low, uint64_t &high) {
        high ^= low;
        low = ((low << 49) | (low >> 15)) ^ high ^ (high << 21);
        high = (high << 28) | (high >> 36);
    };
    uint64_t stepped[3][2][2];
    uint64_t climateSteps[3][4][2];
    uint32_t weights[3][2][2];
    for (int climate = 0; climate < 3; climate++) {
        for (int octave = 0; octave < 2; octave++) {
            uint64_t low = octaveSalts[climate][octave][0], high = octaveSalts[climate][octave][1];
            step(low, high);
            stepped[climate][octave][0] = low;
            stepped[climate][octave][1] = high;
        }
        uint64_t low = climateSalts[climate][0], high = climateSalts[climate][1];
        for (int k = 0; k < 4; k++) {
            climateSteps[climate][k][0] = low;
            climateSteps[climate][k][1] = high;
            step(low, high);
        }
        for (int half = 0; half < 2; half++) {
            for (int octave = 0; octave < 2; octave++) {
                weights[climate][half][octave] = (uint32_t) llround(GATE_WEIGHTS[climate][half][octave] * 1024.0);
            }
        }
    }
    cudaMemcpyToSymbol(GATE_STEPPED_SALTS, stepped, sizeof(stepped));
    cudaMemcpyToSymbol(GATE_CLIMATE_STEPS, climateSteps, sizeof(climateSteps));
    cudaMemcpyToSymbol(GATE_INT_WEIGHTS, weights, sizeof(weights));
}

// A gate total (like GATE_THRESHOLD) in the GPU gate's whole number steps
static uint32_t gateSteps(double total) {
    return (uint32_t) llround(total * 67108864.0); // 2^26
}
