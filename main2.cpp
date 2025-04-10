#include <iostream>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <bitset>
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <linux/kernel-page-flags.h>
#include <stdint.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <map>
#include <list>
#include <utility>
#include <math.h>
#include <fstream>
#include <x86intrin.h>
#include <string>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <random>
#include <set>
#include <sys/syscall.h>
#include <errno.h>

// If your system headers don't define these, we define them here:
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT) // 2MB huge page
#endif

#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT) // 1GB huge page
#endif

// These headers presumably contain your custom logic for reading IMC info, etc.
#include "solver.h"
#include "sysinfo.h"

using namespace std;

//=============================================================================
// Globals & Constants
//=============================================================================

// File descriptor for /proc/self/pagemap
static int g_pagemap_fd = -1;

// Minimum PMU counter we accept as a "successful" measurement.
// If results are below this threshold, the measurement is discarded.
static const long long pmuThreshold = 3000;

// We store addresses in maps keyed by channel, rank, bank, etc.
typedef std::map<size_t, std::vector<uint64_t>> AddressSet;

//=============================================================================
// Allocation routines (try 1GB, 2MB, then 4KB pages).
//=============================================================================
/**
 * Attempt to allocate the requested size using 1GB huge pages.
 * Returns a pointer on success, or MAP_FAILED on failure.
 */
void* tryAllocate1Gb(uint64_t size)
{
    void* space = mmap(nullptr, size,
                       PROT_READ | PROT_WRITE,
                       MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE |
                       MAP_HUGETLB | MAP_HUGE_1GB,
                       -1, 0);
    return space;
}

/**
 * Attempt to allocate the requested size using 2MB huge pages.
 * Returns a pointer on success, or MAP_FAILED on failure.
 */
void* tryAllocate2Mb(uint64_t size)
{
    void* space = mmap(nullptr, size,
                       PROT_READ | PROT_WRITE,
                       MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE |
                       MAP_HUGETLB | MAP_HUGE_2MB,
                       -1, 0);
    return space;
}

/**
 * Attempt to allocate the requested size using normal 4KB pages.
 * Returns a pointer on success, or MAP_FAILED on failure.
 */
void* tryAllocate4Kb(uint64_t size)
{
    void* space = mmap(nullptr, size,
                       PROT_READ | PROT_WRITE,
                       MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE,
                       -1, 0);
    return space;
}

/**
 * Tries to allocate 'size' bytes in memory, preferring 1GB pages,
 * then 2MB pages, then standard 4KB pages. Also attempts mlock() to
 * pin the memory, preventing swapping.
 */
void* allocate(uint64_t size)
{
    uint64_t sizeGb = size / (1024ULL * 1024ULL * 1024ULL);

    // 1GB pages
    {
        void* space = tryAllocate1Gb(size);
        if (space != MAP_FAILED)
        {
            if (mlock(space, size) == 0)
            {
                std::cout << "Allocated " << sizeGb << "GB using 1GB pages\n";
                return space;
            }
            munmap(space, size);
        }
    }

    // 2MB pages
    {
        void* space = tryAllocate2Mb(size);
        if (space != MAP_FAILED)
        {
            if (mlock(space, size) == 0)
            {
                std::cout << "Allocated " << sizeGb << "GB using 2MB pages\n";
                return space;
            }
            munmap(space, size);
        }
    }

    // 4KB pages
    {
        void* space = tryAllocate4Kb(size);
        if (space != MAP_FAILED)
        {
            if (mlock(space, size) == 0)
            {
                std::cout << "Allocated " << sizeGb << "GB using 4KB pages\n";
                return space;
            }
            munmap(space, size);
        }
    }

    // If all attempts fail, report failure
    std::cout << "Failed to allocate " << sizeGb << "GB\n";
    return nullptr;
}

//=============================================================================
// Pagemap & Physical Address Routines
//=============================================================================
/**
 * Extracts the frame number from a pagemap value.
 * The PFN is in bits 0..53 of the returned value.
 */
uint64_t frameNumberFromPagemap(uint64_t value)
{
    // bits 0..53 are the page frame number
    return value & ((1ULL << 54) - 1);
}

/**
 * Opens /proc/self/pagemap so we can read it later to convert
 * virtual -> physical addresses.
 */
void initPagemap()
{
    g_pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    assert(g_pagemap_fd >= 0);
}

/**
 * Retrieves the physical address for the given virtual address
 * using the /proc/self/pagemap interface.
 */
uint64_t getPhysicalAddr(uint64_t virtualAddr)
{
    uint64_t value;
    off_t offset = (virtualAddr / 4096ULL) * sizeof(value);
    int got = pread(g_pagemap_fd, &value, sizeof(value), offset);
    assert(got == 8);

    // bit 63 => "page present" must be set
    assert(value & (1ULL << 63));

    uint64_t frame_num = frameNumberFromPagemap(value);

    // Combine with offset-in-page:
    return (frame_num * 4096ULL) | (virtualAddr & 0xFFFULL);
}

//=============================================================================
// Perf Event (PMU) Routines to measure memory channel/rank/bank usage
//=============================================================================

/**
 * Sets up a perf_event measurement (PMU) for a given CPU, channel, rank, bank,
 * using the PMU type provided (default is 0xd, but user can override -c <hex>).
 *
 * @param cpuid   The CPU to bind to.
 * @param channel The memory channel ID.
 * @param rank    The DIMM rank ID.
 * @param bank    The memory bank ID.
 * @param pmuType The perf_event_attr.type to use (e.g. 0xd).
 * @return        File descriptor for the perf event, or < 0 on error.
 */
int setupMeasure(int cpuid, unsigned int channel, unsigned int rank, unsigned int bank, unsigned int pmuType)
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));

    // Get the IMCs from SysInfo (your platform-specific code might differ).
    auto imcs = SysInfo::getImcs();
    if (imcs.empty())
    {
        std::cerr << "No memory controller PMU found\n";
        exit(EXIT_FAILURE);
    }

    // Typically, Intel IMC is type=0xd, but user can override with -c <val>.
    pe.type = pmuType;
    pe.size = sizeof(struct perf_event_attr);

    // config = (bank << 8) + (0xb0 + rank)
    unsigned int bankBits = (bank << 8);
    unsigned int rankBits = 0xb0 + rank; 
    pe.config = bankBits | rankBits;

    pe.read_format   = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    pe.sample_type   = PERF_SAMPLE_IDENTIFIER;
    pe.disabled      = 1;
    pe.exclude_kernel= 0;
    pe.exclude_hv    = 0;
    pe.precise_ip    = 0;

    // Create the perf_event FD for the specified CPU.
    int fd = syscall(__NR_perf_event_open, &pe, -1, cpuid, -1, 0);
    return fd; // < 0 if failed
}

/**
 * Starts measuring on the given perf_event file descriptor.
 * No-op if fd < 0 (invalid).
 */
void startMeasure(int fd)
{
    // If fd < 0, there's no valid measurement to start
    if (fd < 0) return;

    _mm_mfence();

    if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1)
    {
        perror("ioctl PERF_EVENT_IOC_RESET");
        exit(EXIT_FAILURE);
    }
    if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1)
    {
        perror("ioctl PERF_EVENT_IOC_ENABLE");
        exit(EXIT_FAILURE);
    }
}

/**
 * Stops measuring on the given perf_event file descriptor, reads
 * the count, and returns the result. Returns -1 if something fails
 * or if fd < 0.
 */
long long stopMeasure(int fd)
{
    if (fd < 0) return -1;

    _mm_mfence();

    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) == -1)
    {
        perror("ioctl PERF_EVENT_IOC_DISABLE");
        exit(EXIT_FAILURE);
    }

    struct read_format {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    } rf;

    if (read(fd, &rf, sizeof(rf)) < 0)
    {
        return -1;
    }
    return static_cast<long long>(rf.value);
}

//=============================================================================
// Address Access & Generation
//=============================================================================

/**
 * Repeatedly reads from the given address to generate PMU events
 * on that memory location. Each read is separated by cache flush
 * to force actual memory access.
 *
 * @param addr       The address to access repeatedly.
 * @param numAccess  How many times to read the address.
 */
void accessAddress(uint64_t addr, size_t numAccess)
{
    volatile uint64_t *p = reinterpret_cast<volatile uint64_t*>(addr);
    for (size_t i = 0; i < numAccess; i++)
    {
        _mm_clflush((void*)p); // flush from cache
        _mm_lfence();
        (void)(*p);            // read from memory
        _mm_lfence();
    }
}

/**
 * Generates a random address offset within [base, base+size),
 * aligned to 64 bytes (cache line).
 */
uint64_t getRandomAddress(uint64_t base, uint64_t size)
{
    uint64_t part1 = static_cast<uint64_t>(rand());
    uint64_t part2 = static_cast<uint64_t>(rand());
    uint64_t offset = ((part1 << 32ULL) | part2) % size;

    const uint64_t clSize = 64ULL;
    offset = (offset / clSize) * clSize;
    return base + offset;
}

/**
 * Attempts a bit-flip approach for the next address: flipping one bit
 * in the physical address to find nearby addresses that might differ
 * in important bits. Falls back to random if invalid or crossing pages.
 */
uint64_t getNextAddress(uint64_t oldAddr, uint64_t base, uint64_t size)
{
    static const size_t MIN_SHIFT = 6; 
    static size_t shift = MIN_SHIFT;
    static uint64_t baseAddr = 0;

    if (shift == MIN_SHIFT)
    {
        baseAddr = oldAddr;
    }

    // Physical address of the old base
    uint64_t oldPhys = getPhysicalAddr(baseAddr);

    // Flip a single bit
    uint64_t candidate = baseAddr ^ (1ULL << shift);

    if(candidate >= base && candidate < (base + size))
    {
        // Force read to ensure it's valid
        volatile uint8_t *p = reinterpret_cast<uint8_t*>(candidate);
        (void)(*p);

        uint64_t phys = getPhysicalAddr(candidate);
        std::bitset<64> diff(phys ^ oldPhys);

        // If more than 1 bit changed, revert to random address
        if (diff.count() > 1)
        {
            shift = MIN_SHIFT;
            return getRandomAddress(base, size);
        }
        else
        {
            shift++;
            return candidate;
        }
    }
    else
    {
        shift = MIN_SHIFT;
        return getRandomAddress(base, size);
    }
}

//=============================================================================
// Solver & Cleanup Routines
//=============================================================================

/**
 * Calculates how many bits remain after we remove some leading/trailing
 * stuck bits from consideration.
 */
uint64_t getUsableBits(uint64_t removeFront, uint64_t removeBack)
{
    return (64ULL - removeFront - removeBack);
}

/**
 * Shifts out 'removeFront' low bits and masks out 'removeBack' high bits
 * from each stored address in the map, so only the bits that can vary remain.
 * This helps isolate the bits used for channel/bank/etc.
 */
void cleanAddresses(std::map<size_t,std::vector<size_t>> &addresses,
                    uint64_t removeFront,
                    uint64_t removeBack)
{
    std::cout << "\n=== Debug: Addresses BEFORE Cleaning ===\n";
    for (auto &kv : addresses)
    {
        std::cout << "Set " << kv.first << ": ";
        for (auto a : kv.second) std::cout << std::hex << a << " ";
        std::cout << "\n";
    }
    std::cout << "=========================================\n";

    uint64_t usableBits = getUsableBits(removeFront, removeBack);
    uint64_t mask = ((1ULL << usableBits) - 1ULL);

    for (auto &kv : addresses)
    {
        for (auto &val : kv.second)
        {
            val >>= removeFront;
            val &= mask;
        }
    }

    std::cout << "\n=== Debug: Addresses AFTER Cleaning ===\n";
    for (auto &kv : addresses)
    {
        std::cout << "Set " << kv.first << ": ";
        for (auto a : kv.second) std::cout << std::hex << a << " ";
        std::cout << "\n";
    }
    std::cout << "=========================================\n";
}

/**
 * Removes empty sets (no addresses) from the map so they won't confuse the solver.
 */
std::map<size_t,std::vector<uint64_t>>
compactSets(const std::map<size_t,std::vector<uint64_t>> &addresses)
{
    std::map<size_t,std::vector<uint64_t>> newMap;
    size_t idx = 0;
    for (auto &kv : addresses)
    {
        if (!kv.second.empty())
        {
            newMap[idx] = kv.second;
            idx++;
        }
    }
    return newMap;
}

/**
 * Prints the solution for a single bit from the solver.
 * If the solver found an exact solution, we list the involved, uninvolved,
 * and unknown bits. If no exact solution is found, we now print *all* bits
 * with their percentage. (Previously, it only printed bits above a threshold.)
 */
void printSolution(const Solver::Solution &s, size_t offset)
{
    if (s.exists)
    {
        // A perfect linear solution was found for this bit
        std::cout << "Involved bits:   ";
        for (auto b : s.involvedBits)
            std::cout << (offset + b) << " ";
        std::cout << "\n";

        std::cout << "Uninvolved bits: ";
        for (auto b : s.uninvolvedBits)
            std::cout << (offset + b) << " ";
        std::cout << "\n";

        std::cout << "Unknown bits:    ";
        for (auto b : s.unknownBits)
            std::cout << (offset + b) << " ";
        std::cout << "\n";
    }
    else
    {
        // We have no exact solution, so show partial "confidence" info for ALL bits
        std::cout << "(No exact solution found.)\n";
        std::map<size_t, size_t> bitFrequency;

        // Collect frequency counts for all unknown/involved bits
        for (auto b : s.unknownBits)
        {
            bitFrequency[b] = 0;
        }
        for (auto b : s.involvedBits)
        {
            bitFrequency[b]++;
        }

        // Find the maximum vote count
        size_t maxVotes = 0;
        for (auto &bf : bitFrequency)
        {
            if (bf.second > maxVotes) 
                maxVotes = bf.second;
        }

        // Print each bit with its percentage probability
        std::cout << "Bit percentages:\n";
        for (auto &bf : bitFrequency)
        {
            double c = (maxVotes > 0) ? (100.0 * bf.second / (double)maxVotes) : 0.0;
            std::cout << "  Bit " << (offset + bf.first) << ": " << c << "%\n";
        }
        std::cout << "\n";
    }
}

/**
 * Prints the solutions for all bits in a solver run.
 */
void printSolutions(const std::vector<Solver::Solution> &solList,
                    size_t offset)
{
    for (size_t i = 0; i < solList.size(); i++)
    {
        std::cout << "Solver bit " << i << " =>\n";
        printSolution(solList[i], offset);
        std::cout << "\n";
    }
}

/**
 * Builds the solver matrix and attempts to solve for each addressing bit.
 * We estimate the number of bits used for a function from the log2 of
 * the number of distinct sets.
 */
std::vector<Solver::Solution>
calculateAddressingFunction(const std::map<size_t,std::vector<uint64_t>> &addresses,
                            size_t addrFuncBits,
                            size_t usableBits)
{
    std::vector<Solver::Solution> allSolutions;

    for (size_t bit = 0; bit < addrFuncBits; bit++)
    {
        std::vector<uint64_t> matrix;
        Solver solver;
        uint64_t mask = (1ULL << bit);

        // Build a matrix row for each address, along with the bit's value
        for (auto &kv : addresses)
        {
            uint64_t bitValue = (kv.first & mask) >> bit;
            for (auto row : kv.second)
            {
                uint64_t rowWithResult = (row << 1ULL) | bitValue;
                matrix.push_back(rowWithResult);
            }
        }

        // We need at least a few addresses to attempt a solution
        if (matrix.size() < 5)
        {
            std::cerr << "WARNING: Not enough data to solve for bit "
                      << bit << " => only " << matrix.size() << " entries.\n";
            allSolutions.push_back(Solver::Solution());
            continue;
        }

        // Attempt to solve
        solver.solve(matrix, usableBits);
        Solver::Solution s = solver.getSolution(matrix);
        allSolutions.push_back(s);
    }

    return allSolutions;
}

/**
 * Helper function that:
 *  1) "Cleans" addresses to remove stuck bits.
 *  2) Compacts them.
 *  3) Builds and solves a matrix for each bit in the addressing function.
 *  4) Prints the solutions.
 */
void prepareSolvePrint(std::map<size_t,std::vector<uint64_t>> sets,
                       size_t removeFront,
                       size_t removeBack)
{
    // Convert from uint64_t to size_t for the cleaning function
    std::map<size_t,std::vector<size_t>> tmp;
    for (auto &kv : sets)
    {
        std::vector<size_t> tv;
        for (auto x : kv.second)
            tv.push_back(static_cast<size_t>(x));
        tmp[kv.first] = std::move(tv);
    }

    // Perform the cleaning (strip out stuck bits)
    cleanAddresses(tmp, removeFront, removeBack);

    // Convert back to 64-bit
    std::map<size_t,std::vector<uint64_t>> cleaned;
    for (auto &kv : tmp)
    {
        std::vector<uint64_t> cv;
        for (auto x : kv.second)
            cv.push_back(static_cast<uint64_t>(x));
        cleaned[kv.first] = std::move(cv);
    }

    // Remove empty sets
    cleaned = compactSets(cleaned);
    if (cleaned.size() <= 1)
    {
        std::cerr << "WARNING: Not enough distinct sets to solve. Skipping.\n";
        return;
    }

    // Estimate how many bits are used based on the count of sets
    size_t expectedBits = static_cast<size_t>(ceil(log2(cleaned.size())));

    // Solve for these bits
    auto solutions = calculateAddressingFunction(cleaned,
                                                 expectedBits,
                                                 getUsableBits(removeFront, removeBack));

    // Print some debug lines
    std::cout << "\n=== Debug: First few addresses after cleaning ===\n";
    size_t ccount = 0;
    for (auto &kv : cleaned)
    {
        std::cout << "Set " << kv.first << ": ";
        for (auto &addr : kv.second)
        {
            std::cout << std::hex << addr << " ";
            if (++ccount >= 10) break;
        }
        std::cout << "\n";
        if (ccount >= 10) break;
    }
    std::cout << "===============================================\n";

    // Print the solver solutions
    printSolutions(solutions, removeFront);
}

//=============================================================================
// Main function
//=============================================================================
int main(int argc, char *argv[])
{
    // Command-line arguments/flags
    bool verbose           = false;
    bool considerTad       = false;
    unsigned int sizeGb    = 20;    // default memory region = 20 GB
    size_t numAddressTotal = 5000;  // number of successful addresses
    size_t numAccess       = 4000;  // times we access each address
    double confThreshold   = 50.0;  // previously used for threshold; still accepted
    unsigned int pmuType   = 0xd;   // default PMU type is 0xd

    // Parse arguments. New flag: '-c' to override PMU type.
    int opt;
    while ((opt = getopt(argc, argv, "vrs:n:a:t:c:")) != -1)
    {
        switch (opt)
        {
            case 'v':
                verbose = true;
                break;
            case 'r':
                considerTad = true;
                break;
            case 's':
                sizeGb = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'n':
                numAddressTotal = static_cast<size_t>(atoi(optarg));
                break;
            case 'a':
                numAccess = static_cast<size_t>(atoi(optarg));
                break;
            case 't':
                confThreshold = atof(optarg);
                break;
            case 'c':
                // User can specify the PMU type in hex or decimal
                pmuType = strtoul(optarg, nullptr, 0);
                break;
            default:
                std::cerr 
                    << "Usage: " << argv[0] << "\n"
                    << "  -v  (verbose)\n"
                    << "  -r  (consider TAD regions)\n"
                    << "  -s <size GB>\n"
                    << "  -n <#successful addresses to collect>\n"
                    << "  -a <#accesses per test>\n"
                    << "  -t <confidence threshold %> (deprecated usage)\n"
                    << "  -c <PMU type> (default 0xd)\n";
                return EXIT_FAILURE;
        }
    }

    // Identify current CPU / NUMA node
    unsigned int nodeid=0, cpuid=0;
    if (syscall(SYS_getcpu, &cpuid, &nodeid, nullptr) == -1)
    {
        std::cout << "Cannot determine node id\n";
        return EXIT_FAILURE;
    }
    if (verbose)
        std::cout << "Running on socket (NUMA node) " << nodeid << std::endl;

    // Open pagemap for reading (virtual -> physical mapping)
    initPagemap();

    // Allocate memory
    uint64_t totalBytes = (uint64_t)sizeGb * 1024ULL * 1024ULL * 1024ULL;
    void* space = allocate(totalBytes);
    if (!space)
    {
        std::cerr << "Allocation failed.\n";
        return EXIT_FAILURE;
    }
    uint64_t spaceBase = reinterpret_cast<uint64_t>(space);

    // Seed the random generator
    srand(3344);

    // We store addresses in separate sets for each channel/rank/bank/bankGroup
    std::map<size_t,std::vector<uint64_t>> channelAddrs;
    std::map<size_t,std::vector<uint64_t>> rankAddrs;
    std::map<size_t,std::vector<uint64_t>> bankAddrs;
    std::map<size_t,std::vector<uint64_t>> bankGroupAddrs;

    int successfulMatches = 0;
    int failedMatches     = 0;

    // We want numAddressTotal successful addresses with PMU counts > pmuThreshold.
    std::cout << "Collecting " << numAddressTotal 
              << " successful address samples...\n";

    uint64_t nextVA = spaceBase;

    // Keep looping until we collect enough successful addresses
    while (successfulMatches < (int)numAddressTotal)
    {
        // Pick the next address (bit-flip or random)
        nextVA = getNextAddress(nextVA, spaceBase, totalBytes);
        uint64_t phys = getPhysicalAddr(nextVA);

        static const unsigned int maxChannels = 4;
        static const unsigned int maxRanks    = 8;
        static const unsigned int maxBanks    = 16;

        long long results[512];
        memset(results, 0, sizeof(results));

        // Measure memory events for each ch/rank/bank
        for (unsigned int ch = 0; ch < maxChannels; ch++)
        {
            for (unsigned int rk = 0; rk < maxRanks; rk++)
            {
                for (unsigned int bk = 0; bk < maxBanks; bk++)
                {
                    int fd = setupMeasure(cpuid, ch, rk, bk, pmuType);
                    if (fd < 0)
                        continue; // skip if we can't open the perf event

                    startMeasure(fd);
                    accessAddress(nextVA, numAccess);
                    long long count = stopMeasure(fd);
                    close(fd);

                    int idx = (ch << 7) | (rk << 4) | bk;
                    results[idx] = count;
                }
            }
        }

        // Find the best match channel/rank/bank
        long long maxVal = 0;
        int maxIdx = -1;
        for (int i = 0; i < 512; i++)
        {
            if (results[i] > maxVal)
            {
                maxVal = results[i];
                maxIdx = i;
            }
        }

        // If below threshold, discard this address
        if (maxIdx < 0 || maxVal < pmuThreshold)
        {
            failedMatches++;
            if (verbose)
            {
                std::cout << "Discarding [0x" << std::hex << phys << std::dec
                          << "] maxVal=" << maxVal 
                          << " < " << pmuThreshold << "\n";
            }
            continue;
        }

        // Decode the best channel/rank/bank from maxIdx
        int bestCh   = (maxIdx >> 7) & 0x3;
        int bestRank = (maxIdx >> 4) & 0x7;
        int bestBank = (maxIdx     ) & 0xF;

        // Store this address in the appropriate sets
        channelAddrs[bestCh].push_back(phys);
        rankAddrs[bestRank].push_back(phys);
        bankAddrs[bestBank].push_back(phys);

        int bankGroup = bestBank / 4; 
        bankGroupAddrs[bankGroup].push_back(phys);

        successfulMatches++;
        if (verbose)
        {
            std::cout << "[0x" << std::hex << phys << std::dec
                      << "] CH=" << bestCh
                      << " RANK=" << bestRank
                      << " BANK=" << bestBank
                      << " => MaxVal=" << maxVal 
                      << " (Success #" << successfulMatches << ")\n";
        }
    }

    std::cout << "failedMatches=" << failedMatches
              << ", successfulMatches=" << successfulMatches << "\n\n";

    // Print how many addresses ended up in each set
    for (size_t c = 0; c < 4; c++)
        std::cout << "Channel " << c << " => " << channelAddrs[c].size() << " addresses\n";
    for (size_t r = 0; r < 8; r++)
        std::cout << "Rank " << r << " => " << rankAddrs[r].size() << " addresses\n";
    for (size_t b = 0; b < 16; b++)
        std::cout << "Bank " << b << " => " << bankAddrs[b].size() << " addresses\n";
    for (size_t g = 0; g < 4; g++)
        std::cout << "BankGroup " << g << " => " << bankGroupAddrs[g].size() << " addresses\n";

    // Gather all successful physical addresses into a single set
    std::set<uint64_t> allUsedPhys;
    for (auto &kv : channelAddrs)
    {
        for (auto p : kv.second)
        {
            allUsedPhys.insert(p);
        }
    }

    // Compute the AND & OR of all addresses to detect stuck bits
    uint64_t andAll = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t orAll  = 0ULL;
    for (auto p : allUsedPhys)
    {
        andAll &= p;
        orAll  |= p;
    }

    if (verbose)
    {
        std::cout << "Physical addresses used: " << allUsedPhys.size() << "\n";
        std::cout << "AND of addresses = 0x" << std::hex << andAll << std::dec << "\n";
        std::cout << " OR of addresses = 0x" << std::hex << orAll  << std::dec << "\n";
    }

    // Determine which bits never vary (stuck bits)
    std::bitset<64> andAllBits(andAll);
    std::bitset<64> orAllBits(orAll);
    std::bitset<64> unknownBits(0ULL);
    for (size_t i = 0; i < 64; i++)
    {
        // If bit i never becomes 1 or never becomes 0, it's stuck.
        if (!orAllBits.test(i) || andAllBits.test(i))
        {
            unknownBits.set(i);
        }
    }

    // removeFront => first non-stuck bit from the bottom
    uint64_t removeFront = 0;
    for (size_t i = 0; i < 64; i++)
    {
        if (!unknownBits.test(i))
        {
            removeFront = i;
            break;
        }
    }

    // removeBack => first non-stuck bit from the top
    uint64_t removeBack = 0;
    for (int i = 63; i >= 0; i--)
    {
        if (!unknownBits.test(size_t(i)))
        {
            removeBack = 63 - i;
            break;
        }
    }

    if (verbose)
    {
        std::cout << "removeFront=" << removeFront
                  << ", removeBack=" << removeBack
                  << ", confThreshold=" << confThreshold << "%\n";
    }

    // If we consider TAD (address hashing) logic, place code here...
    if (considerTad)
    {
        // Example placeholder for TAD region logic
        // ...
    }
    else
    {
        // Solve for channel bits
        std::cout << "\n=== Solving Channel Bits ===\n";
        prepareSolvePrint(channelAddrs, removeFront, removeBack);

        // Solve for rank bits
        std::cout << "\n=== Solving Rank Bits ===\n";
        prepareSolvePrint(rankAddrs, removeFront, removeBack);

        // Solve for bank bits
        std::cout << "\n=== Solving Bank Bits ===\n";
        prepareSolvePrint(bankAddrs, removeFront, removeBack);

        // Solve for bank group bits
        std::cout << "\n=== Solving Bank Group Bits ===\n";
        prepareSolvePrint(bankGroupAddrs, removeFront, removeBack);
    }

    return 0;
}
