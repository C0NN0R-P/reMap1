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

#include "solver.h"
#include "sysinfo.h"

using namespace std;

#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)
#define PAGE_SHIFT      12
#define PAGEMAP_LENGTH  8

static int g_pagemap_fd = -1;

// You can tweak this threshold to discard low PMU readings.
static const long long pmuThreshold = 3000;

// Data structure for sets of addresses
typedef std::map<size_t, std::vector<uint64_t>> AddressSet;

///////////////////////////////////////////////////////
//             ALLOCATION HELPERS
///////////////////////////////////////////////////////
void* tryAllocate1Gb(uint64_t size)
{
    void* space = mmap(nullptr, size,
                       PROT_READ | PROT_WRITE,
                       MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE |
                       MAP_HUGETLB | MAP_HUGE_1GB,
                       -1, 0);
    return space;
}

void* tryAllocate2Mb(uint64_t size)
{
    void* space = mmap(nullptr, size,
                       PROT_READ | PROT_WRITE,
                       MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE |
                       MAP_HUGETLB | MAP_HUGE_2MB,
                       -1, 0);
    return space;
}

void* tryAllocate4Kb(uint64_t size)
{
    void* space = mmap(nullptr, size,
                       PROT_READ | PROT_WRITE,
                       MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE,
                       -1, 0);
    return space;
}

void* allocate(uint64_t size)
{
    uint64_t sizeGb = size / (1024ULL * 1024ULL * 1024ULL);

    // Attempt 1GB pages
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

    // Attempt 2MB pages
    space = tryAllocate2Mb(size);
    if (space != MAP_FAILED)
    {
        if (mlock(space, size) == 0)
        {
            std::cout << "Allocated " << sizeGb << "GB using 2MB pages\n";
            return space;
        }
        munmap(space, size);
    }

    // Attempt 4KB pages
    space = tryAllocate4Kb(size);
    if (space != MAP_FAILED)
    {
        if (mlock(space, size) == 0)
        {
            std::cout << "Allocated " << sizeGb << "GB using 4KB pages\n";
            return space;
        }
        munmap(space, size);
    }

    std::cout << "Failed to allocate " << sizeGb << "GB\n";
    return nullptr;
}

///////////////////////////////////////////////////////
//             PAGEMAP & ADDRESSING
///////////////////////////////////////////////////////
uint64_t frameNumberFromPagemap(uint64_t value)
{
    return value & ((1ULL << 54) - 1);
}

void initPagemap()
{
    g_pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    assert(g_pagemap_fd >= 0);
}

uint64_t getPhysicalAddr(uint64_t virtualAddr)
{
    uint64_t value;
    off_t offset = (virtualAddr / 4096ULL) * sizeof(value);
    int got = pread(g_pagemap_fd, &value, sizeof(value), offset);
    assert(got == 8);

    // Check "present" bit:
    assert(value & (1ULL << 63));
    uint64_t frame_num = frameNumberFromPagemap(value);
    return (frame_num * 4096ULL) | (virtualAddr & 0xFFFULL);
}

///////////////////////////////////////////////////////
//             PERF EVENT (PMU) HELPERS
///////////////////////////////////////////////////////
int setupMeasure(int cpuid, unsigned int channel, unsigned int rank, unsigned int bank)
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));

    auto imcs = SysInfo::getImcs();
    if (imcs.empty())
    {
        std::cerr << "No memory controller PMU found\n";
        exit(EXIT_FAILURE);
    }

    pe.type = 0xd; // IMC PMU type
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

    int fd = syscall(__NR_perf_event_open, &pe, -1, cpuid, -1, 0);
    // If fd == -1, we skip it later
    return fd;
}

void startMeasure(int fd)
{
    if (fd == -1) return;
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

long long stopMeasure(int fd)
{
    if (fd == -1) return -1;
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
        // Not fatal, but no measurement
        return -1;
    }
    return static_cast<long long>(rf.value);
}

///////////////////////////////////////////////////////
//       ADDRESS ACCESS / GENERATION
///////////////////////////////////////////////////////
void accessAddress(uint64_t addr, size_t numAccess)
{
    volatile uint64_t *p = reinterpret_cast<volatile uint64_t*>(addr);
    for (size_t i = 0; i < numAccess; i++)
    {
        _mm_clflush((void*)p);
        _mm_lfence();
        (void)(*p);  // read
        _mm_lfence();
    }
}

uint64_t getRandomAddress(uint64_t base, uint64_t size)
{
    uint64_t part1 = static_cast<uint64_t>(rand());
    uint64_t part2 = static_cast<uint64_t>(rand());
    uint64_t offset = ((part1 << 32ULL) | part2) % size;

    // 64‑byte alignment
    const uint64_t clSize = 64ULL;
    offset = (offset / clSize) * clSize;
    return base + offset;
}

uint64_t getNextAddress(uint64_t oldAddr, uint64_t base, uint64_t size)
{
    // “Bit flipping” approach
    static const size_t MIN_SHIFT = 6;
    static size_t shift = MIN_SHIFT;
    static uint64_t baseAddr = 0;

    if (shift == MIN_SHIFT)
    {
        baseAddr = oldAddr;
    }
    uint64_t oldPhys = getPhysicalAddr(baseAddr);
    uint64_t candidate = baseAddr ^ (1ULL << shift);

    if(candidate >= base && candidate < (base + size))
    {
        // Validate physically
        volatile uint8_t *p = reinterpret_cast<uint8_t*>(candidate);
        (void)(*p);
        uint64_t phys = getPhysicalAddr(candidate);

        std::bitset<64> diff(phys ^ oldPhys);
        if (diff.count() > 1)
        {
            // Jumped frames
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

///////////////////////////////////////////////////////
//             SOLVER & CLEANUP
///////////////////////////////////////////////////////
uint64_t getUsableBits(uint64_t removeFront, uint64_t removeBack)
{
    return (64ULL - removeFront - removeBack);
}

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

std::map<size_t,std::vector<uint64_t>>
compactSets(const std::map<size_t,std::vector<uint64_t>> &addresses)
{
    // Remove any sets that have zero addresses
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

///////////////////////////////////////////////////////
//    Print Solutions w/ user-set confidence
///////////////////////////////////////////////////////
void printSolution(const Solver::Solution &s, size_t offset, double userConfidence)
{
    if (s.exists)
    {
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
        // We didn't find an exact solution. Show partial info:
        std::cout << "(No exact solution found.)\n";

        // Track how often bits appear in involvedBits:
        std::map<size_t, size_t> bitFrequency;
        for (auto b : s.unknownBits)
        {
            bitFrequency[b] = 0;
        }
        for (auto b : s.involvedBits)
        {
            bitFrequency[b]++;
        }

        // Find max votes
        size_t maxVotes = 0;
        for (auto &bf : bitFrequency)
        {
            if (bf.second > maxVotes) maxVotes = bf.second;
        }

        // Print bits that meet or exceed userConfidence threshold
        double threshold = userConfidence; 
        std::cout << "Likely bits (>=" << threshold << "% confidence): ";
        for (auto &bf : bitFrequency)
        {
            double c = (maxVotes > 0) ? 100.0*(bf.second)/(double)maxVotes : 0.0;
            if (c >= threshold)
            {
                std::cout << (offset + bf.first) << " (" << c << "%) ";
            }
        }
        std::cout << "\n";
    }
}

void printSolutions(const std::vector<Solver::Solution> &solList, size_t offset, double userConfidence)
{
    for (size_t i = 0; i < solList.size(); i++)
    {
        std::cout << "Solver bit " << i << " =>\n";
        printSolution(solList[i], offset, userConfidence);
        std::cout << "\n";
    }
}

///////////////////////////////////////////////////////
//   CALCULATE + SOLVE
///////////////////////////////////////////////////////
std::vector<Solver::Solution>
calculateAddressingFunction(const std::map<size_t,std::vector<uint64_t>> &addresses,
                            size_t addrFuncBits,
                            size_t usableBits)
{
    std::vector<Solver::Solution> allSolutions;
    for (size_t bit = 0; bit < addrFuncBits; bit++)
    {
        // Build the solver matrix
        std::vector<uint64_t> matrix;
        Solver solver;
        uint64_t mask = (1ULL << bit);

        for (auto &kv : addresses)
        {
            uint64_t bitValue = (kv.first & mask) >> bit;
            for (auto row : kv.second)
            {
                uint64_t rowWithResult = (row << 1ULL) | bitValue;
                matrix.push_back(rowWithResult);
            }
        }

        if (matrix.size() < 5)
        {
            // Arbitrary check to see if there's enough data
            std::cerr << "WARNING: Not enough data to solve for bit " << bit
                      << " => only " << matrix.size() << " entries.\n";
            // We'll skip solving or just store an empty solution
            allSolutions.push_back(Solver::Solution());
            continue;
        }

        solver.solve(matrix, usableBits);
        Solver::Solution s = solver.getSolution(matrix);
        allSolutions.push_back(s);
    }
    return allSolutions;
}

///////////////////////////////////////////////////////
//   WRAPPER: PREPARE, SOLVE, PRINT
///////////////////////////////////////////////////////
void prepareSolvePrint(std::map<size_t,std::vector<uint64_t>> sets,
                       size_t removeFront,
                       size_t removeBack,
                       double userConfidence)
{
    // Convert to size_t for cleaning
    std::map<size_t,std::vector<size_t>> tmp;
    for (auto &kv : sets)
    {
        std::vector<size_t> tv;
        for (auto x : kv.second) tv.push_back(static_cast<size_t>(x));
        tmp[kv.first] = std::move(tv);
    }
    cleanAddresses(tmp, removeFront, removeBack);

    // Convert back to uint64_t after cleaning
    std::map<size_t,std::vector<uint64_t>> cleaned;
    for (auto &kv : tmp)
    {
        std::vector<uint64_t> cv;
        for (auto x : kv.second) cv.push_back(static_cast<uint64_t>(x));
        cleaned[kv.first] = std::move(cv);
    }

    // Remove empty sets
    cleaned = compactSets(cleaned);
    if (cleaned.size() <= 1)
    {
        std::cerr << "WARNING: Not enough distinct sets to solve. Skipping.\n";
        return;
    }

    // Expect log2(#sets) bits in the addressing function
    size_t expectedBits = static_cast<size_t>(ceil(log2(cleaned.size())));

    // Solve
    auto solutions = calculateAddressingFunction(cleaned,
                                                 expectedBits,
                                                 getUsableBits(removeFront, removeBack));

    // Print debug
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

    // Print solutions
    printSolutions(solutions, removeFront, userConfidence);
}

///////////////////////////////////////////////////////
//               MAIN FUNCTION
///////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    bool verbose          = false;
    bool considerTad      = false;
    unsigned int sizeGb   = 20;
    size_t numAddressTotal = 5000;
    size_t numAccess      = 4000;

    // Default to 50% confidence instead of 95%
    double confThreshold  = 50.0; 

    int opt;
    while ((opt = getopt(argc, argv, "vrs:n:a:t:")) != -1)
    {
        switch (opt)
        {
            case 'v': verbose = true; break;
            case 'r': considerTad = true; break;
            case 's': sizeGb = static_cast<unsigned int>(atoi(optarg)); break;
            case 'n': numAddressTotal = static_cast<size_t>(atoi(optarg)); break;
            case 'a': numAccess = static_cast<size_t>(atoi(optarg)); break;
            case 't': confThreshold = atof(optarg); break;
            default:
                std::cerr << "Usage: " << argv[0] << "\n"
                          << "  -v  (verbose)\n"
                          << "  -r  (consider TAD regions)\n"
                          << "  -s <size GB>\n"
                          << "  -n <#addresses to collect>\n"
                          << "  -a <#accesses per test>\n"
                          << "  -t <confidence threshold %> (default 50.0)\n";
                return EXIT_FAILURE;
        }
    }

    // Get CPU / Node
    unsigned int nodeid=0, cpuid=0;
    if (syscall(SYS_getcpu, &cpuid, &nodeid, nullptr) == -1)
    {
        std::cout << "Cannot determine node id\n";
        return EXIT_FAILURE;
    }
    if (verbose)
        std::cout << "Running on socket (NUMA node) " << nodeid << std::endl;

    // Pagemap
    initPagemap();

    // Allocate Memory
    uint64_t totalBytes = (uint64_t)sizeGb * 1024ULL * 1024ULL * 1024ULL;
    void* space = allocate(totalBytes);
    if (!space)
    {
        std::cerr << "Allocation failed.\n";
        return EXIT_FAILURE;
    }
    uint64_t spaceBase = reinterpret_cast<uint64_t>(space);

    srand(3344);

    // Data structures
    std::set<uint64_t> usedPhysicalAddrs;
    std::map<size_t,std::vector<uint64_t>> channelAddrs;
    std::map<size_t,std::vector<uint64_t>> rankAddrs;
    std::map<size_t,std::vector<uint64_t>> bankAddrs;
    std::map<size_t,std::vector<uint64_t>> bankGroupAddrs;

    int successfulMatches = 0;
    int failedMatches     = 0;

    // Collect addresses
    uint64_t nextVA = spaceBase;
    std::cout << "Collecting address samples...\n";
    while (usedPhysicalAddrs.size() < numAddressTotal)
    {
        nextVA = getNextAddress(nextVA, spaceBase, totalBytes);
        uint64_t phys = getPhysicalAddr(nextVA);

        // Skip duplicates
        if (usedPhysicalAddrs.count(phys)) 
            continue;
        usedPhysicalAddrs.insert(phys);

        // For each channel/rank/bank, measure performance counters
        static const unsigned int maxChannels = 4;
        static const unsigned int maxRanks    = 8;
        static const unsigned int maxBanks    = 16;

        long long results[512];
        memset(results, 0, sizeof(results));

        for (unsigned int ch = 0; ch < maxChannels; ch++)
        {
            for (unsigned int rk = 0; rk < maxRanks; rk++)
            {
                for (unsigned int bk = 0; bk < maxBanks; bk++)
                {
                    int fd = setupMeasure(cpuid, ch, rk, bk);
                    if (fd < 0) continue; // skip if no FD
                    startMeasure(fd);
                    accessAddress(nextVA, numAccess);
                    long long count = stopMeasure(fd);
                    close(fd);

                    int idx = (ch << 7) | (rk << 4) | bk;
                    results[idx] = count;
                }
            }
        }

        // Find the channel/rank/bank with the highest count
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

        // Discard if maxVal < pmuThreshold (e.g. 3000)
        if (maxIdx < 0 || maxVal < pmuThreshold)
        {
            if (verbose)
            {
                std::cout << "Discarding [0x" << std::hex << phys << std::dec
                          << "] because maxVal=" << maxVal
                          << " < " << pmuThreshold << "\n";
            }
            failedMatches++;
            continue;
        }

        // decode
        int bestCh   = (maxIdx >> 7) & 0x3;
        int bestRank = (maxIdx >> 4) & 0x7;
        int bestBank = (maxIdx     ) & 0xF;

        // Add to sets
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
                      << " => MaxVal=" << maxVal << "\n";
        }
    }

    // Print stats
    for (size_t c = 0; c < 4; c++)
        std::cout << "Channel " << c << " => " << channelAddrs[c].size() << " addresses\n";
    for (size_t r = 0; r < 8; r++)
        std::cout << "Rank " << r << " => " << rankAddrs[r].size() << " addresses\n";
    for (size_t b = 0; b < 16; b++)
        std::cout << "Bank " << b << " => " << bankAddrs[b].size() << " addresses\n";
    for (size_t g = 0; g < 4; g++)
        std::cout << "BankGroup " << g << " => " << bankGroupAddrs[g].size() << " addresses\n";
    std::cout << "failedMatches=" << failedMatches
              << ", successfulMatches=" << successfulMatches << "\n";

    // Identify bits that never vary or always vary
    uint64_t andAll = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t orAll  = 0ULL;
    for (auto p : usedPhysicalAddrs)
    {
        andAll &= p;
        orAll  |= p;
    }

    std::bitset<64> andAllBits(andAll);
    std::bitset<64> orAllBits(orAll);

    if (verbose)
    {
        std::cout << "AND of all addrs: " << andAllBits << "\n";
        std::cout << " OR of all addrs: " << orAllBits  << "\n";
    }

    // Bits that are "stuck" => unknown
    std::bitset<64> unknownBits(0ULL);
    for (size_t i = 0; i < 64; i++)
    {
        // If orAllBits[i] == 0 => all addresses have bit=0
        // If andAllBits[i] == 1 => all addresses have bit=1
        // => "stuck" bit
        if (orAllBits.test(i) == false || andAllBits.test(i) == true)
            unknownBits.set(i);
    }

    // removeFront => first free (not stuck) bit
    uint64_t removeFront = 0;
    for (size_t i = 0; i < 64; i++)
    {
        if (!unknownBits.test(i))
        {
            removeFront = i;
            break;
        }
    }

    // removeBack => last free (not stuck) bit
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

    // Possibly handle TAD
    if (considerTad)
    {
        // e.g. do your TAD region logic here
    }
    else
    {
        // Solve for channel bits
        std::cout << "\n=== Solving Channel Bits ===\n";
        prepareSolvePrint(channelAddrs, removeFront, removeBack, confThreshold);

        // Solve for rank bits
        std::cout << "\n=== Solving Rank Bits ===\n";
        prepareSolvePrint(rankAddrs, removeFront, removeBack, confThreshold);

        // Solve for bank bits
        std::cout << "\n=== Solving Bank Bits ===\n";
        prepareSolvePrint(bankAddrs, removeFront, removeBack, confThreshold);

        // Solve for bank group bits
        std::cout << "\n=== Solving Bank Group Bits ===\n";
        prepareSolvePrint(bankGroupAddrs, removeFront, removeBack, confThreshold);
    }

    return 0;
}
