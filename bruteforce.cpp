#include <iostream>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <bitset>
#include <iostream>
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
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <random>
#include <set>
#include <sys/syscall.h>
#include "solver.h"
#include "sysinfo.h"


using namespace std;


#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)
#define PAGE_SHIFT 12
#define PAGEMAP_LENGTH 8

static int g_pagemap_fd = -1;

typedef std::map<size_t,std::vector<uint64_t>> AddressSet;


void* tryAllocate1Gb(uint64_t size)
{
    auto space = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                      MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB
                      , -1, 0);
    return space;
}

void* tryAllocate2Mb(uint64_t size)
{
    auto space = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                      MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_2MB
                      , -1, 0);
    return space;
}

void* tryAllocate4Kb(uint64_t size)
{
    auto space = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                      MAP_POPULATE | MAP_ANONYMOUS | MAP_PRIVATE
                      , -1, 0);
    return space;
}


void* allocate(uint64_t size)
{
    auto sizeGb = size / (1024*1024*1024ULL);
    auto space = tryAllocate1Gb(size);
    auto l = mlock(space,size);
    if(space != (void*) -1 && l == 0)
    {
        std::cout << "Allocated " << sizeGb << "GB using 1GB pages" << endl;
        return space;
    }
    space = tryAllocate2Mb(size);
    l = mlock(space,size);
    if(space != (void*) -1 && l == 0)
    {
        std::cout << "Allocated " << sizeGb << "GB using 2MB pages" << endl;
        return space;
    }
    space = tryAllocate4Kb(size);
    l = mlock(space,size);
    if(space != (void*) -1 && l == 0)
    {
        std::cout << "Allocated " << sizeGb << "GB using 4KB pages" << endl;
        return space;
    }
    std::cout << "Failed to allocate " << sizeGb << "GB" << endl;
    return nullptr;
}

std::vector<uint64_t> getUsedSets(AddressSet addressSet)
{
    std::vector<uint64_t> usedSets;
    for(auto list : addressSet)
    {
        if(!list.second.empty())
        {
            usedSets.push_back(list.first);
        }
    }
    return usedSets;
}

uint64_t frameNumberFromPagemap(uint64_t value) {
    return value & ((1ULL << 54) - 1);
}

void initPagemap() {
    g_pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    assert(g_pagemap_fd >= 0);
}

uint64_t getPhysicalAddr(uint64_t virtualAddr) {
    uint64_t value;
    off_t offset = (virtualAddr / 4096) * sizeof(value);
    int got = pread(g_pagemap_fd, &value, sizeof(value), offset);
    assert(got == 8);
    assert(value & (1ULL << 63)); // Check the page present flag
    uint64_t frame_num = frameNumberFromPagemap(value);
    return (frame_num * 4096) | (virtualAddr & (4095));
}

void access(uint64_t addr,size_t numAccess)
{
    volatile uint64_t *p = (volatile uint64_t *) addr;
    for (unsigned int i = 0; i < numAccess; i++)
    {
        _mm_clflush((void*)p);
        _mm_lfence();
        *p;
        _mm_lfence();
    }
}

int setupMeasure(int cpuid, unsigned int channel, unsigned int rank, unsigned int bank, bool bankGroup=false)
{
    static int fd = -1;
    struct perf_event_attr pe;

    memset(&pe, 0, sizeof(struct perf_event_attr));

    auto imcs = SysInfo::getImcs();
    if (imcs.size() == 0)
    {
        std::cout << "No memory controller PMU found" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto imc = imcs[channel];

    pe.type = 0xd;  // Ensure IMC PMU is correct
    pe.size = sizeof(struct perf_event_attr);  // Ensure correct size
    unsigned int bankBits = 0b00010000;
    if(bankGroup  == true)
    {
        bankBits = 0b00010001 + bank;
    }
    else
    {
        bankBits = bank;
    }
    bankBits = bankBits << 8;
    auto rankBits = 0xb0 + rank;
    auto bits = bankBits | rankBits;
    pe.config = bits;
    pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    pe.sample_type = PERF_SAMPLE_IDENTIFIER; 
    pe.disabled = 1;
    pe.exclude_kernel = 0;
    pe.exclude_hv = 0;
    pe.precise_ip = 0;

    if (fd == -1) {
        fd = syscall(__NR_perf_event_open, &pe, -1, cpuid, -1, 0);
        if (fd == -1) {
            std::cerr << "perf_event_open failed: " << strerror(errno) << " (errno: " << errno << ")" << std::endl;
            std::cout << "Setup of performance counters failed" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    return fd;
}

void startMeasure(int fd, long long &initialCount)
{
    if (fd == -1) {
        std::cout << "ERROR: Invalid file descriptor (fd = -1), skipping measurement" << std::endl;
        return;
    }

    _mm_mfence();  // Ensure memory operations are completed before measurement
    if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1) {
        perror("ioctl PERF_EVENT_IOC_RESET failed");
        exit(EXIT_FAILURE);
    }
    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) == -1) {  // Disable counter before reading
        perror("ioctl PERF_EVENT_IOC_DISABLE failed");
        exit(EXIT_FAILURE);
    }

    // Read and store the initial count
    struct read_format {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    };

    read_format rf;
    read(fd, &rf, sizeof(rf));
    if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {  // Enable counter for measurement
        perror("ioctl PERF_EVENT_IOC_ENABLE failed");
        exit(EXIT_FAILURE);
    }
}

long long stopMeasure(int fd, long long initialCount1)
{
    if (fd == -1) {
        std::cout << "ERROR: Invalid file descriptor (fd = -1), skipping measurement" << std::endl;
        return -1;
    }

    _mm_mfence();  // Memory fence to ensure correct ordering
    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) == -1) {  // Stop counter
        perror("ioctl PERF_EVENT_IOC_DISABLE failed");
        exit(EXIT_FAILURE);
    }
    struct read_format {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    };

    read_format rf;
    read(fd, &rf, sizeof(rf));
    long long finalCount1 = rf.value;
    long long eventDifference = finalCount1 - initialCount1; // Compute the actual difference

    return eventDifference;  // Return the measured difference
}


uint64_t getRandomAddress(uint64_t base, uint64_t size)
{
    size_t part1 = static_cast<size_t>(rand());
    size_t part2 = static_cast<size_t>(rand());
    size_t offset = ((part1 << 32ULL) | part2) % size;
    auto clSize = 64ULL;
    offset = (offset / clSize) * clSize;
    return  base + offset;
}

uint64_t getNextAddress(uint64_t oldAddr, uint64_t base, uint64_t size)
{
    static const size_t MIN_SHIFT = 6;
    static size_t shift = MIN_SHIFT;
    static uint64_t baseAddr = 0;
    if(shift == MIN_SHIFT)
    {
        baseAddr = oldAddr;
    }
    auto oldPhys = getPhysicalAddr(baseAddr);
    auto addr = baseAddr ^ (1ULL << shift);
    if(addr >= base && addr < base+size)
    {
        volatile uint8_t* p;
        p = (uint8_t*) addr;
        *p;
        auto phys = getPhysicalAddr(addr);
        auto diff = std::bitset<64>(phys) ^ std::bitset<64>(oldPhys);
        if(diff.count() > 1)
        {
            //moved into a new frame
            shift = MIN_SHIFT;
            return getRandomAddress(base,size);
        }
        else
        {
            shift++;
            return addr;
        }
    }
    else
    {
        shift = MIN_SHIFT;
        return getRandomAddress(base,size);
    }
}

uint64_t getUsableBits(uint64_t removeFront, uint64_t removeBack)
{
    uint64_t usableBits = 64ULL - removeFront - removeBack;
    return  usableBits;
}

void cleanAddresses(std::map<size_t,std::vector<size_t>>& addresses,
                    uint64_t removeFront, uint64_t removeBack)
{
    std::cout << "\n=== Debug: Addresses Before Cleaning ===\n";
    for (const auto& list : addresses)
    {
        std::cout << "Set " << list.first << ": ";
        for (auto a : list.second)
        {
            std::cout << std::hex << a << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=========================================\n";

    auto usableBits = getUsableBits(removeFront, removeBack);
    uint64_t mask = 1;
    mask = mask << usableBits;
    mask = mask - 1ULL;

    for(auto& list : addresses)
    {
        for(auto& a : list.second)
        {
            a >>= removeFront;
            a &= mask;
        }
    }

    std::cout << "\n=== Debug: Addresses After Cleaning ===\n";
    for (const auto& list : addresses)
    {
        std::cout << "Set " << list.first << ": ";
        for (auto a : list.second)
        {
            std::cout << std::hex << a << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=========================================\n";
}


std::vector<Solver::Solution> calculateAddressingFunction(const std::map<size_t,std::vector<size_t>>& addresses, size_t addrFuncBits, size_t usableBits)
{
    std::vector<Solver::Solution> sList;
    for(size_t bit = 0; bit < addrFuncBits; bit++)
    {
        std::vector<uint64_t> matrix;
        Solver s;
        for(auto adrList : addresses)
        {
            uint64_t mask = 1ULL << bit;
            auto bitValue = (adrList.first & mask) >> bit;
            for(auto row : adrList.second)
            {
                auto rowWithResult = (row << 1ULL) | bitValue;
                matrix.push_back(rowWithResult);
            }
        }
        std::cout << "\n=== Debug: Solver Input Matrix ===\n";
        for (auto row : matrix) {
            std::cout << std::bitset<64>(row) << std::endl;
        }
        std::cout << "====================================\n";

        if (matrix.size() < 10) {
            std::cerr << "ERROR: Not enough unique data for solver - matrix size = " << matrix.size() << std::endl;
            return {}; 
        }

        s.solve(matrix,usableBits);
        auto sol = s.getSolution(matrix);
        sList.push_back(sol);
    }
    return sList;
}

void printSolution(const Solver::Solution& s, size_t offset)
{
    if(s.exists)
    {
        std::cout << "Involved bits: ";
        for (auto b : s.involvedBits)
        {
            std::cout << offset + b << " ";
        }
        std::cout << std::endl;
        std::cout << "Uninvolved bits: ";
        for (auto b : s.uninvolvedBits)
        {
            std::cout << offset + b << " ";
        }
        std::cout << std::endl;
        std::cout << "Unknown bits: ";
        for (auto b : s.unknownBits)
        {
            std::cout << offset + b << " ";
        }
        std::cout << std::endl;
    }
    else
    {
        std::cout << "No exact solution found" << std::endl;
        
        std::map<size_t, size_t> bitFrequency;
        for (auto b : s.unknownBits)
            bitFrequency[b] = 0;

        for (auto b : s.involvedBits)
            bitFrequency[b]++;

        size_t maxVotes = 0;
        for (auto [bit, votes] : bitFrequency)
            if (votes > maxVotes) maxVotes = votes;

        std::cout << "Likely bits (>=95% confidence): ";
        for (auto [bit, votes] : bitFrequency) {
            double confidence = (maxVotes > 0) ? (100.0 * votes / maxVotes) : 0.0;
            if (confidence >= 95.0) {
                std::cout << std::dec << (offset + static_cast<size_t>(bit)) << " (" << confidence << "%) ";
            }
        }
        std::cout << std::endl;
    }
}

void printSolutions(const std::vector<Solver::Solution> sList, size_t offset)
{
    for(size_t i = 0; i< sList.size(); i++)
    {
        cout << "Bit " << i << ": " << endl;
        printSolution(sList[i],offset);
    }
}


std::map<size_t,std::vector<uint64_t>> compactSets(const std::map<size_t,std::vector<uint64_t>>& addresses)
{
    std::map<size_t,std::vector<uint64_t>> newAddresses;
    size_t newIdx = 0;
    for(const auto& adr : addresses)
    {
        if(!adr.second.empty())
        {
            newAddresses.insert(make_pair(newIdx,adr.second));
            newIdx++;
        }
    }
    return newAddresses;
}


void prepareSolvePrint(AddressSet adrs,size_t removeFront, size_t removeBack)
{
    cleanAddresses(adrs,removeFront,removeBack);
    adrs = compactSets(adrs);
    auto expectedBits = static_cast<size_t>(ceil(log2(adrs.size())));
    auto cSol = calculateAddressingFunction(adrs,expectedBits,getUsableBits(removeFront,removeBack));
    std::cout << "\n=== DEBUG: First 10 addresses before solving ===\n";
    int count = 0;
    for (const auto &entry : adrs) {
        std::cout << "Set " << entry.first << ": ";
        for (const auto &addr : entry.second) {
            std::cout << std::hex << addr << " ";
            if (++count >= 10) break; // only print first 10
        }
        std::cout << std::endl;
        if (count >= 10) break;
    }
    std::cout << "================================================\n";

    printSolutions(cSol,removeFront);
}

int main(int argc, char *argv[])
{

    int opt;
    bool verbose = false;
    bool considerTadRegions = false;
    unsigned int sizeGb = 20;
    size_t numAddressTotal = 5000;
    size_t numAccess = 4000;
    int successfulMatches = 0;
    int failedPmuMatches = 0;
    while ((opt = getopt(argc, argv, "vrs:n:a:")) != -1)
    {
        switch (opt)
        {
        case 'v': verbose=true; break;
        case 'r': considerTadRegions = true; break;
        case 's': sizeGb = atoi(optarg); break;
        case 'n': numAddressTotal = atoi(optarg); break;
        case 'a': numAccess = atoi(optarg); break;

        default:
            std::cout <<"Usage: " << argv[0] <<"\n"
                     << "[-v] Verbose output\n"
                     << "[-r] Resolve addressing function assuming multiple TAD regions\n"
                     << "[-s <memPoolSize>]\n"
                     << "[-n <number of samples collected>]\n"
                     << "[-a <number of accesses for one component test>]\n"
                     << "Must be run as root to resolve physical addresses\n"
                     << "Must be pinned to one socket and its local memory\n";
            exit(EXIT_FAILURE);
        }
    }

    unsigned int nodeid;
    unsigned int cpuid;
    auto status = syscall(SYS_getcpu, &cpuid, &nodeid, nullptr);
    if(status == -1)
    {
        cout << "Can not determine node id" << endl;
        return EXIT_FAILURE;
    }
    if(verbose) cout << "Running on socket " << nodeid << endl;

    initPagemap();
    const uint64_t size = sizeGb * 1024 * 1024 * 1024ULL;
    auto space = (uint64_t) allocate(size);
    if(space == 0)
    {
        return EXIT_FAILURE;
    }
    srand(3344);

    std::set<uint64_t> usedAddresses;
    std::map<size_t,std::vector<uint64_t>>channelAddresses;
    std::map<size_t,std::vector<uint64_t>>rankAddresses;
    std::map<size_t,std::vector<uint64_t>>bankAddresses;
    std::map<size_t,std::vector<uint64_t>>bankGroupAddresses;



    cout << "Collecting address samples ..." << endl;
    auto adr=space;
    while(usedAddresses.size() < numAddressTotal)
    {
        adr = getNextAddress(adr,space,size);
        auto physicalAddress = getPhysicalAddr(adr);
        if(usedAddresses.count(physicalAddress) > 0)
        {
            continue;
        }
        usedAddresses.insert(physicalAddress);
        int identifiedChannel = -1;
        int identifiedRank = -1;
        int identifiedBank = -1;
        int identifiedBankGroup = -1;
        bool found = false;
        //long long maxCount = 0; // Track highest event count found

        std::vector<uint64_t> results[512];

        for (unsigned int channel = 0; channel < 4; channel++) {
            for (unsigned int rank = 0; rank < 8; rank++) {
                for (unsigned int bank = 0; bank < 16; bank++) {
                    long long initialCount = 0;
                    usleep(1);
                    int fd = setupMeasure(cpuid, channel, rank, bank);
                    startMeasure(fd, initialCount);
                    access(adr, numAccess);
                    auto count = stopMeasure(fd, initialCount);

                    if (count >= 0 * numAccess) {
                        int idx = (channel << 7) | (rank << 4) | bank;
                        auto& counts = results[idx];
                        counts.push_back(count);
                        std::sort(counts.begin(), counts.end(), std::greater<>());
                        if (counts.size() > 3)
                            counts.resize(3);
                    }
                }
            }
        }

        uint64_t maxValue = 0;
        int maxIndex = -1;
        for (int idx = 0; idx < 512; idx++) {
            //if (results[idx] > maxValue) {
            //    maxValue = results[idx];
            //    maxIndex = i;
            //}
            if (!results[idx].empty()) {
                int channel = (idx >> 7) & 0b11;
                int rank = (idx >> 4) & 0b111;
                int bank = idx & 0b1111;

                std::cout << "Channel " << channel << ", Rank " << rank << ", Bank " << bank << " => Top counts: ";
                for (auto val : results[idx]) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
           } 
       }

       // int bestChannel = (maxIndex >> 7) & 0b11;
       // int bestRank = (maxIndex >> 4) & 0b111;
       // int bestBank = maxIndex & 0b1111;

       // std::cout << "Max PMU count: " << maxValue << " at index " << maxIndex << " (Channel " << bestChannel << ", Rank " << bestRank << ", Bank " << bestBank << ")" << std::endl; 

    //    if(found && identifiedChannel < 4 && identifiedRank < 8 && identifiedBank < 16)
    //    {
    //        successfulMatches++;
    //        if(verbose) cout << bitset<64>(physicalAddress);
    //        channelAddresses[identifiedChannel].push_back(physicalAddress);
    //        rankAddresses[identifiedRank].push_back(physicalAddress);
    //        bankAddresses[identifiedBank].push_back(physicalAddress);
    //        identifiedBankGroup = identifiedBank / 4;
    //        bankGroupAddresses[identifiedBankGroup].push_back(physicalAddress);
    //        if(verbose) cout << " Channel " << identifiedChannel << " Rank " << identifiedRank << " Bank " << identifiedBank << " BankGroup " << identifiedBankGroup << endl;
    //    }
    //    else {
    //    failedPmuMatches++;
    //    }
    }

    for (size_t i = 0 ;i < 4; i++)
    {
        std::cout << "Captured " << channelAddresses[i].size() << " addresses on channel " << i << endl;
    }
    for (size_t j = 0 ;j < 8; j++)
    {
        std::cout << "Captured " << rankAddresses[j].size() << " addresses on rank " << j << endl;
    }
    for(size_t k = 0; k < 16; k++)
    {
        std::cout << "Captured " << bankAddresses[k].size() << " addresses on bank " << k << endl;
    }
    for(size_t k = 0; k < 4; k++)
    {
        std::cout << "Captured " << bankGroupAddresses[k].size() << " addresses on bankGroup " << k << endl;
    }
    cout << endl;

    std::cout << "Failed PMU matches: " << failedPmuMatches << std::endl;
    std::cout << "Valid collected samples: " << successfulMatches << std::endl;

    uint64_t andAll = std::numeric_limits<uint64_t>::max();
    uint64_t orAll = 0;
    for(auto a : usedAddresses)
    {
        orAll |= a;
        andAll &= a;
    }
    // bits with value 0 in orAll are 0 in all addresses
    // bits with value 1 in andAll are 1 in all addresses
    std::bitset<64> orAllBits(orAll);
    std::bitset<64> andAllBits(andAll);
    if(verbose) std::cout << "And all bits: " << andAllBits  << endl;
    if(verbose) std::cout << "Or all bits:  " << orAllBits  << endl;
    std::bitset<64> unknownBits = 0;
    for(size_t i = 0; i < 64; i++)
    {
        if(orAllBits[i] == 0 || andAllBits[i] == 1)
        {
            unknownBits[i] = 1;
        }
    }
    if(verbose) std::cout << "Unknown bits: " << unknownBits  << endl;


    uint64_t removeFront = 0;
    for(size_t i = 0; i < 64; i++)
    {
        if(unknownBits[i] == 0)
        {
            removeFront = i;
            break;
        }
    }
    if(verbose) std::cout << "Remove " << removeFront << " from front" << endl;

    uint64_t removeBack = 0;
    for(size_t i = 64; i-- > 0;)
    {
        if(unknownBits[i] == 0)
        {
            removeBack = 63 - i ;
            break;
        }
    }
    if(verbose) std::cout << "Remove " << removeBack << " from back" << endl;
    if(verbose) cout << endl;


    if(considerTadRegions)
    {
        auto tadRegions = SysInfo::getTadRegions(nodeid);
        std::map<size_t,AddressSet> regionControllerAddresses;
        std::map<size_t,std::map<size_t,AddressSet>> regionControllerChannelAddresses;

        //build zones
        for(size_t c = 0; c < channelAddresses.size(); c++)
        {
            auto adrList = channelAddresses.at(c);
            for(auto a : adrList)
            {
                for(auto rIt = tadRegions.begin(); rIt != tadRegions.end(); rIt++)
                {
                    auto limitAddress = rIt->first;
                    auto controllerList = rIt->second;
                    if(a <= limitAddress) //found a region that matches
                    {
                        if(controllerList.size() > 1) // this region has controller interleaving on
                        {
                            auto contr = SysInfo::channelToController(c);
                            regionControllerAddresses[limitAddress][contr].push_back(a);
                        }
                        for(size_t contr = 0; contr < controllerList.size(); contr++)
                        {
                            if(controllerList.at(contr).channelInterleaving > 1)
                            {
                                auto contrRef = SysInfo::channelToController(c);
                                if(contrRef == contr)
                                {
                                    regionControllerChannelAddresses[limitAddress][contr][c].push_back(a);
                                }
                            }
                        }
                        break; // regions are sorted. break after first match
                    }
                }
            }
        }


        cout << "Channels" << endl;
        unsigned int regionIndex = 0;
        size_t lastLimitAddress = 0;
        const size_t REGION_UNIT = 1024*1024;
        for(auto tadRegion : tadRegions)
        {
            auto limitAddress = tadRegion.first;
            cout << "Region " << regionIndex << " from " << lastLimitAddress/REGION_UNIT << "M to " << limitAddress/REGION_UNIT << "M:" << endl;
            auto controllerAddrSetIt = regionControllerAddresses.find(limitAddress);
            if(controllerAddrSetIt != regionControllerAddresses.end() && !controllerAddrSetIt->second.empty() )
            {
                cout << "Controller Interleaving:" << endl;
                prepareSolvePrint(controllerAddrSetIt->second,removeFront,removeBack);
            }
            else
            {
                if(tadRegion.second.size()==0)
                {
                    cout << "No addresses captured in this region" << endl;
                    lastLimitAddress = limitAddress;
                    regionIndex++;
                    continue;
                }
                else if(tadRegion.second.size() == 1)
                {
                    cout << "Single controller" << endl;
                }
            }
            auto controllerChannelAddrSetIt = regionControllerChannelAddresses.find(limitAddress);
            if(controllerChannelAddrSetIt != regionControllerChannelAddresses.end())
            {
                unsigned int contrIndex = 0;
                auto controllerChannelAddrSet = controllerChannelAddrSetIt->second;
                for(auto contr : controllerChannelAddrSet)
                {
                    cout << "Channel interleaving in controller " << contrIndex << ":" << endl;
                    prepareSolvePrint(contr.second,removeFront,removeBack);
                    contrIndex++;
                }
            }
            lastLimitAddress = limitAddress;
            regionIndex++;
        }
        cout << endl;
    }
    else // Assume there is only one TAD region
    {
        cout << "Channels" << endl;
        prepareSolvePrint(channelAddresses,removeFront,removeBack);
        cout << endl;
    }

    cout << "Ranks" << endl;
    auto setNums = getUsedSets(rankAddresses);
    cout << "Used ranks: ";
    for(auto s : setNums)
    {
        cout << s << " ";
    }
    cout << endl;
    prepareSolvePrint(rankAddresses,removeFront,removeBack);
    cout << endl;

    cout << "Banks" << endl;
    prepareSolvePrint(bankAddresses,removeFront,removeBack);
    cout << endl;

    cout << "Bank Groups" << endl;
    prepareSolvePrint(bankGroupAddresses,removeFront,removeBack);
    cout << endl;

    return 0;
}
