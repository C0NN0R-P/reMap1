#include <iostream>
#include <cstdint>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <cstdlib>
#include <map>
#include <vector>
#include <cerrno>

static const size_t PAGE_SIZE = 4096;
static const size_t PAGEMAP_ENTRY_SIZE = 8; // Each entry is 64 bits in /proc/<pid>/pagemap

// DRAM fields (bank is bits [13..16], row bits [18..34], etc.)
struct DRAMFields {
    uint64_t byte_offset;
    uint64_t column;
    uint64_t bank;
    uint64_t rank;
    uint64_t row;
};

//------------------------------------------------------------------------------
// 1) virtual_to_physical (reads /proc/self/pagemap)
//------------------------------------------------------------------------------
uint64_t virtual_to_physical(uint64_t vaddr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        std::perror("open /proc/self/pagemap");
        return 0;
    }

    uint64_t page_index = vaddr / PAGE_SIZE;
    off_t offset = page_index * PAGEMAP_ENTRY_SIZE;

    if (lseek(fd, offset, SEEK_SET) == (off_t)-1) {
        std::perror("lseek");
        close(fd);
        return 0;
    }

    uint64_t entry = 0;
    ssize_t bytes_read = read(fd, &entry, PAGEMAP_ENTRY_SIZE);
    if (bytes_read < 0) {
        std::perror("read");
        close(fd);
        return 0;
    } else if (bytes_read != PAGEMAP_ENTRY_SIZE) {
        std::cerr << "Unexpected read size: " << bytes_read << std::endl;
        close(fd);
        return 0;
    }
    close(fd);

    // Check if page is present (bit 63)
    if ((entry & (1ULL << 63)) == 0) {
        // Not present or swapped
        return 0;
    }

    // PFN = bits [0..54]
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    uint64_t page_offset = vaddr % PAGE_SIZE;
    uint64_t paddr = (pfn << 12) + page_offset;
    return paddr;
}

//------------------------------------------------------------------------------
// 2) decode_dram_fields
//------------------------------------------------------------------------------
DRAMFields decode_dram_fields(uint64_t phys) {
    DRAMFields df;
    df.byte_offset =  (phys >> 0) & 0x7;       // bits [0..2]
    df.column      =  (phys >> 3) & 0x3FF;     // bits [3..12]
    df.bank        =  (phys >> 13) & 0xF;      // bits [13..16]
    df.rank        =  (phys >> 17) & 0x1;      // bit [17]
    df.row         =  (phys >> 18) & 0x1FFFF;  // bits [18..34]
    return df;
}

//------------------------------------------------------------------------------
// parse_size_input: Accepts "10G", "20M", "50K", or numeric for bytes
//------------------------------------------------------------------------------
size_t parse_size_input(const std::string &input) {
    if (input.empty()) return 0;

    char suffix = input.back();
    std::string numericPart = input;
    numericPart.pop_back(); // remove last char

    uint64_t multiplier = 1;
    bool hasSuffix = false;
    switch (suffix) {
        case 'K': case 'k':
            multiplier = 1024ULL;
            hasSuffix = true;
            break;
        case 'M': case 'm':
            multiplier = 1024ULL * 1024ULL;
            hasSuffix = true;
            break;
        case 'G': case 'g':
            multiplier = 1024ULL * 1024ULL * 1024ULL;
            hasSuffix = true;
            break;
        default:
            // no recognized suffix -> entire string is digits
            numericPart = input;
            break;
    }

    uint64_t value = 0;
    try {
        value = std::stoull(numericPart);
    } catch (...) {
        return 0;
    }
    return static_cast<size_t>(value * multiplier);
}

//------------------------------------------------------------------------------
// MAIN
//------------------------------------------------------------------------------
int main() {
    std::cout << "Enter size to allocate (e.g. 500M, 2G, 4096): ";
    std::string input;
    std::cin >> input;

    size_t size_bytes = parse_size_input(input);
    if (size_bytes == 0) {
        std::cerr << "Invalid size!\n";
        return 1;
    }

    size_t num_pages = (size_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    size_t alloc_size = num_pages * PAGE_SIZE;

    std::cout << "Allocating " << num_pages << " pages = " << alloc_size << " bytes\n";

    // 1) Allocate
    char* region = (char*) malloc(alloc_size);
    if (!region) {
        std::cerr << "malloc failed\n";
        return 1;
    }

    // 2) Touch each page, partial pattern
    for (size_t i = 0; i < num_pages; i++) {
        // Force the page to commit
        region[i * PAGE_SIZE] = (char)i;

        // small pattern fill
        uint32_t* pagePtr = (uint32_t*)&region[i * PAGE_SIZE];
        for (size_t j = 0; j < 16; j++) {
            pagePtr[j] = 0xDEAD0000 + (uint32_t)((i << 8) ^ j);
        }
    }

    // 3) Frequency maps
    std::map<uint64_t, size_t> row_freq;
    std::map<uint64_t, size_t> col_freq;
    std::map<uint64_t, size_t> rank_freq;
    std::map<uint64_t, size_t> bank_freq;

    // 4) For each page, check PFN consistency and do readback
    for (size_t i = 0; i < num_pages; i++) {
        uintptr_t va = (uintptr_t)&region[i * PAGE_SIZE];

        // a) Read physical address first time
        uint64_t pa1 = virtual_to_physical(va);
        // b) Read second time
        uint64_t pa2 = virtual_to_physical(va);

        if (!pa1 || !pa2) {
            std::cerr << "Page " << i << ": not present or error\n";
            continue;
        }
        if (pa1 != pa2) {
            std::cerr << "Page " << i << ": PFN mismatch! pa1=0x"
                      << std::hex << pa1 << ", pa2=0x" << pa2 << std::dec << "\n";
        }

        // c) decode DRAM fields from pa1
        DRAMFields df = decode_dram_fields(pa1);

        // d) partial readback
        uint32_t* pagePtr = (uint32_t*)&region[i * PAGE_SIZE];
        for (size_t j = 0; j < 16; j++) {
            uint32_t expected = 0xDEAD0000 + (uint32_t)((i << 8) ^ j);
            if (pagePtr[j] != expected) {
                std::cerr << "Readback mismatch at Page " << i << ", j=" << j
                          << ": got=0x" << std::hex << pagePtr[j]
                          << ", expected=0x" << expected << std::dec << "\n";
                // break or keep scanning
                break;
            }
        }

        // e) Update frequency
        row_freq[df.row]++;
        col_freq[df.column]++;
        rank_freq[df.rank]++;
        bank_freq[df.bank]++;
    }

    // 5) free
    free(region);

    // 6) Print frequency analysis
    std::cout << "\nFrequency Analysis Results:\n";

    // Rows
    std::cout << "\n-- Rows --\n";
    for (const auto& kv : row_freq) {
        std::cout << "Row " << kv.first << " appears " << kv.second << " times\n";
    }

    // Columns
    std::cout << "\n-- Columns --\n";
    for (const auto& kv : col_freq) {
        std::cout << "Column " << kv.first << " appears " << kv.second << " times\n";
    }

    // Ranks
    std::cout << "\n-- Ranks --\n";
    for (const auto& kv : rank_freq) {
        std::cout << "Rank " << kv.first << " appears " << kv.second << " times\n";
    }

    // Banks
    std::cout << "\n-- Banks --\n";
    for (const auto& kv : bank_freq) {
        std::cout << "Bank " << kv.first << " appears " << kv.second << " times\n";
    }

    return 0;
}
