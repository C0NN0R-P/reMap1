#include <iostream>
#include <cstdint>
#include <unistd.h>     // for close(), read(), etc.
#include <sys/mman.h>   // for mmap
#include <fcntl.h>      // for open()
#include <cstring>      // for memset, strcmp
#include <cstdlib>      // for malloc/free

static const size_t PAGE_SIZE = 4096;
static const size_t PAGEMAP_ENTRY_SIZE = 8; // Each entry is 64 bits in /proc/<pid>/pagemap

//-----------------------------------------
// We'll rename 'bank_index' to just 'bank'
// The decode is bits [13..16] => 0..15
//-----------------------------------------
struct DRAMFields {
    uint64_t byte_offset; // bits [0..2]
    uint64_t column;      // bits [3..12]
    uint64_t bank;        // bits [13..16]
    uint64_t rank;        // bit  [17]
    uint64_t row;         // bits [18..34]
};

//------------------------------------------------------------------------------
// 1) Translate a virtual address → physical address
//    Returns 0 on error or if page not present
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

    // bit 63: page present?
    if ((entry & (1ULL << 63)) == 0) {
        // Page not present
        return 0;
    }

    // PFN is bits [0..54]
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    uint64_t page_offset = vaddr % PAGE_SIZE;
    uint64_t paddr = (pfn << 12) + page_offset;
    return paddr;
}

//------------------------------------------------------------------------------
// 2) Decode physical address → DRAM fields
//    combined 'bank' bits [13..16], row bits [18..34], etc.
//------------------------------------------------------------------------------
DRAMFields decode_dram_fields(uint64_t phys) {
    DRAMFields fields;
    fields.byte_offset = (phys >> 0) & 0x7;         // bits [0..2]
    fields.column      = (phys >> 3) & 0x3FF;       // bits [3..12]
    fields.bank        = (phys >> 13) & 0xF;        // bits [13..16]
    fields.rank        = (phys >> 17) & 0x1;        // bit  [17]
    fields.row         = (phys >> 18) & 0x1FFFF;    // bits [18..34]
    return fields;
}

//------------------------------------------------------------------------------
// Parse user input like "10G", "20K", "50M", or just a numeric "12345" for bytes
//------------------------------------------------------------------------------
size_t parse_size_input(const std::string& input) {
    // We handle suffix K/k -> *1024, M/m -> *1024^2, G/g -> *1024^3
    // If no suffix, treat as raw bytes
    // Return 0 on invalid parse

    if (input.empty()) return 0;

    // Find numeric portion + possible suffix
    // A simple approach: check last char for 'k','K','m','M','g','G'
    char suffix = input.back();
    std::string numericPart = input;
    numericPart.pop_back(); // remove last char

    uint64_t multiplier = 1;
    bool hasSuffix = false;
    switch (suffix) {
        case 'K':
        case 'k':
            multiplier = 1024ULL;
            hasSuffix = true;
            break;
        case 'M':
        case 'm':
            multiplier = 1024ULL * 1024ULL;
            hasSuffix = true;
            break;
        case 'G':
        case 'g':
            multiplier = 1024ULL * 1024ULL * 1024ULL;
            hasSuffix = true;
            break;
        default:
            // not a recognized suffix, maybe it was just digits
            // so treat the entire input as numeric
            numericPart = input; // restore
            break;
    }

    // Convert numericPart to a number
    uint64_t value = 0;
    try {
        value = std::stoull(numericPart);
    } catch (...) {
        // parse error
        return 0;
    }
    return static_cast<size_t>(value * multiplier);
}

//------------------------------------------------------------------------------
// MAIN: Allocate a user-specified size, iterate pages
//       Reorder output to: Row, Column, Rank, Bank
//------------------------------------------------------------------------------
int main() {
    std::cout << "Enter the size to allocate (e.g. 10G, 20K, 50M, or 4096 bytes): ";
    std::string sizeStr;
    std::cin >> sizeStr;

    size_t size_bytes = parse_size_input(sizeStr);
    if (size_bytes == 0) {
        std::cerr << "Invalid size input!\n";
        return 1;
    }

    // Round up to page multiples
    size_t num_pages = (size_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    size_t alloc_size = num_pages * PAGE_SIZE;

    std::cout << "Allocating " << num_pages << " pages (" << alloc_size << " bytes)\n";

    // Allocate memory (malloc or mmap)
    char* region = (char*) malloc(alloc_size);
    if (!region) {
        std::cerr << "malloc failed!\n";
        return 1;
    }

    // Touch each page so it's actually mapped
    for (size_t i = 0; i < num_pages; i++) {
        region[i * PAGE_SIZE] = (char) i;  // ensures the page is mapped
    }

    // Iterate each page and decode
    for (size_t i = 0; i < num_pages; i++) {
        uintptr_t va = (uintptr_t) &region[i * PAGE_SIZE];
        uint64_t paddr = virtual_to_physical((uint64_t) va);
        if (paddr == 0) {
            std::cerr << "Page " << i << ": not present or error\n";
            continue;
        }
        DRAMFields df = decode_dram_fields(paddr);

        // Print summary: Reordered => Row, Column, Rank, Bank
        std::cout << "Page " << i << ": \n"
                  << "  VA = 0x" << std::hex << va
                  << ", PA = 0x" << paddr << std::dec << "\n"
                  << "  Row   = "   << df.row
                  << ", Column = " << df.column
                  << ", Rank = "   << df.rank
                  << ", Bank = "   << df.bank
                  << "\n\n";
    }

    // Clean up
    free(region);
    return 0;
}
