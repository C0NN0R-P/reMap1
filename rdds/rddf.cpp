#include <iostream>
#include <cstdint>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <cstdlib>
#include <map>      // For frequency maps

static const size_t PAGE_SIZE = 4096;
static const size_t PAGEMAP_ENTRY_SIZE = 8; // Each entry = 64 bits in /proc/<pid>/pagemap

// DRAM fields: same as before, but 'bank' = bits[13..16].
struct DRAMFields {
    uint64_t byte_offset; // bits [0..2]
    uint64_t column;      // bits [3..12]
    uint64_t bank;        // bits [13..16]
    uint64_t rank;        // bit  [17]
    uint64_t row;         // bits [18..34]
};

//------------------------------------------------------------------------------
// Translate virtual address -> physical address
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
        // Not present
        return 0;
    }

    // PFN = bits [0..54]
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    uint64_t page_offset = vaddr % PAGE_SIZE;
    uint64_t paddr = (pfn << 12) + page_offset;
    return paddr;
}

//------------------------------------------------------------------------------
// Decode PA -> DRAM fields
//------------------------------------------------------------------------------
DRAMFields decode_dram_fields(uint64_t phys) {
    DRAMFields df;
    df.byte_offset = (phys >> 0) & 0x7;        // bits [0..2]
    df.column      = (phys >> 3) & 0x3FF;      // bits [3..12]
    df.bank        = (phys >> 13) & 0xF;       // bits [13..16] (0..15)
    df.rank        = (phys >> 17) & 0x1;       // bit [17]
    df.row         = (phys >> 18) & 0x1FFFF;   // bits [18..34]
    return df;
}

//------------------------------------------------------------------------------
// Parse an input size, allowing "10G", "20K", "50M", or numeric bytes
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
// MAIN: allocate, decode pages, track frequency of Row/Col/Rank/Bank
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

    // Round up to pages
    size_t num_pages = (size_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    size_t alloc_size = num_pages * PAGE_SIZE;

    std::cout << "Allocating " << num_pages << " pages = " << alloc_size << " bytes\n";

    // Allocate
    char* region = (char*) malloc(alloc_size);
    if (!region) {
        std::cerr << "malloc failed\n";
        return 1;
    }

    // Touch each page
    for (size_t i = 0; i < num_pages; i++) {
        region[i * PAGE_SIZE] = (char)i;
    }

    // Frequency maps: Row, Column, Rank, Bank
    std::map<uint64_t, size_t> row_freq;
    std::map<uint64_t, size_t> col_freq;
    std::map<uint64_t, size_t> rank_freq;
    std::map<uint64_t, size_t> bank_freq;

    // Iterate pages
    for (size_t i = 0; i < num_pages; i++) {
        uintptr_t va = (uintptr_t) &region[i * PAGE_SIZE];
        uint64_t pa = virtual_to_physical((uint64_t) va);
        if (!pa) {
            std::cerr << "Page " << i << ": not present or error\n";
            continue;
        }

        DRAMFields df = decode_dram_fields(pa);

        // Optionally, if you want to see them individually:
        /*
        std::cout << "Page " << i << ": "
                  << "VA=0x" << std::hex << va
                  << " PA=0x" << pa << std::dec
                  << " => Row=" << df.row
                  << " Col=" << df.column
                  << " Rank=" << df.rank
                  << " Bank=" << df.bank
                  << "\n";
        */

        // Increment frequency
        row_freq[df.row]++;
        col_freq[df.column]++;
        rank_freq[df.rank]++;
        bank_freq[df.bank]++;
    }

    free(region);

    // Print results
    std::cout << "\nFrequency Analysis Results:\n";

    // Rows
    std::cout << "\n-- Rows --\n";
    for (const auto &kv : row_freq) {
        std::cout << "Row " << kv.first << " appears " << kv.second << " times\n";
    }

    // Columns
    std::cout << "\n-- Columns --\n";
    for (const auto &kv : col_freq) {
        std::cout << "Column " << kv.first << " appears " << kv.second << " times\n";
    }

    // Ranks
    std::cout << "\n-- Ranks --\n";
    for (const auto &kv : rank_freq) {
        std::cout << "Rank " << kv.first << " appears " << kv.second << " times\n";
    }

    // Banks
    std::cout << "\n-- Banks --\n";
    for (const auto &kv : bank_freq) {
        std::cout << "Bank " << kv.first << " appears " << kv.second << " times\n";
    }

    return 0;
}
