#include "solver.h"
#include <iostream>
#include <bitset>
#include <cassert>
#include <limits>
#include <set>
#include <algorithm> // for std::set_intersection

/* 
 * solver.cpp
 *
 * Corrected version addressing pivot checks, for-loop range,
 * and clarifying the "no solution" scenario.
 */

//bool Solver::debug = false;

/**
 * swapRows: Helper that exchanges two rows in 'matrix'.
 */
void Solver::swapRows(Matrix& matrix, size_t r1, size_t r2)
{
    std::swap(matrix[r1], matrix[r2]);
}

/**
 * xorRows: Xors 'matrix[row]' into every other row that has the bit 'col' set.
 * 
 * The logic is standard for a bitwise elimination step in GF(2):
 *   If row X is the pivot row for column col, for every other row Y where col is set,
 *   we do rowY ^= pivotRow, effectively clearing col in rowY.
 */
void Solver::xorRows(Matrix& matrix, size_t row, size_t col)
{
    uint64_t mask = (1ULL << col);
    uint64_t pivotVal = matrix[row];

    for (size_t r = 0; r < matrix.size(); r++)
    {
        if (r != row)
        {
            // If this row has bit 'col' set, we xor with pivotVal.
            if ((matrix[r] & mask) != 0)  // corrected from '== mask'
            {
                matrix[r] ^= pivotVal;
            }
        }
    }
}

/**
 * findRowToSwap: searches rows below 'row' for one that has
 * bit 'col' set, and returns that row index for pivoting.
 */
size_t Solver::findRowToSwap(const Matrix& matrix, size_t row, size_t col)
{
    uint64_t mask = (1ULL << col);
    for (size_t r = row + 1; r < matrix.size(); r++)
    {
        if ((matrix[r] & mask) != 0)  // corrected from '== mask'
        {
            return r;
        }
    }
    return row;
}

/**
 * printMatrix: debug utility to print matrix rows as bit strings.
 */
void Solver::printMatrix(const Matrix& matrix)
{
    for (auto row : matrix)
    {
        std::cout << std::bitset<bitSize>(row) << std::endl;
    }
    std::cout << std::endl;
}

/**
 * getSetBit: if there's exactly one set bit in 'bits', returns its index,
 *            otherwise returns max().
 */
size_t Solver::getSetBit(std::bitset<bitSize> bits)
{
    for (size_t i = 0; i < bitSize; i++)
    {
        if (bits[i] == 1)
        {
            return i;
        }
    }
    return std::numeric_limits<size_t>::max();
}

/**
 * getSetBits: returns the indices of all set bits in 'bits'.
 */
std::set<size_t> Solver::getSetBits(std::bitset<bitSize> bits)
{
    std::set<size_t> out;
    assert(bits.count() > 1);
    for (size_t i = 0; i < bitSize; i++)
    {
        if (bits[i] == 1)
        {
            out.insert(i);
        }
    }
    return out;
}

/**
 * solve: performs a mod-2 Gaussian elimination to reduce 'matrix'.
 *
 * The matrix has 'colMax + 1' columns effectively: colMax for address bits,
 * plus 1 for the final result bit. We adjust the loop to ensure columns down to 0
 * are included (not skipping c=1 or c=0).
 */
void Solver::solve(Matrix& matrix, size_t colMax)
{
    // Start from colMax, plus 1 for the "RHS" bit. We'll iterate down to 0 inclusive.
    // We'll also track row 'r' for pivoting.
    for (size_t c = colMax + 1, r = 0; c-- > 0 && r < matrix.size(); )
    {
        uint64_t mask = (1ULL << c);

        // Check if pivot bit is set in row 'r':
        // corrected from 'if ((matrix[r] & mask) == 1)' to '!= 0'.
        if ((matrix[r] & mask) != 0)
        {
            xorRows(matrix, r, c);
            r++;
        }
        else
        {
            // find pivot in subsequent rows
            size_t rowToSwap = findRowToSwap(matrix, r, c);
            if (rowToSwap == r)
            {
                // no pivot found in lower rows => move to next column
                continue;
            }
            else
            {
                // swap, then xor
                swapRows(matrix, r, rowToSwap);
                xorRows(matrix, r, c);
                r++;
            }
        }

        if (debug)
        {
            printMatrix(matrix);
        }
    }
}

/**
 * getSolution: interprets each row (post-elimination) as a set of bits + a rightmost bit.
 * - If bits.count()==0 && rhs==1 => "No solution" contradiction (0=1).
 * - If bits.count()==1 && rhs==0 => that single bit is "uninvolved".
 * - If bits.count()==1 && rhs==1 => that single bit is "involved".
 * - If bits.count()>=2 => we label them "unknownBits".
 *
 * WARNING: This approach lumps multi-bit solutions into 'unknownBits',
 * so it won't explicitly express a multi-bit XOR. It's a simplified classification.
 */
Solver::Solution Solver::getSolution(const std::vector<uint64_t>& matrix)
{
    Solution s;
    s.exists = true;

    for (auto row : matrix)
    {
        // bits = everything but the last bit
        // rhs  = the last bit (lowest bit of row)
        std::bitset<bitSize> bits(row >> 1);
        std::bitset<1> rhs(row & 1);

        // If row is all zero but final bit is 1 => contradiction in linear algebra
        // e.g. 0*x1 + 0*x2 + ... = 1. That means "no solution".
        if (bits.count() == 0 && rhs == 1)
        {
            std::cout << "No solution exists (contradiction row: [all zero | 1]).\n";
            s.exists = false;
            break;
        }
        else if (bits.count() == 1 && rhs == 0)
        {
            // Single bit with 0 => that bit is 'uninvolved'
            s.uninvolvedBits.insert(getSetBit(bits));
        }
        else if (bits.count() == 1 && rhs == 1)
        {
            // Single bit with 1 => that bit is 'involved'
            s.involvedBits.insert(getSetBit(bits));
        }
        else if (bits.count() >= 2)
        {
            // If there's more than one set bit, we treat it as 'unknown' in this simplified logic.
            auto setBits = getSetBits(bits);
            s.unknownBits.insert(setBits.begin(), setBits.end());
        }
    }

    // If we never triggered 'exists=false', we still might have overlap of bits in 'involved' vs. 'uninvolved'.
    if (s.exists)
    {
        // Remove bits that appear in both sets (this can happen if the code isn't fully consistent).
        std::set<uint64_t> common;
        std::set_intersection(s.involvedBits.begin(), s.involvedBits.end(),
                              s.uninvolvedBits.begin(), s.uninvolvedBits.end(),
                              std::inserter(common, common.begin()));
        for (auto c : common)
        {
            s.involvedBits.erase(c);
            s.uninvolvedBits.erase(c);
        }
    }

    return s;
}
