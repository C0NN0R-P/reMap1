#!/bin/bash

# Usage: ./decode_pa.sh <physical_address>
# Example: ./decode_pa.sh 0x1234567890

ADDR="$1"

if [[ -z "$ADDR" ]]; then
    echo "Usage: $0 <physical_address>"
    exit 1
fi

DECODE_PATH="/sys/kernel/debug/skx_decode/addr"

# Check if the decoder interface exists
if [[ ! -e "$DECODE_PATH" ]]; then
    echo "Error: Decoder interface not found at $DECODE_PATH"
    echo "Make sure the module is inserted and debugfs is mounted:"
    echo "  sudo mount -t debugfs none /sys/kernel/debug"
    exit 2
fi

# Write the address using sudo tee (avoiding redirection issues)
echo "Decoding physical address: $ADDR"
echo "$ADDR" | sudo tee "$DECODE_PATH" > /dev/null

# Show the latest kernel log lines with matches
echo
echo "=== Kernel Messages ==="
dmesg | tail -n 50 | grep -i skx_decoder
