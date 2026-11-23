# eip5656-mcopy-sim

EIP-5656 **MCOPY** simulator in pure Python — no dependencies.

MCOPY (opcode `0x27`, Cancun) copies a slice of EVM memory onto another
slice of the same memory in a single instruction:

```
MCOPY(destOffset, srcOffset, size)
```

## Gas

```
gas = 3                          # verylow base cost
    + 3 * ceil(size / 32)        # per-word copy cost
    + memory_expansion_cost(...) # if the copy touches new memory
```

Memory expansion follows the standard EVM rule, charged once on the
highest byte touched by either the source or the destination range:

```
words = ceil(highest_byte / 32)
expansion_cost = 3 * words + words**2 // 512
```

A hand-rolled `MLOAD`/`MSTORE` loop pays 3 gas per word for the load
*and* 3 per word for the store, plus per-iteration loop overhead — so
MCOPY moves a word for 3 gas total and wins by at least 3 gas per word
even before the loop instructions are counted. Overlapping ranges copy
safely, like `memmove`: the bytes read are the bytes that were there
when the copy started.

## Usage

```python
from mcopy import Memory, mcopy_gas, loop_copy_gas

mem = Memory()
mem.store(0, bytes(range(64)))   # expansion gas charged on store
gas = mem.mcopy(128, 0, 64)      # 21 gas; copy is intact and overlap-safe
print(mem.load(128, 64) == bytes(range(64)))  # True
print(mem.gas_spent)             # 27 (6 store + 21 copy)

mcopy_gas(64, 0, src_offset=0, dest_offset=64)  # (9, 12, 21)
loop_copy_gas(64, 0, dest_offset=64)            # 24 (lower bound, no loop overhead)
```

## Run

```bash
python mcopy.py --demo   # copy a 64-byte blob and price it against a loop
python mcopy.py --test   # 39 doctests
```

## What's inside

- `Memory` — byte-addressable EVM memory with expansion-cost accounting;
  reads past the end return zero bytes, writes expand memory like the EVM.
- `mcopy_gas(size, current_words, src_offset, dest_offset)` — gas
  breakdown `(base_plus_copy, expansion, total)`.
- `memory_expansion_cost(current_words, new_words)` — the
  `3w + w²//512` expansion rule.
- `loop_copy_gas(...)` — lower bound for the equivalent
  `MLOAD`/`MSTORE` loop, for comparison.
