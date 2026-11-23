"""EIP-5656 MCOPY simulator in pure Python.

MCOPY (opcode 0x27, Cancun) copies a slice of EVM memory onto another
slice of the same memory, in one instruction:

    MCOPY(destOffset, srcOffset, size)

Gas (EIP-5656):

    gas = 3                                  # verylow base cost
          + 3 * ceil(size / 32)              # per-word copy cost
          + memory_expansion_cost(...)       # if the copy touches new memory

Memory expansion follows the standard EVM rule, charged on the highest
byte touched by either the source or the destination range:

    words = ceil(highest_byte / 32)
    expansion_cost = 3 * words + words**2 // 512

Unlike a hand-rolled MLOAD/MSTORE loop (which pays 3 gas per word for the
load *and* 3 per word for the store, plus loop overhead), MCOPY moves a
word for 3 gas total, and overlapping ranges copy safely, like memmove:
the bytes read are the bytes that were there when the copy started, not
bytes already overwritten by an earlier part of the same copy.

No dependencies; everything here is stdlib-only.

Run the demo:   python mcopy.py --demo
Run the tests:   python mcopy.py --test
"""

from __future__ import annotations

import math
import sys

__all__ = [
    "MCOPY_OPCODE",
    "VERYLOW_GAS",
    "COPY_GAS_PER_WORD",
    "Memory",
    "memory_expansion_cost",
    "mcopy_gas",
    "mcopy",
    "loop_copy_gas",
]

MCOPY_OPCODE = 0x27
VERYLOW_GAS = 3          # base cost of MCOPY (verylow tier)
COPY_GAS_PER_WORD = 3    # per 32-byte word copied

WORD_SIZE = 32


def _words(nbytes: int) -> int:
    """Number of 32-byte words needed to hold ``nbytes`` bytes."""
    if nbytes < 0:
        raise ValueError("byte count must not be negative")
    return (nbytes + WORD_SIZE - 1) // WORD_SIZE


def memory_expansion_cost(current_words: int, new_words: int) -> int:
    """Extra gas to grow EVM memory from ``current_words`` to ``new_words``.

    The EVM memory cost function is ``3 * w + w**2 // 512``; expansion is
    charged as the difference between the cost at the new size and the
    cost at the old size. Shrinking memory never refunds gas.

    >>> memory_expansion_cost(0, 0)
    0
    >>> memory_expansion_cost(0, 1)
    3
    >>> memory_expansion_cost(0, 32)
    98
    >>> memory_expansion_cost(32, 32)
    0
    >>> memory_expansion_cost(32, 33)
    3
    >>> memory_expansion_cost(10, 5)
    0
    """
    if current_words < 0 or new_words < 0:
        raise ValueError("word counts must not be negative")
    if new_words <= current_words:
        return 0

    def cost(w: int) -> int:
        return 3 * w + (w * w) // 512

    return cost(new_words) - cost(current_words)


def mcopy_gas(size: int, current_words: int,
              src_offset: int = 0, dest_offset: int = 0) -> tuple[int, int, int]:
    """Gas breakdown for ``MCOPY(dest_offset, src_offset, size)``.

    Returns ``(base_plus_copy, expansion, total)`` where ``base_plus_copy``
    is ``3 + 3 * ceil(size / 32)`` and ``expansion`` is the memory-expansion
    cost for the highest byte touched by either range. Expansion is
    charged once, on the furthest byte of the two ranges, not twice.

    >>> mcopy_gas(0, 0)
    (3, 0, 3)
    >>> mcopy_gas(32, 0)
    (6, 3, 9)
    >>> mcopy_gas(33, 0)
    (9, 6, 15)
    >>> mcopy_gas(64, 0, src_offset=0, dest_offset=64)
    (9, 12, 21)
    >>> mcopy_gas(32, 4, src_offset=0, dest_offset=0)
    (6, 0, 6)
    """
    if size < 0 or src_offset < 0 or dest_offset < 0:
        raise ValueError("MCOPY arguments must not be negative")
    base = VERYLOW_GAS + COPY_GAS_PER_WORD * _words(size)
    if size == 0:
        return base, 0, base
    highest = max(src_offset + size, dest_offset + size)
    expansion = memory_expansion_cost(current_words, _words(highest))
    return base, expansion, base + expansion


class Memory:
    """A byte-addressable EVM memory with expansion-cost accounting.

    Reading beyond the current size returns zero bytes (like the EVM);
    writing or copying into untouched memory expands it and charges the
    standard expansion cost. ``gas_spent`` accumulates every charge so a
    sequence of operations can be costed end to end.
    """

    def __init__(self) -> None:
        self._data = bytearray()
        self.gas_spent = 0

    @property
    def size(self) -> int:
        """Current memory size in bytes."""
        return len(self._data)

    @property
    def size_words(self) -> int:
        """Current memory size in 32-byte words."""
        return _words(len(self._data))

    def _expand_to(self, nbytes: int) -> int:
        """Grow memory to cover ``nbytes`` bytes; returns expansion gas."""
        new_words = _words(nbytes)
        cost = memory_expansion_cost(self.size_words, new_words)
        if new_words > self.size_words:
            self._data.extend(b"\x00" * (new_words * WORD_SIZE - len(self._data)))
        self.gas_spent += cost
        return cost

    def load(self, offset: int, size: int) -> bytes:
        """Read ``size`` bytes starting at ``offset`` (zero-filled past end).

        >>> m = Memory()
        >>> m.load(0, 4)
        b'\\x00\\x00\\x00\\x00'
        >>> m.store(0, b"abcd")
        3
        >>> m.load(0, 4)
        b'abcd'
        >>> m.load(2, 4)
        b'cd\\x00\\x00'
        """
        if offset < 0 or size < 0:
            raise ValueError("offset and size must not be negative")
        if size == 0:
            return b""
        end = offset + size
        if end > len(self._data):
            chunk = bytes(self._data[offset:]) if offset < len(self._data) else b""
            return chunk + b"\x00" * (size - len(chunk))
        return bytes(self._data[offset:end])

    def store(self, offset: int, data: bytes) -> int:
        """Write ``data`` at ``offset``; returns the expansion gas charged.

        >>> m = Memory()
        >>> m.store(0, b"hello")
        3
        >>> m.size
        32
        >>> m.store(64, b"!")
        6
        >>> m.size
        96
        """
        if offset < 0:
            raise ValueError("offset must not be negative")
        if not data:
            return 0
        cost = self._expand_to(offset + len(data))
        self._data[offset:offset + len(data)] = data
        return cost

    def mcopy(self, dest_offset: int, src_offset: int, size: int) -> int:
        """Execute MCOPY: copy ``size`` bytes from ``src_offset`` to ``dest_offset``.

        Overlapping ranges are safe: the source bytes are snapshotted
        before any byte is written, exactly like the EVM's memmove
        semantics. Returns the total gas charged for this instruction.

        >>> m = Memory()
        >>> m.store(0, bytes(range(8)))
        3
        >>> m.mcopy(32, 0, 8)
        9
        >>> m.load(32, 8)
        b'\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07'
        >>> # Forward overlap: dest starts inside the source range.
        >>> m.mcopy(2, 0, 6)
        6
        >>> list(m.load(0, 8))
        [0, 1, 0, 1, 2, 3, 4, 5]
        >>> # Zero-size copy costs only the base 3 gas and touches nothing.
        >>> m.mcopy(1000, 0, 0)
        3
        >>> m.size
        64
        >>> # Gas accounting: 3 (store) + 9 + 6 + 3, no double-counted expansion.
        >>> m.gas_spent
        21
        """
        if dest_offset < 0 or src_offset < 0 or size < 0:
            raise ValueError("MCOPY arguments must not be negative")
        base, expansion, total = mcopy_gas(
            size, self.size_words, src_offset, dest_offset)
        if size == 0:
            self.gas_spent += total
            return total
        chunk = self.load(src_offset, size)  # snapshot before writing
        expansion_actual = self._expand_to(max(src_offset + size,
                                               dest_offset + size))
        self._data[dest_offset:dest_offset + size] = chunk
        self.gas_spent += base  # expansion already charged by _expand_to
        return base + expansion_actual


def mcopy(dest_offset: int, src_offset: int, size: int,
          memory: Memory | None = None) -> tuple[Memory, int]:
    """One-shot MCOPY on a fresh (or supplied) memory; returns ``(memory, gas)``.

    >>> mem, gas = mcopy(0, 0, 32)
    >>> gas
    9
    >>> mem.size
    32
    >>> mem, gas = mcopy(64, 0, 64)
    >>> gas
    21
    >>> mem.size
    128
    """
    mem = memory if memory is not None else Memory()
    return mem, mem.mcopy(dest_offset, src_offset, size)


def loop_copy_gas(size: int, current_words: int,
                  src_offset: int = 0, dest_offset: int = 0) -> int:
    """Gas of copying ``size`` bytes with a hand-rolled MLOAD/MSTORE loop.

    Each word costs 3 (MLOAD) + 3 (MSTORE) = 6 gas, plus the same memory
    expansion MCOPY pays, plus loop overhead the EVM charges per
    iteration (JUMP, ADD, LT, ...). The overhead is not modelled here,
    so this is a *lower bound* — MCOPY still wins by at least 3 gas
    per word even before the loop instructions are counted.

    >>> loop_copy_gas(32, 0)
    9
    >>> loop_copy_gas(64, 0, dest_offset=64)
    24
    >>> loop_copy_gas(0, 0)
    0
    """
    if size < 0 or src_offset < 0 or dest_offset < 0:
        raise ValueError("arguments must not be negative")
    if size == 0:
        return 0
    highest = max(src_offset + size, dest_offset + size)
    expansion = memory_expansion_cost(current_words, _words(highest))
    return 6 * _words(size) + expansion


def demo() -> None:
    """Copy a 64-byte calldata blob with MCOPY and price it against a loop."""
    payload = bytes(range(64))
    mem = Memory()
    mem.store(0, payload)
    print("stored 64-byte payload at offset 0 "
          f"(expansion gas so far: {mem.gas_spent})")
    gas = mem.mcopy(128, 0, 64)
    print(f"MCOPY(128, 0, 64): {gas} gas "
          f"(total spent: {mem.gas_spent}), copy intact: "
          f"{mem.load(128, 64) == payload}")
    loop = loop_copy_gas(64, mem.size_words - 4, dest_offset=128)
    print(f"same copy with an MLOAD/MSTORE loop would cost >= {loop} gas "
          f"(before loop overhead)")
    print(f"MCOPY saves at least {loop - gas} gas on 64 bytes")


def _test() -> int:
    import doctest
    results = doctest.testmod(verbose=False)
    print(f"{results.attempted} doctests, {results.failed} failures")
    return 1 if results.failed else 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(_test())
    demo()
