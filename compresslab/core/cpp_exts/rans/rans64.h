// 64-bit rANS encoder/decoder - public domain - Fabian 'ryg' Giesen 2014
//
// This uses 64-bit states (63-bit actually) which allows renormalizing
// by writing out a whole 32 bits at a time (b=2^32) while still
// retaining good precision and allowing for high probability resolution.
//
// The only caveat is that this version requires 64-bit arithmetic; in
// particular, the encoder approximation in the bottom half requires a
// fast way to obtain the top 64 bits of an unsigned 64*64 bit product.
//
// In short, as written, this code works on 64-bit targets only!

#ifndef RANS64_HEADER
#define RANS64_HEADER

#include <stdint.h>

#ifdef assert
#define Rans64Assert assert
#else
#define Rans64Assert(x)
#endif

// --------------------------------------------------------------------------

// This code needs support for 64-bit long multiplies with 128-bit result
// (or more precisely, the top 64 bits of a 128-bit result). This is not
// really portable functionality, so we need some compiler-specific hacks
// here.

#if defined(_MSC_VER)

#include <intrin.h>

static inline uint64_t Rans64MulHi(uint64_t a, uint64_t b)
{
    return __umulh(a, b);
}

#elif defined(__GNUC__)

static inline uint64_t Rans64MulHi(uint64_t a, uint64_t b)
{
    return (uint64_t) (((unsigned __int128)a * b) >> 64);
}

#else

#error Unknown/unsupported compiler!

#endif

// --------------------------------------------------------------------------

// L ('l' in the paper) is the lower bound of our normalization interval.
// Between this and our 32-bit-aligned emission, we use 63 (not 64!) bits.
// This is done intentionally because exact reciprocals for 63-bit uints
// fit in 64-bit uints: this permits some optimizations during encoding.
#define RANS64_L (1ull << 31)  // lower bound of our normalization interval

// State for a rANS encoder. Yep, that's all there is to it.
typedef uint64_t Rans64State;

// Initialize a rANS encoder.
static inline void Rans64EncInit(Rans64State* r)
{
    *r = RANS64_L;
}

// Encodes a single symbol with range start "start" and frequency "freq".
// All frequencies are assumed to sum to "1 << scale_bits", and the
// resulting bytes get written to ptr (which is updated).
//
// NOTE: With rANS, you need to encode symbols in *reverse order*, i.e. from
// beginning to end! Likewise, the output bytestream is written *backwards*:
// ptr starts pointing at the end of the output buffer and keeps decrementing.
/**
 * @param r             The rANS state
 * @param pptr          The pointer to the output buffer
 * @param start         CDF[s]
 * @param freq          freq[s] = CDF[s+1] - CDF[s]
 * @param scale_bits    The number of bits used to represent the cumulative frequency
 */
static inline void Rans64EncPut(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    Rans64Assert(freq != 0);

    // renormalize (never needs to loop)
    uint64_t x = *r;
    uint64_t x_max = ((RANS64_L >> scale_bits) << 32) * freq; // this turns into a shift.
    // if (x / freq << scale_bits) >= 2 ^ 63, we need to write 32 bits to the encoded bitstream
    // and refresh the state
    if (x >= x_max) { 
        *pptr -= 1; // move backward to write a new 32-bit word
        **pptr = (uint32_t) x;
        x >>= 32; // now the state statisfy 2 ^ 31 <= x < 2 ^ 63
        Rans64Assert(x < x_max);
    }

    // x = C(s,x)
    // x_{next} = (x // freq[s]) * 2^scale_bits + CDF[s] + (x % freq[s])
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
}

// Flushes the rANS encoder.
static inline void Rans64EncFlush(Rans64State* r, uint32_t** pptr)
{
    uint64_t x = *r;

    *pptr -= 2;
    (*pptr)[0] = (uint32_t) (x >> 0);
    (*pptr)[1] = (uint32_t) (x >> 32);
}

// Initializes a rANS decoder.
// Unlike the encoder, the decoder works forwards as you'd expect.
static inline void Rans64DecInit(Rans64State* r, uint32_t** pptr)
{
    uint64_t x;

    x  = (uint64_t) ((*pptr)[0]) << 0;
    x |= (uint64_t) ((*pptr)[1]) << 32;
    *pptr += 2;
    *r = x;
}

// Returns the current cumulative frequency (map it to a symbol yourself!)
static inline uint32_t Rans64DecGet(Rans64State* r, uint32_t scale_bits)
{
    return *r & ((1u << scale_bits) - 1);
}

// Advances in the bit stream by "popping" a single symbol with range start
// "start" and frequency "freq". All frequencies are assumed to sum to "1 << scale_bits",
// and the resulting bytes get written to ptr (which is updated).
static inline void Rans64DecAdvance(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    uint64_t mask = (1ull << scale_bits) - 1;

    // s, x = D(x)
    uint64_t x = *r;
    x = freq * (x >> scale_bits) + (x & mask) - start;

    // renormalize
    if (x < RANS64_L) {
        x = (x << 32) | **pptr;
        *pptr += 1;
        Rans64Assert(x >= RANS64_L);
    }

    *r = x;
}

// --------------------------------------------------------------------------

#endif // RANS64_HEADER
