#include <cupy/complex.cuh>
using C = complex<double>;
using Index = unsigned long long;

__device__ Index modular_source(Index word, Index modulus, Index inverse,
        int shift, Index mask, int flag) {
    const Index local = (word >> shift) & mask;
    if (local >= modulus || (flag >= 0 && !(word & (1ULL << flag)))) return word;
    return word ^ ((local ^ ((local * inverse) % modulus)) << shift);
}
extern "C" __global__ void transform_modular(const C* old, C* out, Index modulus,
        Index inverse, int shift, Index mask, int flag, Index n, Index total, int mixed) {
    const Index t=(Index)blockIdx.x*blockDim.x + threadIdx.x;
    if (t>=total) return;
    if (!mixed) {
        out[t]=old[modular_source(t,modulus,inverse,shift,mask,flag)];
        return;
    }
    const Index a=modular_source((t/n)%n,modulus,inverse,shift,mask,flag);
    const Index b=modular_source(t%n,modulus,inverse,shift,mask,flag);
    out[t]=old[(t/(n*n))*n*n+a*n+b];
}

// Wire zero is the most significant bit. The ordering of wires[] is semantic.
__device__ Index gather(Index word, const int* shifts, int k) {
    Index local = 0;
    for (int j = 0; j < k; ++j)
        local = (local << 1) | ((word >> shifts[j]) & 1ULL);
    return local;
}
__device__ Index replace_bits(Index word, Index local, const int* shifts, int k) {
    for (int j = 0; j < k; ++j) {
        const Index mask = 1ULL << shifts[j];
        word = (word & ~mask) | (((local >> (k-j-1)) & 1ULL) << shifts[j]);
    }
    return word;
}
extern "C" __global__ void transform_wave(const C* old, C* out, const C* gate,
        const int* shifts, int k, Index dimension) {
    const Index y = (Index)blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= dimension) return;
    const Index d = 1ULL << k, row = gather(y, shifts, k);
    C sum(0, 0);
    for (Index i=0; i<d; ++i)
        sum += gate[row*d+i] * old[replace_bits(y,i,shifts,k)];
    out[y] = sum;
}
extern "C" __global__ void transform_density(const C* old, C* out, const C* kraus,
        const int* shifts, int k, int count, Index n, Index total) {
    const Index t = (Index)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const Index x=t%n, y=(t/n)%n, base=t/(n*n)*(n*n), d=1ULL<<k;
    const Index u=gather(y,shifts,k), v=gather(x,shifts,k);
    C sum(0,0);
    for (int a=0; a<count; ++a) {
        const C* K=kraus+a*d*d;
        for (Index i=0;i<d;++i) for (Index j=0;j<d;++j)
            sum += K[u*d+i] * old[base+replace_bits(y,i,shifts,k)*n+
                                      replace_bits(x,j,shifts,k)] * conj(K[v*d+j]);
    }
    out[t]=sum;
}
extern "C" __global__ void measure_z(const C* old, C* out, int qshift,
        int cshift, Index n, Index total) {
    const Index t=(Index)blockIdx.x*blockDim.x+threadIdx.x;
    if (t>=total) return;
    const Index x=t%n, y=(t/n)%n, c=t/(n*n), mask=1ULL<<cshift;
    const Index b=(y>>qshift)&1ULL;
    if (((x>>qshift)&1ULL)!=b || ((c>>cshift)&1ULL)!=b) { out[t]=C(0,0); return; }
    // Both possible previous values of the overwritten classical bit contribute.
    out[t]=old[(c&~mask)*n*n+y*n+x]+old[(c|mask)*n*n+y*n+x];
}
