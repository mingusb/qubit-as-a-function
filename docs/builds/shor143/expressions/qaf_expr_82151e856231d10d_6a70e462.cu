
    
    
#include <cupy/complex.cuh>
using C = complex<double>;
__device__ bool natural(C a) {
    return a.imag()==0 && a.real()>=0 && a.real()<9007199254740992.0
           && floor(a.real())==a.real();
}
__device__ C bit_value(C a, C b) {
    if (!natural(a) || !natural(b)) return C(nan(""),0);
    if (b.real()>=53) return C(0,0);
    return C(((unsigned long long)a.real() >> (int)b.real()) & 1ULL,0);
}
__device__ C mod_value(C a, C b) {
    if (!natural(a) || !natural(b) || b.real()==0) return C(nan(""),0);
    return C(fmod(a.real(),b.real()),0);
}

__device__ double qreal(C z,int &fuel) { if(z.imag()!=0 || !isfinite(z)) {fuel=-1;return 0;}return z.real();}
__device__ C qdiv(C a,C b,int &fuel) {if(b==C(0,0)){fuel=-1;return C(0,0);}return a/b;}
__device__ __noinline__ C q_power_of_two(int &fuel,int depth,C v_n);
__device__ __noinline__ C q_pow_mod(int &fuel,int depth,C v_a,C v_e,C v_n);
__device__ __noinline__ C q_inverse(int &fuel,int depth,C v_a,C v_n);
__device__ __noinline__ C q_power_of_two(int &fuel,int depth,C v_n) {
if(depth>32 || fuel<0) {fuel=-1; return {};}
if (--fuel<0) return {};
auto v_v=C(1.00000000000000000e+00,0);
if (--fuel<0) return {};
while(qreal(C((qreal(v_n,fuel)>qreal(C(0.00000000000000000e+00,0),fuel))?1:0,0),fuel)!=0) {
if (--fuel<0) return {};
if (--fuel<0) return {};
v_v=(C(2.00000000000000000e+00,0)*v_v);
if (--fuel<0) return {};
v_n=(v_n-C(1.00000000000000000e+00,0));
}
if (--fuel<0) return {};
return v_v;
fuel=-1; return {};
}
__device__ __noinline__ C q_pow_mod(int &fuel,int depth,C v_a,C v_e,C v_n) {
if(depth>32 || fuel<0) {fuel=-1; return {};}
if (--fuel<0) return {};
auto v_y=C(1.00000000000000000e+00,0);
if (--fuel<0) return {};
while(qreal(C((qreal(v_e,fuel)>qreal(C(0.00000000000000000e+00,0),fuel))?1:0,0),fuel)!=0) {
if (--fuel<0) return {};
if (--fuel<0) return {};
auto v_odd=mod_value(v_e,C(2.00000000000000000e+00,0));
if (--fuel<0) return {};
if(qreal(v_odd,fuel)!=0) {
if (--fuel<0) return {};
if (--fuel<0) return {};
v_y=mod_value((v_y*v_a),v_n);
}
if (--fuel<0) return {};
v_a=mod_value((v_a*v_a),v_n);
if (--fuel<0) return {};
v_e=qdiv((v_e-v_odd),C(2.00000000000000000e+00,0),fuel);
}
if (--fuel<0) return {};
return v_y;
fuel=-1; return {};
}
__device__ __noinline__ C q_inverse(int &fuel,int depth,C v_a,C v_n) {
if(depth>32 || fuel<0) {fuel=-1; return {};}
if (--fuel<0) return {};
auto v_r=v_n;
if (--fuel<0) return {};
auto v_s=v_a;
if (--fuel<0) return {};
auto v_u=C(0.00000000000000000e+00,0);
if (--fuel<0) return {};
auto v_v=C(1.00000000000000000e+00,0);
if (--fuel<0) return {};
while(qreal(C((qreal(v_s,fuel)>qreal(C(0.00000000000000000e+00,0),fuel))?1:0,0),fuel)!=0) {
if (--fuel<0) return {};
if (--fuel<0) return {};
auto v_rem=mod_value(v_r,v_s);
if (--fuel<0) return {};
auto v_q=qdiv((v_r-v_rem),v_s,fuel);
if (--fuel<0) return {};
auto v_w=(v_u-(v_q*v_v));
if (--fuel<0) return {};
v_r=v_s;
if (--fuel<0) return {};
v_s=v_rem;
if (--fuel<0) return {};
v_u=v_v;
if (--fuel<0) return {};
v_v=v_w;
}
if (--fuel<0) return {};
if(qreal(C((qreal(v_u,fuel)<qreal(C(0.00000000000000000e+00,0),fuel))?1:0,0),fuel)!=0) {
if (--fuel<0) return {};
if (--fuel<0) return {};
return (v_u+v_n);
}
if (--fuel<0) return {};
return v_u;
fuel=-1; return {};
}
    extern "C" __global__ void qaf_expr_82151e856231d10d(const CArray<long long, 0, 1, 1> _raw_y, const CArray<long long, 0, 1, 1> _raw_x, CArray<thrust::complex<double>, 0, 1, 1> _raw_value, CIndexer<0, 1> _ind) {
      ;
      #pragma unroll 1
      CUPY_FOR(i, _ind.size()) {
        _ind.set(i);
        const long long &y = _raw_y[_ind.get()];
const long long &x = _raw_x[_ind.get()];
thrust::complex<double> &value = _raw_value[_ind.get()];
int fuel=4096, depth=0; value = C(-1.57079632679489656e+00,0); if(fuel<0) value=C(nan(""),0);;
      }
      ;
    }
    