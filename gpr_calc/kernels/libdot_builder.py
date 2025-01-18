import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef("""
    void dot_kee_many(int n1, int n2, int d, int x2i, double zeta, double sigma, double sigma02, double* x1, int* ele1, int* x1_inds, double* x2, int* ele2, int* x2_inds, double* pout);
    void dot_kef_many(int n1, int n2, int d, int x2i, double zeta, double* x1, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
    void dot_kef_many_stress(int n1, int n2, int d, int x2i, double zeta, double* x1, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
    void dot_kff_many(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta, double* x1, double* dx1dr, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
    void dot_kff_many_stress(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta, double* x1, double* dx1dr, int* ele1, int* x1_inds, double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
""")
ffibuilder.set_source("gpr_calc.kernels._dot_kernel", 
    '#include "dot_kernel.h"', 
    sources=["gpr_calc/kernels/dot_kernel.cpp"],
    include_dirs=["gpr_calc/kernels/"],
    language="c++", 
    extra_compile_args=["-std=c++11"], 
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
