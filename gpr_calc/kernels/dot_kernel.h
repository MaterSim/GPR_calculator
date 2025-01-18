#ifndef DOT_KERNEL_H
#define DOT_KERNEL_H

extern "C" void dot_kee_many(int n1, int n2, int d, int x2i, double zeta, double sigma2, double sigma02,
                         double* x1, int* ele1, int* x1_inds, 
                         double* x2, int* ele2, int* x2_inds, 
                         double* pout);

extern "C" void dot_kef_many(int n1, int n2, int d, int x2i, double zeta, 
                         double* x1, int* ele1, int* x1_inds, 
                         double* x2, double* dx2dr, int* ele2, int* x2_inds, 
                         double* pout);

extern "C" void dot_kef_many_stress(int n1, int n2, int d, int x2i, double zeta, 
                        double* x1, int* ele1, int* x1_inds, 
                        double* x2, double* dx2dr, int* ele2, int* x2_inds, 
                        double* pout);

extern "C" void dot_kff_many(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta,
                         double* x1, double* dx1dr, int* ele1, int* x1_inds, 
                         double* x2, double* dx2dr, int* ele2, int* x2_inds, 
                         double* pout);

extern "C" void dot_kff_many_stress(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta,
                        double* x1, double* dx1dr, int* ele1, int* x1_inds, 
                        double* x2, double* dx2dr, int* ele2, int* x2_inds, 
                        double* pout);

#endif // DOT_KERNEL_H
