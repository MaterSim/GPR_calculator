#ifndef RBF_KERNEL_H 
#define RBF_KERNEL_H

extern "C" void rbf_kee_many(int n1, int n2, int d, int x2i, double zeta, 
        double sigma2, double l2, double* x1, int* ele1, int* x1_inds, 
        double* x2, int* ele2, int* x2_inds, double* pout);
        
extern "C" void rbf_kee_many_with_grad(int n1, int n2, int d, int x2i, double zeta, 
        double sigma2, double l2, double* x1, int* ele1, int* x1_inds, 
        double* x2, int* ele2, int* x2_inds, double* pout, double* dpout_dl);
 
extern "C" void rbf_kef_many(int n1, int n2, int d, int x2i, double zeta, 
        double sigma2, double l2, double* x1, int* ele1, int* x1_inds, 
        double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
       
extern "C" void rbf_kef_many_with_grad(int n1, int n2, int d, int x2i, double zeta, 
        double sigma2, double l, double* x1, int* ele1, int* x1_inds, 
        double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
     
extern "C" void rbf_kef_many_stress(int n1, int n2, int d, int x2i, double zeta, 
        double sigma2, double l2, double* x1, int* ele1, int* x1_inds, 
        double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
     
extern "C" void rbf_kff_many(int n1, int n2, int n2_start, int n2_end, int d, 
        int x2i, double zeta, double sigma2, double l2, double tol, 
        double* x1, double* dx1dr, int* ele1, int* x1_inds, 
        double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
       
extern "C" void rbf_kff_many_with_grad(int n1, int n2, int n2_start, int n2_end, int d, 
        int x2i, double zeta, double sigma2, double l, 
        double* x1, double* dx1dr, int* ele1, int* x1_inds, 
        double* x2, double* dx2dr, int* ele2, int* x2_inds, 
        double* pout, double* dpout_dl);
    
extern "C" void rbf_kff_many_stress(int n1, int n2, int n2_start, int n2_end, int d, 
        int x2i, double zeta, double sigma2, double l2, double tol, 
        double* x1, double* dx1dr, int* ele1, int* x1_inds, 
        double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
 
extern "C" void rbf_kff_many_diag(int n1, int d, 
        int x2i, double zeta, double sigma2, double l2, double tol, 
        double* x1, double* dx1dr, int* ele1, int* x1_inds, 
        double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout);
 
#endif // RBF_KERNEL_H
