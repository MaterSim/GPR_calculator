#include "rbf_kernel.h"
#include <cmath>
#include <iostream>

extern "C"
void rbf_kee_many(int n1, int n2, int d, int x2i, double zeta, double sigma2, double l2,
              double* x1, int* ele1, int* x1_inds, 
              double* x2, int* ele2, int* x2_inds, 
              double* pout){
    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, x1x2_dot;
    double dx, D;
    int i, ii, jj, _i, _j, _ele1, _ele2;

    for(ii=0;ii<n1;ii++){
        _ele1 = ele1[ii];
        _i = x1_inds[ii];

        dval = 0.0;
        for(i=0;i<d;i++){
            dval += x1[ii*d+i]*x1[ii*d+i];
        }
        x1_norm = sqrt(dval);

        if(x1_norm>eps){
            for(jj=0;jj<n2;jj++){
                _ele2 = ele2[jj];
    	    	_j = x2_inds[jj];

    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);

                if(_ele1==_ele2 && x2_norm>eps){
                    x1x2_dot = 0.;
                    for(i=0;i<d;i++){
    	    	        x1x2_dot+=x1[ii*d+i]*x2[jj*d+i];
    	    		}
                    dx = x1x2_dot/(x1_norm*x2_norm);
                    D = pow(dx, zeta);
                    pout[_i*x2i+_j] += sigma2 * exp(-(1-D)/(2*l2));
                }
            }
        }
    }
};

extern "C"
void rbf_kee_many_with_grad(int n1, int n2, int d, int x2i, double zeta, double sigma2, double l2,
                        double* x1, int* ele1, int* x1_inds, 
                        double* x2, int* ele2, int* x2_inds, 
                        double* pout, double* dpout_dl){
    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, x1x2_dot;
    double dx, K, D;
    int i, ii, jj, _i, _j, _ele1, _ele2;
    
    for(ii=0;ii<n1;ii++){
        _ele1 = ele1[ii];
        _i = x1_inds[ii];

        dval = 0.0;
        for(i=0;i<d;i++){
            dval += x1[ii*d+i]*x1[ii*d+i];
        }
        x1_norm = sqrt(dval);

        if(x1_norm>eps){
            for(jj=0;jj<n2;jj++){
                _ele2 = ele2[jj];
    	    	_j = x2_inds[jj];

    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);

                if(_ele1==_ele2 && x2_norm>eps){
                    x1x2_dot = 0.;
                    for(i=0;i<d;i++){
    	    	        x1x2_dot+=x1[ii*d+i]*x2[jj*d+i];
    	    		}
                    dx = x1x2_dot/(x1_norm*x2_norm);
                    D = pow(dx, zeta);
                    K = sigma2 * exp(-(1-D)/(2*l2));

                    pout[_i*x2i+_j] += K;
                    dpout_dl[_i*x2i+_j] += K * (1-D); 
                }
            }           
        }
    }
};


extern "C"
void rbf_kef_many(int n1, int n2, int d, int x2i, double zeta, double sigma2, double l2,
              double* x1, int* ele1, int* x1_inds, 
              double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){
    double eps=1e-8;
    double dval, D, K, dK_dD;
    double x1_norm, x2_norm, _x1x2_norm, x1x2_dot, x1x2_dot_norm13;
    double dx, d1, dd_dx2;
    double C1, C2, C3;
    int i, ii, jj, _i, _j, _ele1, _ele2;
    double *dD_dx2;

    dD_dx2 = new double[d];

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];

    	dval = 0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);

    	if(x1_norm > eps){
    	    for(jj=0; jj<n2; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];

    	    	dval = 0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);

    	    	if(_ele1==_ele2 && x2_norm > eps){
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
                    x1x2_dot = 0;
    	    	    for(i=0;i<d;i++){
    	    		x1x2_dot += x1[ii*d+i]*x2[jj*d+i];
    	    		}
                    dx = x1x2_dot * _x1x2_norm;
                    D = pow(dx, zeta);
                    K = sigma2 * exp(-(1-D)/(2*l2));
                    dK_dD = K / (2*l2);

                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm/(x2_norm*x2_norm);
                    
                    d1 = zeta*pow(dx,zeta-1);

    	    	    for(i=0;i<d;i++){
    	    	        dd_dx2 = x1[ii*d+i]*_x1x2_norm - x1x2_dot_norm13*x2[jj*d+i];
    	    	        dD_dx2[i] = d1*dd_dx2;
    	    	    }

                    C1 = C2 = C3 = 0;
                    for(i=0;i<d;i++){
                        C1 += dx2dr[(jj*d+i)*3 + 0] * dD_dx2[i] * dK_dD;
                        C2 += dx2dr[(jj*d+i)*3 + 1] * dD_dx2[i] * dK_dD;
                        C3 += dx2dr[(jj*d+i)*3 + 2] * dD_dx2[i] * dK_dD;
                    }

                    pout[(_i*x2i+_j)*3]   -= C1;
                    pout[(_i*x2i+_j)*3+1] -= C2;
                    pout[(_i*x2i+_j)*3+2] -= C3;
                }
            }
        }
    }
    delete dD_dx2;
};

extern "C"
void rbf_kef_many_with_grad(int n1, int n2, int d, int x2i, double zeta, double sigma2, double l,
                        double* x1, int* ele1, int* x1_inds, 
                        double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){
    double eps=1e-8;
    double dval, D, K, dK_dD;
    double x1_norm, x2_norm, _x1x2_norm, x1x2_dot, x1x2_dot_norm13;
    double dx, d1, dd_dx2;
    double C1, C2, C3;
    int i, ii, jj, _i, _j, _ele1, _ele2;
    double *dD_dx2;
    double _l, _l3, l2;

    _l = 1 / l;
    _l3 = _l / (l * l);
    l2 = l*l;

    dD_dx2 = new double[d];

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];

    	dval = 0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);

    	if(x1_norm > eps){
    	    for(jj=0; jj<n2; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];

    	    	dval = 0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);

    	    	if(_ele1==_ele2 && x2_norm > eps){
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
                    x1x2_dot = 0;
    	    	    for(i=0;i<d;i++){
    	    		x1x2_dot += x1[ii*d+i]*x2[jj*d+i];
    	    		}
                    dx = x1x2_dot * _x1x2_norm;
                    D = pow(dx, zeta);
                    K = sigma2 * exp(-(1-D)/(2*l2));
                    dK_dD = K / (2*l2);

                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm/(x2_norm*x2_norm);
                    
                    d1 = zeta*pow(dx,zeta-1);

    	    	    for(i=0;i<d;i++){
    	    	        dd_dx2 = x1[ii*d+i]*_x1x2_norm - x1x2_dot_norm13*x2[jj*d+i];
    	    	        dD_dx2[i] = d1*dd_dx2;
    	    	    }

                    C1 = C2 = C3 = 0;
                    for(i=0;i<d;i++){
                        C1 += dx2dr[(jj*d+i)*3 + 0] * dD_dx2[i] * dK_dD;
                        C2 += dx2dr[(jj*d+i)*3 + 1] * dD_dx2[i] * dK_dD;
                        C3 += dx2dr[(jj*d+i)*3 + 2] * dD_dx2[i] * dK_dD;
                    }

                    pout[(_i*x2i+_j)*6+0] -= C1;
                    pout[(_i*x2i+_j)*6+1] -= C2;
                    pout[(_i*x2i+_j)*6+2] -= C3;

                    // dKef_dl
                    pout[(_i*x2i+_j)*6+3] += C1 * ((D-1) * _l3 + 2 * _l);
                    pout[(_i*x2i+_j)*6+4] += C2 * ((D-1) * _l3 + 2 * _l);
                    pout[(_i*x2i+_j)*6+5] += C3 * ((D-1) * _l3 + 2 * _l);
                }
            }
        }
    }
    delete dD_dx2;
};


extern "C"
void rbf_kef_many_stress(int n1, int n2, int d, int x2i, double zeta, double sigma2, double l2,
                     double* x1, int* ele1, int* x1_inds, 
                     double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){
    double eps=1e-8;
    double dval, D, K, dK_dD;
    double x1_norm, x2_norm, _x1x2_norm, x1x2_dot, x1x2_dot_norm13;
    double dx, d1, dd_dx2;
    double C1, C2, C3, C4, C5, C6, C7, C8, C9;
    int i, ii, jj, _i, _j, _ele1, _ele2;
    double *dD_dx2;

    dD_dx2 = new double[d];

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];

    	dval = 0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);

    	if(x1_norm > eps){
    	    for(jj=0; jj<n2; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];

    	    	dval = 0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);

    	    	if(_ele1==_ele2 && x2_norm > eps){
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
                    x1x2_dot = 0;
    	    	    for(i=0;i<d;i++){
    	    		    x1x2_dot += x1[ii*d+i]*x2[jj*d+i];
    	    		}
                    dx = x1x2_dot * _x1x2_norm;
                    D = pow(dx, zeta);
                    K = sigma2 * exp(-(1-D)/(2*l2));
                    dK_dD = K / (2*l2);

                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm/(x2_norm*x2_norm);
                    
                    d1 = zeta*pow(dx,zeta-1);

    	    	    for(i=0;i<d;i++){
    	    	        dd_dx2 = x1[ii*d+i]*_x1x2_norm - x1x2_dot_norm13*x2[jj*d+i];
    	    	        dD_dx2[i] = d1*dd_dx2;
    	    	    }

                    C1 = C2 = C3 = C4 = C5 = C6 = C7 = C8 = C9 = 0;
                    for(i=0; i<d; i++){
                        C1 += dx2dr[(jj*d+i)*9 + 0] * dD_dx2[i] * dK_dD; // Perhaps move this outside the loop?
                        C2 += dx2dr[(jj*d+i)*9 + 1] * dD_dx2[i] * dK_dD;
                        C3 += dx2dr[(jj*d+i)*9 + 2] * dD_dx2[i] * dK_dD;
                        C4 += dx2dr[(jj*d+i)*9 + 3] * dD_dx2[i] * dK_dD;
                        C5 += dx2dr[(jj*d+i)*9 + 4] * dD_dx2[i] * dK_dD;
                        C6 += dx2dr[(jj*d+i)*9 + 5] * dD_dx2[i] * dK_dD;
                        C7 += dx2dr[(jj*d+i)*9 + 6] * dD_dx2[i] * dK_dD;
                        C8 += dx2dr[(jj*d+i)*9 + 7] * dD_dx2[i] * dK_dD;
                        C9 += dx2dr[(jj*d+i)*9 + 8] * dD_dx2[i] * dK_dD;
                    }

                    pout[(_i*x2i+_j)*9+0] -= C1;
                    pout[(_i*x2i+_j)*9+1] -= C2;
                    pout[(_i*x2i+_j)*9+2] -= C3;
                    pout[(_i*x2i+_j)*9+3] -= C4;
                    pout[(_i*x2i+_j)*9+4] -= C5;
                    pout[(_i*x2i+_j)*9+5] -= C6;
                    pout[(_i*x2i+_j)*9+6] -= C7;
                    pout[(_i*x2i+_j)*9+7] -= C8;
                    pout[(_i*x2i+_j)*9+8] -= C9;                   
                }
            }
        }
    }
    delete dD_dx2;
};


extern "C"
void rbf_kff_many(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta, double sigma2, double l2, double tol,
              double* x1, double* dx1dr, int* ele1, int* x1_inds, 
              double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){
    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x2_norm2, _x1x2_norm, x1x2_dot;
    double _x1x2_norm31, x1x2_dot_norm02, x1x2_dot_norm13, x1x2_dot_norm31;
    double  dx, d2, d1, d2d_dx1dx2;
    double dddx1, dddx2, dDdx1, dDdx2, d2Ddx1dx2;
    double D, K, dK_dD;
    double  C1, C2, C3, C4, C5, C6, C7, C8, C9;
    int i, j, ii, jj, _i, _j, _ele1, _ele2;
    double *d2k_dx1dx2, *C;

    
    d2k_dx1dx2 = new double[d*d];
    C = new double[d*3];

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];

    	dval=0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);

    	if(x1_norm > eps){

    	    for(jj=n2_start; jj<n2_end; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];

    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);

    	    	if(_ele1==_ele2 && x2_norm > eps){
    	    	    _x2_norm2 = 1.0/(x2_norm*x2_norm);
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
                    _x1x2_norm31 = _x1x2_norm/(x1_norm*x1_norm);

                    x1x2_dot=0;
    	    	    for(i=0;i<d;i++){
    	    		    x1x2_dot+=x1[ii*d+i]*x2[jj*d+i];
    	    		}
                    dx = x1x2_dot * _x1x2_norm;
                    D = pow(dx, zeta);
                    K = sigma2 * exp(-(1-D)/(2*l2));
                    dK_dD = K / (2*l2);
                    if(dK_dD > tol){

                        x1x2_dot_norm31 = x1x2_dot*_x1x2_norm31;
                        x1x2_dot_norm13 = x1x2_dot*_x1x2_norm*_x2_norm2;
                        x1x2_dot_norm02 = x1x2_dot*_x2_norm2;

                        d2 = pow(dx,zeta-2);
    	    	        d1 = dx*d2;
    	    	        d2 = d2*(zeta-1);

    	    	        for(i=0;i<d;i++){
    	    	            for(j=0;j<d;j++){
    	    	                if(i==j){
                                    dval=1.0;
                                } else {
    	    	        	        dval=0;
                                }

                                dddx1 = x2[jj*d+i]*_x1x2_norm - x1[ii*d+i]*x1x2_dot_norm31;
                                dddx2 = x1[ii*d+j]*_x1x2_norm - x2[jj*d+j]*x1x2_dot_norm13;

                                dDdx1 = zeta * d1 * dddx1;
                                dDdx2 = zeta * d1 * dddx2;

    	    	                d2d_dx1dx2 = (dval-x2[jj*d+i]*x2[jj*d+j]*_x2_norm2)*_x1x2_norm +
    	    	                (x1[ii*d+i]*x2[jj*d+j]*x1x2_dot_norm02-x1[ii*d+i]*x1[ii*d+j])*_x1x2_norm31;

                                d2Ddx1dx2 = zeta * (d1*d2d_dx1dx2 + d2 * dddx1 * dddx2);

                                d2k_dx1dx2[i*d+j] = dK_dD * (d2Ddx1dx2 + ((1/(2*l2)) * dDdx1 * dDdx2)); 
    	    	        	}
    	    	        }
                        
                                

                        C1 = C2 = C3 = C4 = C5 = C6 = C7 = C8 = C9 = 0;
                        for(i=0;i<d;i++){
                            C[i*3+0] = 0;
                            C[i*3+1] = 0;
                            C[i*3+2] = 0;
                            for(j=0;j<d;j++){
                                dval = d2k_dx1dx2[j*d+i];
                                C[i*3+0] += dx1dr[(ii*d+j)*3 + 0] * dval;
                                C[i*3+1] += dx1dr[(ii*d+j)*3 + 1] * dval;
                                C[i*3+2] += dx1dr[(ii*d+j)*3 + 2] * dval;
                            }
                        }
                        
                        for(j=0;j<d;j++){
                                C1 += C[j*3+0]*dx2dr[(jj*d+j)*3 + 0];
                                C2 += C[j*3+1]*dx2dr[(jj*d+j)*3 + 0];
                                C3 += C[j*3+2]*dx2dr[(jj*d+j)*3 + 0];
                                C4 += C[j*3+0]*dx2dr[(jj*d+j)*3 + 1];
                                C5 += C[j*3+1]*dx2dr[(jj*d+j)*3 + 1];
                                C6 += C[j*3+2]*dx2dr[(jj*d+j)*3 + 1];
                                C7 += C[j*3+0]*dx2dr[(jj*d+j)*3 + 2];
                                C8 += C[j*3+1]*dx2dr[(jj*d+j)*3 + 2];
                                C9 += C[j*3+2]*dx2dr[(jj*d+j)*3 + 2];
                        }
                        
                        pout[((0+_i*3)*x2i+_j)*3+0] += C1;
                        pout[((1+_i*3)*x2i+_j)*3+0] += C2;
                        pout[((2+_i*3)*x2i+_j)*3+0] += C3;
                        pout[((0+_i*3)*x2i+_j)*3+1] += C4;
                        pout[((1+_i*3)*x2i+_j)*3+1] += C5;
                        pout[((2+_i*3)*x2i+_j)*3+1] += C6;
                        pout[((0+_i*3)*x2i+_j)*3+2] += C7;
                        pout[((1+_i*3)*x2i+_j)*3+2] += C8;
                        pout[((2+_i*3)*x2i+_j)*3+2] += C9;
                    }
    	    	} 
            }
    	    }
    }

    delete d2k_dx1dx2;
    delete C;
    
};

extern "C"
void rbf_kff_many_with_grad(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta, double sigma2, double l, 
                        double* x1, double* dx1dr, int* ele1, int* x1_inds, 
                        double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout, double* dpout_dl){
    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x2_norm2, _x1x2_norm, x1x2_dot;
    double _x1x2_norm31, x1x2_dot_norm02, x1x2_dot_norm13, x1x2_dot_norm31;
    double  dx, d2, d1, d2d_dx1dx2;
    double dddx1, dddx2, dDdx1, dDdx2, d2Ddx1dx2;
    double D, K, dK_dD;
    double  C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18;
    int i, j, ii, jj, _i, _j, _ele1, _ele2;
    double *dDdx1_dDdx2, *d2k_dx1dx2, *C, *Cl;
    double _l, _l3, l2;

    _l = 1 / l;
    _l3 = _l / (l * l);
    l2 = l*l;
    
    dDdx1_dDdx2 = new double[d*d];
    d2k_dx1dx2 = new double[d*d];
    C = new double[d*3];
    Cl = new double[d*3];

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];

    	dval=0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);

    	if(x1_norm > eps){

    	    for(jj=n2_start; jj<n2_end; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];

    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);

    	    	if(_ele1==_ele2 && x2_norm > eps){
    	    	    _x2_norm2 = 1.0/(x2_norm*x2_norm);
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
                    _x1x2_norm31 = _x1x2_norm/(x1_norm*x1_norm);

                    x1x2_dot=0;
    	    	    for(i=0;i<d;i++){
    	    		    x1x2_dot+=x1[ii*d+i]*x2[jj*d+i];
    	    		}
                    dx = x1x2_dot * _x1x2_norm;
                    D = pow(dx, zeta);
                    K = sigma2 * exp(-(1-D)/(2*l2));
                    dK_dD = K / (2*l2);

                    x1x2_dot_norm31 = x1x2_dot*_x1x2_norm31;
                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm*_x2_norm2;
                    x1x2_dot_norm02 = x1x2_dot*_x2_norm2;

                    d2 = pow(dx,zeta-2);
    	    	    d1 = dx*d2;
    	    	    d2 = d2*(zeta-1);

    	    	    for(i=0;i<d;i++){
    	    	        for(j=0;j<d;j++){
    	    	            if(i==j){
                                dval=1.0;
                            } else {
    	    	    	        dval=0;
                            }

                            dddx1 = x2[jj*d+i]*_x1x2_norm - x1[ii*d+i]*x1x2_dot_norm31;
                            dddx2 = x1[ii*d+j]*_x1x2_norm - x2[jj*d+j]*x1x2_dot_norm13;

                            dDdx1 = zeta * d1 * dddx1;
                            dDdx2 = zeta * d1 * dddx2;
                            
    	    	            d2d_dx1dx2 = (dval-x2[jj*d+i]*x2[jj*d+j]*_x2_norm2)*_x1x2_norm +
    	    	            (x1[ii*d+i]*x2[jj*d+j]*x1x2_dot_norm02-x1[ii*d+i]*x1[ii*d+j])*_x1x2_norm31;

                            d2Ddx1dx2 = zeta * (d1*d2d_dx1dx2 + d2 * dddx1 * dddx2);
                            dDdx1_dDdx2[i*d+j] = (1/(2*l2)) * dDdx1 * dDdx2;

                            d2k_dx1dx2[i*d+j] = dK_dD * (d2Ddx1dx2 + dDdx1_dDdx2[i*d+j]);
    	    	    	}
    	    	    }
                    
                    C1 = C2 = C3 = C4 = C5 = C6 = C7 = C8 = C9 = 0;
                    C10 = C11 = C12 = C13 = C14 = C15 = C16 = C17 = C18 = 0;
                    for(i=0;i<d;i++){
                        C[i*3+0] = 0;
                        C[i*3+1] = 0;
                        C[i*3+2] = 0;
                        Cl[i*3+0] = 0;
                        Cl[i*3+1] = 0;
                        Cl[i*3+2] = 0;

                        for(j=0;j<d;j++){
                            dval = d2k_dx1dx2[j*d+i];
                            C[i*3+0] += dx1dr[(ii*d+j)*3 + 0] * dval;
                            C[i*3+1] += dx1dr[(ii*d+j)*3 + 1] * dval;
                            C[i*3+2] += dx1dr[(ii*d+j)*3 + 2] * dval;
                            dval = dDdx1_dDdx2[j*d+i];
                            Cl[i*3+0] += dx1dr[(ii*d+j)*3 + 0] * dval;
                            Cl[i*3+1] += dx1dr[(ii*d+j)*3 + 1] * dval;
                            Cl[i*3+2] += dx1dr[(ii*d+j)*3 + 2] * dval;
                        }
                    }
                    
                    for(j=0;j<d;j++){
                        C1 += C[j*3+0]*dx2dr[(jj*d+j)*3 + 0];
                        C2 += C[j*3+1]*dx2dr[(jj*d+j)*3 + 0];
                        C3 += C[j*3+2]*dx2dr[(jj*d+j)*3 + 0];
                        C4 += C[j*3+0]*dx2dr[(jj*d+j)*3 + 1];
                        C5 += C[j*3+1]*dx2dr[(jj*d+j)*3 + 1];
                        C6 += C[j*3+2]*dx2dr[(jj*d+j)*3 + 1];
                        C7 += C[j*3+0]*dx2dr[(jj*d+j)*3 + 2];
                        C8 += C[j*3+1]*dx2dr[(jj*d+j)*3 + 2];
                        C9 += C[j*3+2]*dx2dr[(jj*d+j)*3 + 2];

                        C10 += Cl[j*3+0]*dx2dr[(jj*d+j)*3 + 0];
                        C11 += Cl[j*3+1]*dx2dr[(jj*d+j)*3 + 0];
                        C12 += Cl[j*3+2]*dx2dr[(jj*d+j)*3 + 0];
                        C13 += Cl[j*3+0]*dx2dr[(jj*d+j)*3 + 1];
                        C14 += Cl[j*3+1]*dx2dr[(jj*d+j)*3 + 1];
                        C15 += Cl[j*3+2]*dx2dr[(jj*d+j)*3 + 1];
                        C16 += Cl[j*3+0]*dx2dr[(jj*d+j)*3 + 2];
                        C17 += Cl[j*3+1]*dx2dr[(jj*d+j)*3 + 2];
                        C18 += Cl[j*3+2]*dx2dr[(jj*d+j)*3 + 2];
                    }
                    
                    pout[((0+_i*3)*x2i+_j)*3+0] += C1;
                    pout[((1+_i*3)*x2i+_j)*3+0] += C2;
                    pout[((2+_i*3)*x2i+_j)*3+0] += C3;
                    pout[((0+_i*3)*x2i+_j)*3+1] += C4;
                    pout[((1+_i*3)*x2i+_j)*3+1] += C5;
                    pout[((2+_i*3)*x2i+_j)*3+1] += C6;
                    pout[((0+_i*3)*x2i+_j)*3+2] += C7;
                    pout[((1+_i*3)*x2i+_j)*3+2] += C8;
                    pout[((2+_i*3)*x2i+_j)*3+2] += C9;

                    dpout_dl[((0+_i*3)*x2i+_j)*3+0] -= (C1 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C10 * dK_dD);
                    dpout_dl[((1+_i*3)*x2i+_j)*3+0] -= (C2 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C11 * dK_dD);
                    dpout_dl[((2+_i*3)*x2i+_j)*3+0] -= (C3 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C12 * dK_dD);
                    dpout_dl[((0+_i*3)*x2i+_j)*3+1] -= (C4 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C13 * dK_dD);
                    dpout_dl[((1+_i*3)*x2i+_j)*3+1] -= (C5 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C14 * dK_dD);
                    dpout_dl[((2+_i*3)*x2i+_j)*3+1] -= (C6 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C15 * dK_dD);
                    dpout_dl[((0+_i*3)*x2i+_j)*3+2] -= (C7 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C16 * dK_dD);
                    dpout_dl[((1+_i*3)*x2i+_j)*3+2] -= (C8 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C17 * dK_dD);
                    dpout_dl[((2+_i*3)*x2i+_j)*3+2] -= (C9 * ((D-1) * _l3 + 2 * _l) + (_l*2) * C18 * dK_dD);
                        
    	    	} 
            }
    	    }
    }
    delete dDdx1_dDdx2;
    delete d2k_dx1dx2;
    delete C;
    
};

extern "C"
void rbf_kff_many_stress(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta, double sigma2, double l2, double tol,
                     double* x1, double* dx1dr, int* ele1, int* x1_inds, 
                     double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){
    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x2_norm2, _x1x2_norm, x1x2_dot;
    double _x1x2_norm31, x1x2_dot_norm02, x1x2_dot_norm13, x1x2_dot_norm31;
    double  dx, d2, d1, d2d_dx1dx2;
    double dddx1, dddx2, dDdx1, dDdx2, d2Ddx1dx2;
    double D, K, dK_dD;
    double  C1, C2, C3, C4, C5, C6, C7, C8, C9;
    double  C10, C11, C12, C13, C14, C15, C16, C17, C18;
    double  C19, C20, C21, C22, C23, C24, C25, C26, C27;
    int i, j, ii, jj, _i, _j, _ele1, _ele2;
    double *d2k_dx1dx2, *C;

    d2k_dx1dx2=new double[d*d];
    C=new double[d*9];

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];

    	dval=0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);

    	if(x1_norm > eps){

    	    for(jj=n2_start; jj<n2_end; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];

    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);

    	    	if(_ele1==_ele2 && x2_norm > eps){
    	    	    _x2_norm2 = 1.0/(x2_norm*x2_norm);
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
                    _x1x2_norm31 = _x1x2_norm/(x1_norm*x1_norm);

                    x1x2_dot=0;
    	    	    for(i=0;i<d;i++){
    	    		    x1x2_dot+=x1[ii*d+i]*x2[jj*d+i];
    	    		}
                    dx = x1x2_dot * _x1x2_norm;
                    D = pow(dx, zeta);
                    K = sigma2 * exp(-(1-D)/(2*l2));
                    dK_dD = K / (2*l2);

                    if(dK_dD>tol){

                        x1x2_dot_norm31 = x1x2_dot*_x1x2_norm31;
                        x1x2_dot_norm13 = x1x2_dot*_x1x2_norm*_x2_norm2;
                        x1x2_dot_norm02 = x1x2_dot*_x2_norm2;

                        d2 = pow(dx,zeta-2);
    	    	        d1 = dx*d2;
    	    	        d2 = d2*(zeta-1);

    	    	        for(i=0;i<d;i++){
    	    	            for(j=0;j<d;j++){
    	    	                if(i==j){
                                    dval=1.0;
                                } else {
    	    	        	        dval=0;
                                }
                                dddx1 = x2[jj*d+i]*_x1x2_norm - x1[ii*d+i]*x1x2_dot_norm31;
                                dddx2 = x1[ii*d+j]*_x1x2_norm - x2[jj*d+j]*x1x2_dot_norm13;

                                dDdx1 = zeta * d1 * dddx1;
                                dDdx2 = zeta * d1 * dddx2;

    	    	                d2d_dx1dx2 = (dval-x2[jj*d+i]*x2[jj*d+j]*_x2_norm2)*_x1x2_norm +
    	    	                (x1[ii*d+i]*x2[jj*d+j]*x1x2_dot_norm02-x1[ii*d+i]*x1[ii*d+j])*_x1x2_norm31;
                                
                                d2Ddx1dx2 = zeta * (d1*d2d_dx1dx2 + d2 * dddx1 * dddx2);

                                d2k_dx1dx2[i*d+j] = dK_dD * (d2Ddx1dx2 + ((1/(2*l2)) * dDdx1 * dDdx2)); 
    	    	        	}
    	    	        }

                        for(i=0;i<d;i++){
                            C[i*9+0] = 0;
                            C[i*9+1] = 0;
                            C[i*9+2] = 0;
                            C[i*9+3] = 0;
                            C[i*9+4] = 0;
                            C[i*9+5] = 0;
                            C[i*9+6] = 0;
                            C[i*9+7] = 0;
                            C[i*9+8] = 0;
                            for(j=0;j<d;j++){
                                dval = d2k_dx1dx2[j*d+i];
                                C[i*9+0] += dx1dr[(ii*d+j)*9 + 0] * dval;
                                C[i*9+1] += dx1dr[(ii*d+j)*9 + 1] * dval;
                                C[i*9+2] += dx1dr[(ii*d+j)*9 + 2] * dval;
                                C[i*9+3] += dx1dr[(ii*d+j)*9 + 3] * dval;
                                C[i*9+4] += dx1dr[(ii*d+j)*9 + 4] * dval;
                                C[i*9+5] += dx1dr[(ii*d+j)*9 + 5] * dval;
                                C[i*9+6] += dx1dr[(ii*d+j)*9 + 6] * dval;
                                C[i*9+7] += dx1dr[(ii*d+j)*9 + 7] * dval;
                                C[i*9+8] += dx1dr[(ii*d+j)*9 + 8] * dval;
                            }
                        }
                        
                        C1 = C2 = C3 = C4 = C5 = C6 = C7 = C8 = C9 = 0;
                        C10 = C11 = C12 = C13 = C14 = C15 = C16 = C17 = C18 = 0;
                        C19 = C20 = C21 = C22 = C23 = C24 = C25 = C26 = C27 = 0;

                        for(j=0;j<d;j++){
                                C1 +=  C[j*9+0] * dx2dr[(jj*d+j)*3 + 0];
                                C2 +=  C[j*9+1] * dx2dr[(jj*d+j)*3 + 0];
                                C3 +=  C[j*9+2] * dx2dr[(jj*d+j)*3 + 0];
                                C4 +=  C[j*9+3] * dx2dr[(jj*d+j)*3 + 0];
                                C5 +=  C[j*9+4] * dx2dr[(jj*d+j)*3 + 0];
                                C6 +=  C[j*9+5] * dx2dr[(jj*d+j)*3 + 0];
                                C7 +=  C[j*9+6] * dx2dr[(jj*d+j)*3 + 0];
                                C8 +=  C[j*9+7] * dx2dr[(jj*d+j)*3 + 0];
                                C9 +=  C[j*9+8] * dx2dr[(jj*d+j)*3 + 0];
                                C10 += C[j*9+0] * dx2dr[(jj*d+j)*3 + 1];
                                C11 += C[j*9+1] * dx2dr[(jj*d+j)*3 + 1];
                                C12 += C[j*9+2] * dx2dr[(jj*d+j)*3 + 1];
                                C13 += C[j*9+3] * dx2dr[(jj*d+j)*3 + 1];
                                C14 += C[j*9+4] * dx2dr[(jj*d+j)*3 + 1];
                                C15 += C[j*9+5] * dx2dr[(jj*d+j)*3 + 1];
                                C16 += C[j*9+6] * dx2dr[(jj*d+j)*3 + 1];
                                C17 += C[j*9+7] * dx2dr[(jj*d+j)*3 + 1];
                                C18 += C[j*9+8] * dx2dr[(jj*d+j)*3 + 1];
                                C19 += C[j*9+0] * dx2dr[(jj*d+j)*3 + 2];
                                C20 += C[j*9+1] * dx2dr[(jj*d+j)*3 + 2];
                                C21 += C[j*9+2] * dx2dr[(jj*d+j)*3 + 2];
                                C22 += C[j*9+3] * dx2dr[(jj*d+j)*3 + 2];
                                C23 += C[j*9+4] * dx2dr[(jj*d+j)*3 + 2];
                                C24 += C[j*9+5] * dx2dr[(jj*d+j)*3 + 2];
                                C25 += C[j*9+6] * dx2dr[(jj*d+j)*3 + 2];
                                C26 += C[j*9+7] * dx2dr[(jj*d+j)*3 + 2];
                                C27 += C[j*9+8] * dx2dr[(jj*d+j)*3 + 2];
                        }

                        pout[((0+_i*9)*x2i+_j)*3+0] += C1;
                        pout[((1+_i*9)*x2i+_j)*3+0] += C2;
                        pout[((2+_i*9)*x2i+_j)*3+0] += C3;
                        pout[((3+_i*9)*x2i+_j)*3+0] += C4;
                        pout[((4+_i*9)*x2i+_j)*3+0] += C5;
                        pout[((5+_i*9)*x2i+_j)*3+0] += C6;
                        pout[((6+_i*9)*x2i+_j)*3+0] += C7;
                        pout[((7+_i*9)*x2i+_j)*3+0] += C8;
                        pout[((8+_i*9)*x2i+_j)*3+0] += C9;
                        pout[((0+_i*9)*x2i+_j)*3+1] += C10;
                        pout[((1+_i*9)*x2i+_j)*3+1] += C11;
                        pout[((2+_i*9)*x2i+_j)*3+1] += C12;
                        pout[((3+_i*9)*x2i+_j)*3+1] += C13;
                        pout[((4+_i*9)*x2i+_j)*3+1] += C14;
                        pout[((5+_i*9)*x2i+_j)*3+1] += C15;
                        pout[((6+_i*9)*x2i+_j)*3+1] += C16;
                        pout[((7+_i*9)*x2i+_j)*3+1] += C17;
                        pout[((8+_i*9)*x2i+_j)*3+1] += C18;
                        pout[((0+_i*9)*x2i+_j)*3+2] += C19;
                        pout[((1+_i*9)*x2i+_j)*3+2] += C20;
                        pout[((2+_i*9)*x2i+_j)*3+2] += C21;
                        pout[((3+_i*9)*x2i+_j)*3+2] += C22;
                        pout[((4+_i*9)*x2i+_j)*3+2] += C23;
                        pout[((5+_i*9)*x2i+_j)*3+2] += C24;
                        pout[((6+_i*9)*x2i+_j)*3+2] += C25;
                        pout[((7+_i*9)*x2i+_j)*3+2] += C26;
                        pout[((8+_i*9)*x2i+_j)*3+2] += C27;
                    }       
    	    	} 
    	    }
    	}
    }
    delete d2k_dx1dx2;
    delete C;
};


int main(){
    return 0;
}
