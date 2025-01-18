#include "dot_kernel.h"
#include <cmath>
#include <iostream>

extern "C"
void dot_kee_many(int n1, int n2, int d, int x2i, double zeta, double sigma2, double sigma02,
            double* x1, int* ele1, int* x1_inds, 
            double* x2, int* ele2, int* x2_inds, double* pout){

    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x1x2_norm, x1x2_dot; 
    double dx, D;
    int i, ii, jj, _i, _j, _ele1, _ele2;

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];
    
    	dval=0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);
    
    	if(x1_norm > eps){
    
    	    for(jj=0; jj<n2; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];
    
    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);
    
    	    	if(_ele1==_ele2 && x2_norm > eps){
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
   
    	    	    x1x2_dot=0;
    	    	    for(i=0;i<d;i++){
    	    		    x1x2_dot+=x1[ii*d+i]*x2[jj*d+i];
    	    		}
    
                    dx = x1x2_dot*_x1x2_norm;
    	    	    D = pow(dx,zeta);
                    pout[_i*x2i+_j] += sigma2*(D+sigma02);

                    //std::cout << dx << ", " << D <<", " << _j << "\n";
    	    	} //if(_ele1==_ele2 && x2_norm > eps)
    	    }//for(jj=0;jj<n;jj++)
    	}//if(x1_norm > eps){
    }//for(ii=0;ii<n;ii++)
    
};

extern "C"
void dot_kef_many(int n1, int n2, int d, int x2i, double zeta, 
            double* x1, int* ele1, int* x1_inds, 
            double* x2, double* dx2dr, int* ele2, int* x2_inds, 
            double* pout){

    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x1x2_norm, x1x2_dot, x1x2_dot_norm13;
    double  dx, d1, dd_dx2;
    double  C1, C2, C3;
    int i, ii, jj, _i, _j, _ele1, _ele2;
    double *dk_dx2;

    dk_dx2=new double[d];

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];
    
    	dval=0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);
    
    	if(x1_norm > eps){
    
    	    for(jj=0; jj<n2; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];
    
    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);
    
    	    	if(_ele1==_ele2 && x2_norm > eps){
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
   
    	    	    x1x2_dot=0;
    	    	    for(i=0;i<d;i++){
    	    		    x1x2_dot+=x1[ii*d+i]*x2[jj*d+i];
    	    		}
    
                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm/(x2_norm*x2_norm);
 
                    dx = x1x2_dot*_x1x2_norm;
    	    	    d1 = zeta*pow(dx,zeta-1);
                    
                    C1 = C2 = C3 = 0;
    	    	    for(i=0;i<d;i++){
    	    	        dd_dx2 = x1[ii*d+i]*_x1x2_norm - x1x2_dot_norm13*x2[jj*d+i];
    	    	        dk_dx2[i] = d1*dd_dx2;
    	    	    }

                    for(i=0;i<d;i++){
                        C1 += dx2dr[(jj*d+i)*3 + 0] * dk_dx2[i];
                        C2 += dx2dr[(jj*d+i)*3 + 1] * dk_dx2[i];
                        C3 += dx2dr[(jj*d+i)*3 + 2] * dk_dx2[i];
                    }
                    
                    pout[(_i*x2i+_j)*3]   += C1;
                    pout[(_i*x2i+_j)*3+1] += C2;
                    pout[(_i*x2i+_j)*3+2] += C3;
                            
    	    	} //if(_ele1==_ele2 && x2_norm > eps)
    	    }//for(jj=0;jj<n;jj++)
    	    }//if(x1_norm > eps){
    }//for(ii=0;ii<n;ii++)
    delete dk_dx2;
};


extern "C"
void dot_kef_many_stress(int n1, int n2, int d, int x2i, double zeta, 
            double* x1, int* ele1, int* x1_inds, 
            double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){

    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x1x2_norm, x1x2_dot, x1x2_dot_norm13;
    double  dx, d1, dd_dx2;
    double  C1, C2, C3, C4, C5, C6, C7, C8, C9;  
    int i, ii, jj, _i, _j, _ele1, _ele2;
    double *dk_dx2;

    dk_dx2=new double[d];

    for(ii=0; ii<n1; ii++){
    	_ele1 = ele1[ii];
    	_i = x1_inds[ii];
    
    	dval=0.0;
    	for(i=0;i<d;i++){
            dval+=x1[ii*d+i]*x1[ii*d+i];
    	}
    	x1_norm=sqrt(dval);
    
    	if(x1_norm > eps){
    
    	    for(jj=0; jj<n2; jj++){
    	    	_ele2 = ele2[jj];
    	    	_j = x2_inds[jj];
    
    	    	dval=0.0;
    	    	for(i=0;i<d;i++){
                    dval+=x2[jj*d+i]*x2[jj*d+i];
    	    	}
    	    	x2_norm=sqrt(dval);
    
    	    	if(_ele1==_ele2 && x2_norm > eps){
                    _x1x2_norm = 1.0/(x1_norm*x2_norm);
   
    	    	    x1x2_dot=0;
    	    	    for(i=0;i<d;i++){
    	    		    x1x2_dot+=x1[ii*d+i]*x2[jj*d+i];
    	    		}
    
                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm/(x2_norm*x2_norm);
 
                    dx = x1x2_dot*_x1x2_norm;
    	    	    d1 = zeta*pow(dx,zeta-1);
                    
                    C1 = C2 = C3 = C4 = C5 = C6 = C7 = C8 = C9 = 0;
    	    	    for(i=0;i<d;i++){
    	    	        dd_dx2 = x1[ii*d+i]*_x1x2_norm - x1x2_dot_norm13*x2[jj*d+i];
    	    	        dk_dx2[i] = d1*dd_dx2;
    	    	    }

                    for(i=0;i<d;i++){
                        C1 += dx2dr[(jj*d+i)*9 + 0] * dk_dx2[i];
                        C2 += dx2dr[(jj*d+i)*9 + 1] * dk_dx2[i];
                        C3 += dx2dr[(jj*d+i)*9 + 2] * dk_dx2[i];
                        C4 += dx2dr[(jj*d+i)*9 + 3] * dk_dx2[i];
                        C5 += dx2dr[(jj*d+i)*9 + 4] * dk_dx2[i];
                        C6 += dx2dr[(jj*d+i)*9 + 5] * dk_dx2[i];
                        C7 += dx2dr[(jj*d+i)*9 + 6] * dk_dx2[i];
                        C8 += dx2dr[(jj*d+i)*9 + 7] * dk_dx2[i];
                        C9 += dx2dr[(jj*d+i)*9 + 8] * dk_dx2[i];
                    }
                    
                    pout[(_i*x2i+_j)*9] += C1;
                    pout[(_i*x2i+_j)*9+1] += C2;
                    pout[(_i*x2i+_j)*9+2] += C3;
                    pout[(_i*x2i+_j)*9+3] += C4;
                    pout[(_i*x2i+_j)*9+4] += C5;
                    pout[(_i*x2i+_j)*9+5] += C6;
                    pout[(_i*x2i+_j)*9+6] += C7;
                    pout[(_i*x2i+_j)*9+7] += C8;
                    pout[(_i*x2i+_j)*9+8] += C9;
                            
    	    	} //if(_ele1==_ele2 && x2_norm > eps)
    	    }//for(jj=0;jj<n;jj++)
    	    }//if(x1_norm > eps){
    }//for(ii=0;ii<n;ii++)
    delete dk_dx2;
};
extern "C"
void dot_kff_many(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta,
            double* x1, double* dx1dr, int* ele1, int* x1_inds, 
            double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){

    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x2_norm2, _x1x2_norm, x1x2_dot; 
    double _x1x2_norm31, x1x2_dot_norm02, x1x2_dot_norm13, x1x2_dot_norm31;
    double  dx, d2, d1, d2d_dx1dx2;
    double  C1, C2, C3, C4, C5, C6, C7, C8, C9;
    int i, j, ii, jj, _i, _j, _ele1, _ele2;
    double *d2k_dx1dx2, *C;

    d2k_dx1dx2=new double[d*d];
    C=new double[d*3];
    //dk_dD = sigma*sigma*zeta;

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
    
                    x1x2_dot_norm31 = x1x2_dot*_x1x2_norm31;
                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm*_x2_norm2;
                    x1x2_dot_norm02 = x1x2_dot*_x2_norm2;
 
                    dx = x1x2_dot*_x1x2_norm;
    	    	    d2 = pow(dx,zeta-2);
    	    	    d1 = dx*d2;
    	    	    d2 = d2*(zeta-1);
                    
                    C1 = C2 = C3 = C4 = C5 = C6 = C7 = C8 = C9 = 0;
    	    	    for(i=0;i<d;i++){
    	    	        for(j=0;j<d;j++){
    	    	            if(i==j){
                                dval=1.0;
                            } else {
    	    	    	        dval=0;
                            }

    	    	            d2d_dx1dx2 = (dval-x2[jj*d+i]*x2[jj*d+j]*_x2_norm2)*_x1x2_norm +
    	    	            (x1[ii*d+i]*x2[jj*d+j]*x1x2_dot_norm02-x1[ii*d+i]*x1[ii*d+j])*_x1x2_norm31;    

    	    	            d2k_dx1dx2[i*d+j]=((x2[jj*d+i]*_x1x2_norm-x1[ii*d+i]*x1x2_dot_norm31)*
    	    	        	(x1[ii*d+j]*_x1x2_norm-x2[jj*d+j]*x1x2_dot_norm13))*d2 + d1*d2d_dx1dx2;
    	    	    	}
    	    	    }

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

                            
    	    	} //if(_ele1==_ele2 && x2_norm > eps)
    	    }//for(jj=0;jj<n;jj++)
    	    }//if(x1_norm > eps){
    }//for(ii=0;ii<n;ii++)

    delete d2k_dx1dx2;
    delete C;
    
};

extern "C"
void dot_kff_many_stress(int n1, int n2, int n2_start, int n2_end, int d, int x2i, double zeta,
            double* x1, double* dx1dr, int* ele1, int* x1_inds, 
            double* x2, double* dx2dr, int* ele2, int* x2_inds, double* pout){

    double eps=1e-8;
    double dval;
    double x1_norm, x2_norm, _x2_norm2, _x1x2_norm, x1x2_dot; 
    double _x1x2_norm31, x1x2_dot_norm02, x1x2_dot_norm13, x1x2_dot_norm31;
    double  dx, d2, d1, d2d_dx1dx2;
    double  C1, C2, C3, C4, C5, C6, C7, C8, C9;
    double  C10, C11, C12, C13, C14, C15, C16, C17, C18;
    double  C19, C20, C21, C22, C23, C24, C25, C26, C27;
    int i, j, ii, jj, _i, _j, _ele1, _ele2;
    double *d2k_dx1dx2, *C;

    d2k_dx1dx2=new double[d*d];
    C=new double[d*9];
    //dk_dD = sigma*sigma*zeta;

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
    
                    x1x2_dot_norm31 = x1x2_dot*_x1x2_norm31;
                    x1x2_dot_norm13 = x1x2_dot*_x1x2_norm*_x2_norm2;
                    x1x2_dot_norm02 = x1x2_dot*_x2_norm2;
 
                    dx = x1x2_dot*_x1x2_norm;
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

    	    	            d2d_dx1dx2 = (dval-x2[jj*d+i]*x2[jj*d+j]*_x2_norm2)*_x1x2_norm +
    	    	            (x1[ii*d+i]*x2[jj*d+j]*x1x2_dot_norm02-x1[ii*d+i]*x1[ii*d+j])*_x1x2_norm31;    

    	    	            d2k_dx1dx2[i*d+j]=((x2[jj*d+i]*_x1x2_norm-x1[ii*d+i]*x1x2_dot_norm31)*
    	    	        	(x1[ii*d+j]*_x1x2_norm-x2[jj*d+j]*x1x2_dot_norm13))*d2 + d1*d2d_dx1dx2;
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
                            
    	    	} //if(_ele1==_ele2 && x2_norm > eps)
    	    }//for(jj=0;jj<n;jj++)
    	    }//if(x1_norm > eps){
    }//for(ii=0;ii<n;ii++)

    delete d2k_dx1dx2;
    delete C;
    
};

int main(){
    return 0;
}
