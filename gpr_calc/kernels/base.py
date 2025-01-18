import numpy as np

def build_covariance(c_ee, c_ef, c_fe, c_ff, c_se=None, c_sf=None):
    """
    Need to rework
    """
    exist = []
    for x in (c_ee, c_ef, c_fe, c_ff):
        if x is None:
            exist.append(False)
        else:
            exist.append(True)
    if False not in exist:
        return np.block([[c_ee, c_ef], [c_fe, c_ff]])
    elif exist == [False, False, True, True]: # F in train, E/F in predict
        return np.hstack((c_fe, c_ff))
    elif exist == [True, True, False, False]: # E in train, E/F in predict
        return np.hstack((c_ee, c_ef))
    elif exist == [False, True, False, False]: # E in train, F in predict
        return c_ef
    elif exist == [True, False, False, False]: # E in train, E in predict
        return c_ee
    elif exist == [False, False, False, True]: # F in train, F in predict 
        return c_ff
    elif exist == [False, False, True, False]: # F in train, E in predict 
        return c_fe

def get_mask(ele1, ele2):
    ans = ele1[:,None] - ele2[None,:]
    ids = np.where(ans!=0)
    if len(ids[0]) == 0:
        return None
    else:
        return ids



# ======================= Functions related to d and D ===================================
#def fun_D(x1, x2, x1_norm, x2_norm, zeta=2, eps=1e-6):
#    d = x1@x2.T/(eps+np.outer(x1_norm, x2_norm))
#    D = d**zeta
#    return D, d
#
#def fun_dd_dx1(x1, x2, x1_norm, x2_norm):  
#    # x1: m,d
#    # x2: n,d
#    tmp1 = np.einsum("ij,k->kij", x2, x1_norm) # [n,d] x [m] -> [m, n, d]
#    tmp2 = (x1@x2.T)[:,:,None] * (x1/x1_norm[:, None])[:,None,:] #[m, n, d]
#    tmp3 = (x1_norm**2)[:, None, None] * x2_norm[None, :, None]  #[m, n, d]
#    out = (tmp1-tmp2)/tmp3  # m,n,d
#    return out, out.sum(axis=1)
#
#def fun_dd_dx2(x1, x2, x1_norm, x2_norm, eps=1e-6):
#    tmp1 = np.einsum("ij,k->ikj", x1, x2_norm) # [m,d] x [n] -> [m, n, d]
#    tmp2 = (x1@x2.T)[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
#    tmp3 = x1_norm[:, None, None] * (x2_norm**2)[None, :, None] + eps
#    out = (tmp1-tmp2)/tmp3  # m,n,d
#    return out, out.sum(axis=0)
#
#def fun_d2d_dx1dx2(x1, x2, x1_norm, x2_norm):
#    x1_norm3 = x1_norm**3        
#    x2_norm3 = x2_norm**3      
#    x2_norm2 = x2_norm**2      
#    
#    tmp0 = np.ones(x2.shape)
#    tmp1 = (x1[:,None,:]*(tmp0/x2_norm[:,None])[None,:,:])[:,:,None,:]*(x1/x1_norm3[:,None])[:,None,:,None]
#
#    x1x2 = (x1/x1_norm3[:,None])[:,None,:,None]*(x2/x2_norm3[:,None])[None,:,None,:]
#    tmp2 = x1x2 * (x1@x2.T)[:,:,None,None]
#    
#    out1 = tmp1-tmp2
#    tmp3 = np.eye(x2.shape[1])[None,:,:] - np.einsum('ij,ik->ijk',x2,x2/x2_norm2[:,None]) # n*d1*d2
#    out2 = tmp3[None,:,:,:]/(x1_norm[:,None]*x2_norm[None,:])[:,:,None,None]
#    out = out2 - out1
#    return out
#
#def fun_dD_dx1(x1, x2, x1_norm, x2_norm, d, zeta=2):
#    out, _ = fun_dd_dx1(x1, x2, x1_norm, x2_norm)
#    dD_dx1 = np.einsum("ij, ijk->ijk", zeta*d**(zeta-1), out) #m,n;  m,n,d -> m,d
#    return dD_dx1, dD_dx1.sum(axis=1)
#
#def fun_dD_dx2(x1, x2, x1_norm, x2_norm, d, zeta=2):
#    out, _ = fun_dd_dx2(x1, x2, x1_norm, x2_norm)
#    dD_dx2 = np.einsum("ij, ijk->ijk", zeta*d**(zeta-1), out) # -> n,d
#    return dD_dx2, dD_dx2.sum(axis=0)
#
#def fun_d2D_dx1dx2(x1, x2, x1_norm, x2_norm, d, zeta=2):
#
#    d2d_dx1dx2 = fun_d2d_dx1dx2(x1, x2, x1_norm, x2_norm) #[m,n,d1,d2]
#    dd_dx1, _ = fun_dd_dx1(x1, x2, x1_norm, x2_norm) # [m, n, d1]
#    dd_dx2, _ = fun_dd_dx2(x1, x2, x1_norm, x2_norm) # [m, n, d2]
#    d2D_dx1dx2 = np.einsum('ijk, ijl->ijkl', dd_dx1, dd_dx2)
#    d2D_dx1dx2 = np.einsum('ij, ijkl->ijkl', (zeta-1)*d**(zeta-2), d2D_dx1dx2)
#    d2D_dx1dx2 += np.einsum("ij, ijkl->ijkl", d**(zeta-1), d2d_dx1dx2)
#    dD_dx1 = np.einsum("ij, ijk->ijk", zeta*d**(zeta-1), dd_dx1) 
#    dD_dx2 = np.einsum("ij, ijk->ijk", zeta*d**(zeta-1), dd_dx2)
#
#    return zeta*d2D_dx1dx2, (dD_dx1, dD_dx2)

# ===================== Side functions =========================

#if __name__ == "__main__":
#   
#    import torch
#    print("numerical")
#    def d_torch(x1, x2):
#        d=x1@x2.T/(torch.ger(torch.norm(x1, dim=1), torch.norm(x2, dim=1)))
#        return d
#    def D_torch(x1, x2, zeta=2):
#        return d_torch(x1, x2)**zeta
#    
#    def test_fun(x1, x2, x1_norm, x2_norm, D1, d1, zeta=2, target="d"):
#        if target == "d":
#            fun_df_dx1 = fun_dd_dx1
#            fun_df_dx2 = fun_dd_dx2
#            fun_d2f_dx1dx2 = fun_d2d_dx1dx2
#            func = d_torch
#            _, df_dx1_np = fun_df_dx1(x1, x2, x1_norm, x2_norm)
#            _, df_dx2_np = fun_df_dx2(x1, x2, x1_norm, x2_norm)
#            d2f_dx1dx2_np = fun_d2f_dx1dx2(x1, x2, x1_norm, x2_norm)
#
#        elif target == 'D':
#            fun_df_dx1 = fun_dD_dx1
#            fun_df_dx2 = fun_dD_dx2
#            fun_d2f_dx1dx2 = fun_d2D_dx1dx2
#            func = D_torch
#            _, df_dx1_np = fun_df_dx1(x1, x2, x1_norm, x2_norm, d1, zeta)
#            _, df_dx2_np = fun_df_dx2(x1, x2, x1_norm, x2_norm, d1, zeta)
#            d2f_dx1dx2_np, _ = fun_d2f_dx1dx2(x1, x2, x1_norm, x2_norm, d1, zeta)
#
#        print("testing ", target)
#           
#
#        print("df_dx1 from np")
#        print(df_dx1_np)
#        print("df_dx2 from np")
#        print(df_dx2_np)
#        print("d2f_dx1dx2 from np")
#        print(np.transpose(d2f_dx1dx2_np, axes=(0,1,3,2)))
# 
#        t_x1 = torch.tensor(x1, requires_grad=True)
#        t_x2 = torch.tensor(x2, requires_grad=True)
#        if target == 'd':
#            d = func(t_x1, t_x2).sum()
#        elif target == "D":
#            d = func(t_x1, t_x2, zeta).sum()
#        else:
#            d = func(t_x1, t_x2, sigma2, l2, zeta).sum()
#
#        print("df_dx1")
#        print(torch.autograd.grad(d, t_x1, retain_graph=True)[0].numpy())    
#        print("df_dx2")
#        print(torch.autograd.grad(d, t_x2)[0].numpy())    
#        t_x1 = torch.tensor(x1, requires_grad=True)
#        t_x2 = torch.tensor(x2, requires_grad=True)
#        print("d2f_dx1dx2")
#        eps = 1e-6
#        for i in range(t_x2.size()[0]):
#            for j in range(t_x2.size()[1]):
#                tmp = t_x2.clone()
#                tmp[i, j] += eps
#                if target in ['d', 'D']:
#                    d1 = func(t_x1, t_x2).sum()
#                    d2 = func(t_x1, tmp).sum()
#                else:
#                    d1 = func(t_x1, t_x2, sigma2, l2).sum()
#                    d2 = func(t_x1, tmp, sigma2, l2).sum()
#                    
#                grad1 = torch.autograd.grad(d1, t_x1)
#                grad2 = torch.autograd.grad(d2, t_x1)
#                print(((grad2[0]-grad1[0])/eps).numpy())
#
#    m, n, k, sigma2, l2, zeta = 2, 2, 3, 0.81, 0.01, 2
#    x1 = np.random.random([m, k])
#    x2 = np.random.random([n, k])
#    x1_norm = np.linalg.norm(x1, axis=1)
#    x2_norm = np.linalg.norm(x2, axis=1)
#    D1, d1 = fun_D(x1, x2, x1_norm, x2_norm)
#
#    test_fun(x1, x2, x1_norm, x2_norm, D1, d1, zeta, "d")
#    test_fun(x1, x2, x1_norm, x2_norm, D1, d1, zeta, "D")
