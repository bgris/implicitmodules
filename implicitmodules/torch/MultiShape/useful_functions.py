import torch



def kronecker_I2(K):
    K = K.contiguous()
    N = K.shape
    tmp = K.view(-1,1).repeat(1,2).view(N[0],2*N[1]).transpose(1,0).contiguous().view(-1,1).repeat(1,2).view(-1,N[0]*2).transpose(1,0)
    Ktilde = torch.mul(tmp, torch.eye(2).repeat(N))
    return Ktilde
