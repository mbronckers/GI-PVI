import lab as B
from plum import convert
from matrix import AbstractMatrix, Diagonal, structured
import torch

class Normal:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = convert(var, AbstractMatrix)
        
    @classmethod
    def from_naturalnormal(cls, dist):
        return cls(mean=B.mm(B.pd_inv(dist.prec), dist.lam), var=B.pd_inv(dist.prec))

    def kl(self, other: "Normal"):
        """Compute the KL divergence with respect to another normal
        distribution.
        Args:
            other (:class:`.random.Normal`): Other normal.
        Returns:
            scalar: KL divergence.
        """
        return (
            B.iqf_diag(other.var, other.mean - self.mean)[..., 0]
            + B.ratio(self.var, other.var)
            + B.logdet(other.var)
            - B.logdet(self.var)
            - B.cast(self.dtype, self.dim)
        ) / 2
    
    def __eq__(self, __o: "Normal") -> bool:
        return (torch.all(torch.isclose(B.dense(self.mean), B.dense(__o.mean))) and torch.all(torch.isclose(B.dense(self.var), B.dense(__o.var)))).item()

class NaturalNormal:
    def __init__(self, lam, prec):
        """
        :param lam: first natural parameter of Normal dist = precision x mean
        :param prec: second natural parameter of Normal dist = -0.5 x precision \\propto precision 
        """
        self.lam = lam 
        self.prec = convert(prec, AbstractMatrix)
    
    @classmethod
    def from_normal(cls, dist):
        """
        Convert class:Normal into class:NaturalNormal
        - \\eta = [\\Sigma_inv \\mu, -0.5 \\Sigma_inv]^T
        
        """
        return cls(B.mm(B.pd_inv(dist.var), dist.mean), B.pd_inv(dist.var))
    
    def kl(self, other: "NaturalNormal"):
        """Compute the Kullback-Leibler divergence with respect to another normal
        parametrised by its natural parameters.
        Args:
            other (:class:`.NaturalNormal`): Other.
        Returns:
            scalar: KL divergence with respect to `other`.
        """
        ratio = B.solve(B.chol(self.prec), B.chol(other.prec))
        diff = self.mean - other.mean
        return 0.5 * (
            B.sum(ratio**2)
            - B.logdet(B.mm(ratio, ratio, tr_a=True))
            + B.sum(B.mm(other.prec, diff) * diff)
            - B.cast(self.dtype, self.dim)
        )
    
    def sample(self, key: B.RandomState, num: B.Int = 1):
        """
        Sample from distribution using the natural parameters
        """
        key, noise = B.randn(key, B.dtype(self.lam), num, *B.shape(self.lam)) # Sample our noise (epsilon)
        sample = B.mm(B.inv(self.prec), self.lam) + B.triangular_solve(B.cholesky(self.prec), noise)
        
        if not structured(sample):
            sample = B.dense(sample)
        
        return key, sample
        
    def __mul__(self, other: "NaturalNormal"):
        return NaturalNormal(
            self.lam + other.lam, 
            self.prec + other.prec
            )

    def __eq__(self, __o: "NaturalNormal") -> bool:
        return (torch.all(torch.isclose(self.lam, __o.lam)) and torch.all(torch.isclose(self.prec, __o.prec))).item()

    
class NormalPseudoObservation:
    def __init__(self, yz, nz):
        """
        :param yz: inducing point observation (pseudo-observations)
        :param nz: inducing point noise
        """
        self.yz = yz
        self.nz = nz
        
    def __call__(self, z):

        """
        :param z: inducing inputs of that layer which are equal to the outputs of the prev layer inducing inputs, i.e. phi(U_{\\ell-1}) [samples x M x Din]
        """
        # (Dout, M, M).
        # prec_yv = 
        prec_yv = torch.diag(self.nz)
        # (S, Dout, Din, Din).
        prec_w = torch.unsqueeze(z.transpose(-1, -2), 1) @ torch.unsqueeze(prec_yv, 0) @ torch.unsqueeze(z, 1) # [ S x 1 x Din x M ] @ [ 1 x Dout x M x M ] @ [S x 1 x M x Din] = [ S x Dout x Din x Din ]
        lam_w = torch.unsqueeze(z.transpose(-1, -2), 1) @ torch.unsqueeze(prec_yv, 0) @ torch.unsqueeze(torch.unsqueeze(self.yz, 0), -1) # [ S x 1 x Din x M ] @ [ 1 x Dout x M x M ] @ [ 1 x Dout x M x 1 ]
        return NaturalNormal(lam_w, prec_w)