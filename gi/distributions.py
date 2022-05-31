import lab as B
from plum import convert
from matrix import AbstractMatrix, Diagonal, structured
from sympy import Q
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

        See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions for more info
        """
        return (
            B.iqf_diag(other.var, other.mean - self.mean)[..., 0]
            + B.ratio(self.var, other.var)
            + B.logdet(other.var)
            - B.logdet(self.var)
            - B.cast(self.dtype, self.dim) 
        ) / 2
        
    def logpdf(self, x):
        """Compute the log-pdf.
        Args:
            x (input): Values to compute the log-pdf of.
        Returns:
            list[tensor]: Log-pdf for every input in `x`. If it can be
                determined that the list contains only a single log-pdf,
                then the list is flattened to a scalar.
        """
        x = B.uprank(x)

        # Handle missing data. We don't handle missing data for batched computation.
        if B.rank(x) == 2 and B.shape(x, 1) == 1:
            available = B.jit_to_numpy(~B.isnan(x[:, 0]))
            if not B.all(available):
                # Take the elements of the mean, variance, and inputs corresponding to
                # the available data.
                available_mean = B.take(self.mean, available)
                available_var = B.submatrix(self.var, available)
                available_x = B.take(x, available)
                return Normal(available_mean, available_var).logpdf(available_x)

        logpdfs = (
            -(
                B.logdet(self.var)[..., None]  # Correctly line up with `iqf_diag`.
                + B.cast(self.dtype, self.dim) * B.cast(self.dtype, B.log_2_pi)
                + B.iqf_diag(self.var, B.subtract(x, self.mean)[..., None]) # compute diag of matrix product of (a, b) with a being PD
            )
            / 2
        )
        return logpdfs[..., 0] if B.shape(logpdfs, -1) == 1 else logpdfs
        
    @property
    def dtype(self):
        """dtype: Data type of the variance."""
        return B.default_dtype

    @property
    def dim(self):
        """int: Dimensionality."""
        return B.shape_matrix(self.var)[0]

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
        
        self._mean = None
        self._var = None
    
    @property
    def dtype(self):
        """dtype: Data type of the precision."""
        return B.dtype(self.prec)

    @property
    def dim(self):
        """int: Dimensionality."""
        return B.shape_matrix(self.prec)[0]

    @classmethod
    def from_normal(cls, dist):
        """
        Convert class:Normal into class:NaturalNormal
        - \\eta = [\\Sigma_inv \\mu, -0.5 \\Sigma_inv]^T
        
        """
        return cls(B.mm(B.pd_inv(dist.var), dist.mean), B.pd_inv(dist.var))
    
    @property
    def mean(self):
        """column vector: Mean."""
        if self._mean is None:
            self._mean = B.cholsolve(B.chol(self.prec), self.lam)
        return self._mean

    @property
    def var(self):
        """matrix: Variance."""
        if self._var is None:
            self._var = B.pd_inv(self.prec)
        return self._var
    
    def kl(self, other: "NaturalNormal"):
        """Compute the Kullback-Leibler divergence with respect to another normal
        parametrised by its natural parameters.
        Args:
            other (:class:`.NaturalNormal`): Other.
        Returns:
            scalar: KL divergence with respect to `other`.
        
        See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions for more info
        """
        ratio = B.solve(B.chol(self.prec), B.chol(other.prec))  # M in wiki
        diff = self.mean - other.mean                           # mu1 - mu0
        _kl = 0.5 * (
            B.sum(ratio**2)                                     # 
            - B.logdet(B.mm(ratio, ratio, tr_a=True))           # ratio^T @ ratio
            + B.sum(B.mm(other.prec, diff) * diff)              # (diff)^T @ prec @ diff
            - B.cast(self.dtype, self.dim)                      # subtract dimension |K| scalar
        )
        return _kl
    
    def sample(self, key: B.RandomState, num: B.Int = 1):
        """
        Sample from distribution using the natural parameters
        """
        if num > 1:
            key, noise = B.randn(key, B.default_dtype, num, *B.shape(self.lam)) # Sample our noise (epsilon)
        else:
            key, noise = B.randn(key, B.default_dtype, *B.shape(self.lam)) # Sample our noise (epsilon)
            
        sample = B.mm(B.inv(self.prec), self.lam) + B.triangular_solve(B.cholesky(self.prec), noise)
        
        if not structured(sample):
            sample = B.dense(sample) # transform Tensor to Dense matrix
        
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
        :param nz: inducing point noise (precision)
        """
        self.yz = yz    # [M x Dout]
        self.nz = nz    # [Dout x M]
        
    def __call__(self, z):

        """
        :param z: inducing inputs of that layer which are equal to the outputs of the prev layer inducing inputs, i.e. phi(U_{\\ell-1}) [samples x M x Din]
        """
        # (S, 1, M, Din)
        _z = B.expand_dims(z, 1)
        
        # (1, Dout, M, 1).
        _yz = B.expand_dims(B.transpose(self.yz, (-1, -2)))
        _yz = B.expand_dims(_yz, -1)
        
        # (Dout, M, M).
        prec_yz = B.diag_construct(self.nz)
        
        # (1, Dout, M, M).
        _prec_yz = B.expand_dims(prec_yz, 0)
        
        # (S, Dout, Din, Din).
        prec_w = B.mm(B.transpose(_z), B.mm(_prec_yz, _z))
        
        # (S, Dout, Din, 1)
        lam_w = B.mm(B.transpose(_z), B.mm(_prec_yz, _yz))
        # lam_w = B.sum(B.mm(prec_yz, self.yz) * z, -1)
        # prec_w = torch.unsqueeze(z.transpose(-1, -2), 1) @ torch.unsqueeze(prec_yv, 0) @ torch.unsqueeze(z, 1) # [ S x 1 x Din x M ] @ [ 1 x Dout x M x M ] @ [S x 1 x M x Din] = [ S x Dout x Din x Din ]
        # lam_w = torch.unsqueeze(z.transpose(-1, -2), 1) @ torch.unsqueeze(prec_yv, 0) @ torch.unsqueeze(torch.unsqueeze(self.yz, 0), -1) # [ S x 1 x Din x M ] @ [ 1 x Dout x M x M ] @ [ 1 x Dout x M x 1 ]
        return NaturalNormal(lam_w, prec_w)

    def __repr__(self) -> str:
        return f"yz: {self.yz}, \nnz: {self.nz} \n"