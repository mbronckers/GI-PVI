from copy import deepcopy
import logging
from typing import Union
import lab as B
from plum import convert
from matrix import AbstractMatrix, Diagonal, structured
import torch

logger = logging.getLogger()


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
        return (B.iqf_diag(other.var, other.mean - self.mean)[..., 0] + B.ratio(self.var, other.var) + B.logdet(other.var) - B.logdet(self.var) - B.cast(self.dtype, self.dim)) / 2

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

        if len(B.shape(x)) == 2 or len(B.shape(x)) == 3:
            logpdfs = (
                -(
                    B.logdet(self.var)[..., None]  # Correctly line up with `iqf_diag`.
                    + B.cast(self.dtype, self.dim) * B.cast(self.dtype, B.log_2_pi)
                    + B.iqf_diag(self.var, B.subtract(x, self.mean)[..., None])
                    # Compute the diagonal of `transpose(b) inv(a) c` where `a` is assumed to be PD
                    # => (x-m)T @ inv(var) @ (x-m)
                )
                / 2
            )
        elif len(B.shape(x)) == 4:
            logpdfs = (
                -(
                    B.logdet(self.var)[..., None]  # Correctly line up with `iqf_diag`.
                    + B.cast(self.dtype, self.dim) * B.cast(self.dtype, B.log_2_pi)
                    + B.iqf_diag(self.var, B.subtract(x, self.mean))
                    # Compute the diagonal of `transpose(b) inv(a) c` where `a` is assumed to be PD
                    # => (x-m)T @ inv(var) @ (x-m)
                )
                / 2
            )
        else:
            raise ValueError("Unsupported shape of input.")

        return logpdfs[..., 0] if B.shape(logpdfs, -1) == 1 else logpdfs

    @property
    def dtype(self):
        """dtype: Data type of the variance."""
        return B.default_dtype

    @property
    def dim(self):
        """int: Dimensionality."""
        return B.shape_matrix(self.var)[0]


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
    def from_normal(cls, dist: Normal):
        """
        Convert class:Normal into class:NaturalNormal
        - \\eta = [\\Sigma_inv \\mu, -0.5 \\Sigma_inv]^T
        """
        return cls(B.mm(B.pd_inv(dist.var), dist.mean), B.pd_inv(dist.var))

    @property
    def mean(self):
        """column vector: Mean."""
        if self._mean is None:
            # Cholsolve solves the linear system L x = b where L is a lower-triangular cholesky factorization
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
        ratio = B.triangular_solve(B.chol(self.prec), B.chol(other.prec))  # M in wiki
        diff = self.mean - other.mean  # mu1 - mu0
        dT_prec_d = B.sum(B.sum(B.mm(other.prec, diff) * diff, -1), -1)
        logdet = B.logdet(B.mm(ratio, ratio, tr_a=True))
        sum_r = B.sum(ratio**2, -1)

        del diff, ratio
        _kl = 0.5 * (B.sum(sum_r, -1) - logdet + dT_prec_d - B.cast(self.dtype, self.dim))  # ratio^T @ ratio  # (diff)^T @ prec @ diff  # subtract dimension |K| scalar
        return _kl

    def logpdf(self, x):
        return Normal.from_naturalnormal(self).logpdf(x)

    def sample(self, key: B.RandomState, num: B.Int = 1):
        """
        Sample from distribution using the natural parameters
        """
        # Sample noise (epsilon)
        if num > 1:
            key, noise = B.randn(key, B.default_dtype, num, *B.shape(self.lam))  # [num x q.lam.shape]
        else:
            key, noise = B.randn(key, B.default_dtype, *B.shape(self.lam))

        # Sampling from MVN: s = mean + chol(variance)*eps (affine transformation property)
        # dW = torch.triangular_solve(noise, B.dense(B.chol(self.prec)), upper=False, transpose=True).solution  # Ober sampling

        # Non-centered, precision parameterization
        if type(self.prec) == Diagonal:
            # sample = self.mean + B.mm(B.pd_inv(B.chol(self.prec)), noise)
            sample = self.mean + B.mm(B.chol(self.var), noise)
        else:
            # Trisolve solves Ux = y, where U is an upper triangular matrix
            sample = self.mean + B.triangular_solve(B.T(B.chol(self.prec)), noise, lower_a=False)

        del noise
        if not structured(sample):
            sample = B.dense(sample)  # transform Dense to Transform matrix

        return key, sample

    def __mul__(self, other):
        if isinstance(other, NaturalNormal):
            return NaturalNormal(self.lam + other.lam, self.prec + other.prec)
        elif isinstance(other, MeanFieldFactor):
            return MeanFieldFactor(self.lam + other.lam, self.prec + other.prec)
        else:
            raise NotImplementedError

    def __truediv__(self, other: "NaturalNormal"):
        return NaturalNormal(self.lam - other.lam, self.prec - other.prec)

    def __rtruediv__(self, other: "NaturalNormal"):
        return NaturalNormal(other.lam - self.lam, other.prec - self.prec)

    def __copy__(self):
        return NaturalNormal(
            deepcopy(self.lam.detach().clone()),
            deepcopy(B.dense(self.prec).detach().clone()),
        )

    def __repr__(self) -> str:
        return f"lam: {self.lam.shape}, \nprec: {self.prec.shape} \n"


class MeanFieldFactor:
    def __init__(self, lam, prec):
        """
        MeanFieldFactor (MFF) a NaturalNormalFactor with a diagonal precision.
        It is not a proper distribution, so we cannot sample from it. (It can have negative precision.)
        :param lam: first natural parameter of Normal dist = precision x mean
        :param prec: second natural parameter of Normal dist = -0.5 x precision \\propto precision
        """
        self.lam = lam
        self.prec = Diagonal(prec) if len(prec.shape) < 4 else prec

    @property
    def dtype(self):
        """dtype: Data type of the precision."""
        return B.dtype(self.prec)

    @property
    def dim(self):
        """int: Dimensionality."""
        return B.shape_matrix(self.prec)[0]

    def __mul__(self, other: Union["MeanFieldFactor", "NaturalNormal"]):
        if type(other) == MeanFieldFactor:
            return MeanFieldFactor(self.lam + other.lam, self.prec + other.prec)
        elif type(other) == NaturalNormal:
            return NaturalNormal(self.lam + other.lam, self.prec + other.prec)
        else:
            raise TypeError("Can't multiply by this type.")

    def __truediv__(self, other: "MeanFieldFactor"):
        return MeanFieldFactor(self.lam - other.lam, self.prec - other.prec)

    def __rtruediv__(self, other: "MeanFieldFactor"):
        return MeanFieldFactor(other.lam - self.lam, other.prec - self.prec)

    def __copy__(self):
        return MeanFieldFactor(self.lam.detach().clone(), B.dense(self.prec.diag).detach().clone())

    def __repr__(self) -> str:
        return f"lam: {self.lam.shape}, \nprec: {self.prec.shape} \n"


class MeanField(NaturalNormal):
    def __init__(self, lam, prec):
        """
        :param lam: first natural parameter of Normal dist = precision x mean
        :param prec: second natural parameter of Normal dist = -0.5 x precision \\propto precision
        """
        self.lam = lam

        if isinstance(prec, Diagonal):
            self.prec = prec
        else:
            assert lam.shape[:-1] == prec.shape, "Dealing with one dimensional precisions only."
            self.prec = Diagonal(prec)

        self._mean = None
        self._var = None

    @classmethod
    def from_factor(cls, factor: MeanFieldFactor):
        """Converts NaturalNormalFactor into NaturalNormal distribution"""
        MIN_PREC = 1e-4
        # if B.any(factor.prec.diag < 0).item():
        # logger.info(f"MeanField.from_factor: negative precision detected. Setting to {MIN_PREC}")
        # factor.prec.mat[factor.prec.diag < 0] = MIN_PREC
        # if B.any(factor.prec.diag > 1e3).item():
        # logger.info(f"MeanField.from_factor: large precision detected: {B.max(factor.prec.diag)}")
        return cls(lam=factor.lam, prec=factor.prec)

    def kl(self, other: "MeanField"):
        """Compute the Kullback-Leibler divergence with respect to another normal
        parametrised by its natural parameters.
        Args:
            other (:class:`.MeanField`): Other.
        Returns:
            scalar: KL divergence with respect to `other`.

        See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions for more info
        """
        logdet = B.sum(B.log(self.prec.diag) - B.log(other.prec.diag), -1)
        diff = (self.mean - other.mean)[..., 0]  # [50, Dout, 1]
        ratio = B.sum((self.var.diag + diff**2) / other.var.diag, -1)

        return 0.5 * (logdet + ratio - B.cast(self.dtype, self.dim))


class NormalPseudoObservation:
    def __init__(self, yz, nz):
        """
        :param yz: inducing point observation (pseudo-observations)
        :param nz: inducing point noise precision

        The pseudo-observations (_yz) are denoted by v^{\ell} in Ober et al.
        The precision of the inducing likelihood (prec_yz) is denoted by \\Lambda_{\\ell} in Ober et al.
        The inducing inputs (_z) are denoted by \\phi(U_{\\ell-1}) in Ober et al., but also as Xi in their code
        ---
        The mean of the q over the weights is (Sigma_w \\times _z @ prec_yz @ _yz)
        The precision of the q over the weights is (prior precision + z^T @ prec_yz @ z)
        """
        self.yz = yz  # [M x Dout]
        self.nz = nz  # [Dout x M]

    def __call__(self, z):

        """
        :param z: inducing inputs of that layer which are equal to the outputs of the prev layer inducing inputs, i.e. phi(U_{\\ell-1}) [samples x M x Din]

        :returns: N(w; lam_w, prec_w)
        """
        # (S, 1, M, Din)
        _z = B.to_active_device(B.expand_dims(z, 1))

        # (1, Dout, M, 1).
        _yz = B.expand_dims(B.transpose(self.yz, (-1, -2)))
        _yz = B.to_active_device(B.expand_dims(_yz, -1))

        # (Dout, M, M).
        prec_yz = B.diag_construct(self.nz)  # the precision of the "inducing-points likelihood"

        # (1, Dout, M, M).
        _prec_yz = B.expand_dims(prec_yz, 0)

        # (S, Dout, Din, Din).
        prec_w = B.mm(B.transpose(_z), B.mm(_prec_yz, _z))  # zT @ prec_yz @ z = XLX = XiT @ Lambda @ Xi

        # (S, Dout, Din, 1)
        # lam \\propto prec*mean, mean_w = (prec^-1) * XLY => lam_w = XLY
        lam_w = B.mm(B.transpose(_z), B.mm(_prec_yz, _yz))  # @ _z * _nz @ _yz = XLY

        del _prec_yz, _z, _yz
        return NaturalNormal(lam_w, prec_w)

    def __repr__(self) -> str:
        return f"yz: {self.yz.shape}, nz: {self.nz.shape}"

    def __copy__(self):
        return NormalPseudoObservation(deepcopy(self.yz.detach().clone()), deepcopy(self.nz.detach().clone()))
