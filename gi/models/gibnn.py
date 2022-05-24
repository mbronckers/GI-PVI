import lab as B

class GIBNN:
    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self._cache = {}
        
    def sample_posterior(self, key: B.RandomState, ps: dict, ts: dict, z: B.Numeric, S: B.Int=1):
        """
        :param ps: priors. dict<k=layer_name, v=_p>
        :param ts: pseudo-likelihoods. dict<k='layer x', v=dict<k='client x', v=t>>
        :param z: inducing inputs
        :param S: number of samples to propagate through
        """
        # z is [M, D]. Change to [S, M, D]]
        assert len(z.shape) == 2
        z = B.tile(z, S, 1, 1)
        for i, (layer_name, p) in enumerate(ps.items()):
            q = p
            for t in ts[layer_name].values():
                q *= t.compute_factor(z)
            
            # w is [S, Din, Dout].
            key, w = q.sample(key, (S,))
            self._cache[layer_name] = {"w": w, "kl": q.kl(p)}

            z = w @ z # update z
            if i < len(ps.keys()): # non-final layer
                z = self.nonlinearity(z)
                
        return key, self._cache
                
    def propagate(self, x):
        if self._cache is None:
            return None
            
        for i, (layer_name, layer_dict) in enumerate(self._cache.items()):
            x = layer_dict["w"] @ x
            if i < len(self._cache.keys()): # non-final layer
                z = self.nonlinearity(z)
                
        return x
    
    def __call__(self, x):
        return self.propagate(x)

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache