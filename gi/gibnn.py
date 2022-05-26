import lab as B

class GIBNN:
    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self._cache = {}
        
    def sample_posterior(self, key: B.RandomState, ps: dict, ts: dict, zs: dict, S: B.Int=1):
        """
        :param ps: priors. dict<k=layer_name, v=_p>
        :param ts: pseudo-likelihoods. dict<k='layer x', v=dict<k='client x', v=t>>
        :param zs: client-local inducing intputs. dict<k=client_name, v=inducing inputs>
        :param S: number of samples to draw (and thus propagate)

        TODO: 
        - want to pass client-local inducing inputs
        - want to propagate each client's inducing inputs, storing them to pass them to the next layer
        - but want to have the original (0th layer) inducing inputs stored after this sample_posterior function call. 

        - copy of dict is (too?) expensive
        """
        _zs = {} # dict to store propagated inducing inputs

        for client_name, z in zs.items():
            assert len(z.shape) == 2
            # z is [M, D]. Change to [S, M, D]]
            zs[client_name] = B.tile(z, S, 1, 1)
            
        for i, (layer_name, p) in enumerate(ps.items()):

            # Compute new posterior distribution by multiplying client factors
            q = p # prior (posterior)
            for t in ts[layer_name].values():
                q *= t.compute_factor(zs[client_name])
            
            # Sample weights from posterior distribution q, compute KL, and save results
            key, w = q.sample(key, (S,)) # w is [S, Din, Dout].
            kl_qp = q.kl(p)  # Compute KL div
            self._cache[layer_name] = {"w": w, "kl": kl_qp}   # save weight samples and the KL div for every layer

            # Propagate client-local inducing inputs <z> 
            inducing_inputs = zs if i == 0 else _zs
            for client_name, z in inducing_inputs.items():
                z = w @ z # update z
                if i < len(ps.keys()): # non-final layer
                    z = self.nonlinearity(z) # forward and updating the inducing inputs
                
                # Always store in _zs
                _zs[client_name] = z 
                
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