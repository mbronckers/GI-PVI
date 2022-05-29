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

        M inducing points, 
        D input space dimensionality
        """
        _zs = {} # dict to store propagated inducing inputs

        for client_name, client_z in zs.items():
            # assert len(client_z.shape) == 2
            
            # z is [M, D]. Change to [S, M, D]]
            if len(client_z.shape) == 2: 
                # We only need to tile once because we are modifying original zs
                zs[client_name] = B.tile(client_z, S, 1, 1)
            
        for i, (layer_name, p) in enumerate(ps.items()):

            # Compute new posterior distribution by multiplying client factors
            q = p # prior (posterior)
            for t in ts[layer_name].values():
                q *= t(zs[client_name])
            
            # Sample weights from posterior distribution q, compute KL, and save results
            key, w = q.sample(key) # w is [S, Din, Dout]. q already has S passed via zs
            
            # Get rid of last dimension.
            w = w[..., 0]
    
            # Compute KL div
            kl_qp = q.kl(p)  
            
            # Sum across output dimensions.
            kl_qp = B.sum(kl_qp, -1)
            
            self._cache[layer_name] = {"w": w, "kl": kl_qp}   # save weight samples and the KL div for every layer

            # Propagate client-local inducing inputs <z> 
            inducing_inputs = zs if i == 0 else _zs
            for client_name, client_z in inducing_inputs.items():
                client_z = B.mm(client_z, w, tr_b=True) # update z
                if i < len(ps.keys()): # non-final layer
                    client_z = self.nonlinearity(client_z) # forward and updating the inducing inputs
                
                # Always store in _zs
                _zs[client_name] = client_z 
                
        return key, self._cache
                
    def propagate(self, x):
        if self._cache is None:
            return None
            
        for i, (layer_name, layer_dict) in enumerate(self._cache.items()):
            x = B.mm(x, layer_dict["w"], tr_b=True)
            if i < len(self._cache.keys()): # non-final layer
                x = self.nonlinearity(x)
                
        return x
    
    def __call__(self, x):
        return self.propagate(x)

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache