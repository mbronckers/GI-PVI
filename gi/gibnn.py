import lab as B
import torch
class GIBNN:
    def __init__(self, nonlinearity, bias):
        self.nonlinearity = nonlinearity
        self._cache = {}
        self.bias = bias
        
    def process_z(self, zs):
        _zs = {}
        for client_name, client_z in zs.items():
            assert len(client_z.shape) == 2 # [M, Din]
            
            # Add bias vector: [M x Din] to [M x Din+bias]
            if self.bias:
                _bias = B.ones(*client_z.shape[:-1], 1)
                _cz = B.concat(client_z, _bias, axis=-1)
            else:
                _cz = client_z

            # z is [M, D]. Change to [S, M, D]]
            _zs[client_name] = B.tile(_cz, S, 1, 1) # only tile intermediate results
            
        return _zs
    
    def propagate_z(self, zs, w, nonlinearity=True):
        for client_name, client_z in zs.items():
            client_z = B.mm(client_z, w, tr_b=True)         # propagate z. [S x M x Dout]
            
            if nonlinearity:                      # non-final layer
                client_z = self.nonlinearity(client_z)      # forward and updating the inducing inputs
            
                # Add bias vector to any intermediate outputs
                if self.bias:
                    _bias = B.ones(*client_z.shape[:-1], 1)
                    _cz = B.concat(client_z, _bias, axis=-1)
                else:
                    _cz = client_z
            else:
                _cz = client_z
            
            # Always store in _zs
            zs[client_name] = _cz 

    def sample_posterior(self, key: B.RandomState, ps: dict, ts: dict, zs: dict, ts_p: dict = {}, zs_p: dict = {}, S: B.Int):
        """
        :param ps: priors. dict<k=layer_name, v=_p>
        :param ts: pseudo-likelihoods. dict<k='layer x', v=dict<k='client x', v=t>>
        :param zs: client-local inducing intputs. dict<k=client_name, v=inducing inputs>
        :param ts_p: pseudo-likelihoods used to define the prior.
        :param zs_p: client-local inducing inputs used to define the prior.
        :param S: number of samples to draw (and thus propagate)

        M inducing points, 
        D input space dimensionality
        """
        _zs = self.process_z(zs)
        _zs_p = self.process_z(zs_p)

        for i, (layer_name, p) in enumerate(ps.items()):

            # Init posterior to prior
            q = p 
            p_ = p
            
            # Compute new posterior distribution by multiplying client factors
            for client_name, t in ts[layer_name].items():
                q *= t(_zs[client_name])    # propagate prev layer's inducing outputs
                
            # Compute prior distribution by multiplying factors
            for client_name, t in ts_p[layer_name].items():
                p_ *= t(_zs_p[client_name])
            
            # Sample weights from posterior distribution q. q already has S passed via _zs
            key, w = q.sample(key) # w is [S, Dout, Din] of layer i.
            
            # Get rid of last dimension.
            w = w[..., 0] # [S, Dout, Din]
    
            # Compute KL div
            kl_qp = q.kl(p_)  # [S, Dlatent] = [S, Dout]
            
            # Sum across output dimensions. [S]
            kl_qp = B.sum(kl_qp, -1) 

            # Save posterior w samples and KL to cache
            self._cache[layer_name] = {"w": w, "kl": kl_qp}

            # Propagate client-local inducing inputs <z> and store prev layer outputs in _zs
            if i < len(ps.keys()) - 1:     
                self.propagate_z(_zs, w, True)
                self.propagate_z(_zs_p, w, True)
            else:
                self.propagate_z(_zs, w, False)
                self.propagate_z(_zs_p, w, False)
                
        return key, self._cache
                
    def propagate(self, x):
        """Propagates input through BNN using S cached posterior weight samples. 

        :param x: input data

        Returns: output values from propagating x through BNN
        """
        if self._cache is None:
            return None

        if len(x.shape) == 2:
            x = B.tile(x, self.S, 1, 1)
            if self.bias:
                _bias = B.ones(*x.shape[:-1], 1)
                x = B.concat(x, _bias, axis=-1)

        for i, (layer_name, layer_dict) in enumerate(self._cache.items()):
            x = B.mm(x, layer_dict["w"], tr_b=True)
            if i < len(self._cache.keys()) - 1: # non-final layer
                x = self.nonlinearity(x)
                
                if self.bias:
                    _bias = B.ones(*x.shape[:-1], 1)
                    x = B.concat(x, _bias, axis=-1)

        return x
    
    def __call__(self, x):
        return self.propagate(x)

    @property
    def S(self):
        """ Returns cached number of weight samples """
        return self._cache['layer0']['w'].shape[0]

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache