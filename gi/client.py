from __future__ import annotations

from typing import Optional
from gi.distributions import NormalPseudoObservation

class Client:
    """
    Client class that contains: client-local data, the parameters of its factor, and a function how to build the factor.

    Each client in GI-PVI has their own set of inducing points.

    :param data: {X, Y}
    :param name: Client name. Optional
    :param z: Global inducing points. (we have client-local inducing points)
    :param t: Likelihood factors per layer. Dict<k=layer, v=NormalPseudoObservation()>
    :param yz: Pseudo (inducing) observations (outputs)
    :param nz: Pseudo noise
    """
    def __init__(self, name: Optional[str], data, z, t: dict[str, NormalPseudoObservation]):
        self.data = data if data else None
        self.name = name if name else None
        self.z = z
        self.t = t

    def update_nz(self, vs):
        """ Update likelihood factors' precision based on the current state of vs

        Args:
            vs: optimizable variable container
        """
        
        for i, layer_name in enumerate(self.t.keys()):
            var = vs[f"ts.{layer_name}_{self.name}_nz"]
            self.t[layer_name].nz = var