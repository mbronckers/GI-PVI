from typing import Optional

class Client:
    """
    Client class that contains: client-local data, the parameters of its factor, and a function how to build the factor.

    Each client in GI-PVI has their own set of inducing points.

    :param data: {X, Y}
    :param name: Client name. Optional
    :param z: Global inducing points. (we have client-local inducing points)
    :param yz: Pseudo (inducing) observations (outputs)
    :param nz: Pseudo noise
    """
    def __init__(self, data, name: Optional[str], z, yz, nz):
        self.data = data if data else None
        self.name = name if name else None
        self.z = z
        self.yz = yz
        self.nz = nz