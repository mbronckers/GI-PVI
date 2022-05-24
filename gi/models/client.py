class Client:
    """
    :param data: 
    :param name: client name. Optional
    :param z: global inducing points. We have client-local inducing points
    :param yz: pseudo obs
    :param nz: pseudo noise
    """
    def __init__(self, data, name, z, yz, nz):
        self.data = data
        self.name = name
        self.z = z
        self.yz = yz
        self.nz = nz