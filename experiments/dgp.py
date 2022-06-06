import lab as B
import lab.torch

def generate_data(key, size):
    """ Toy regression dataset from paper """
    x = B.zeros(B.default_dtype, size, 1)
    
    key, x[:int(size / 2), :] = B.rand(key, B.default_dtype, int(size / 2), 1)
    x[:int(size / 2)] = x[:int(size / 2)] * 2. - 4.
    key, x[int(size / 2):] = B.rand(key, B.default_dtype, int(size / 2), 1)
    x[int(size / 2):] = x[int(size / 2):] * 2. + 2.
    
    key, eps = B.randn(key, B.default_dtype, size, 1)
    y = x ** 3. + 3*eps

    # Rescale the outputs to have unit variance
    scale = B.std(y)
    y = y/scale
    
    return key, x, y

def generate_test_data(key, size):
    """ Toy (test) regression dataset from paper """
    x = B.zeros(B.default_dtype, size, 1)
    
    key, x = B.rand(key, B.default_dtype, int(size), 1)
    x = x * 4. - 2.
    
    key, eps = B.rand(key, B.default_dtype, size, 1)
    y = x ** 3. + 3*eps
    print(B.mean(eps))
    
    # Rescale the outputs to have unit variance
    scale = B.std(y)
    y = y/scale
    
    return key, x, y

def generate_data2(key, size, xmin, xmax):
    
    key, eps1 = B.rand(key, B.default_dtype, int(size), 1)
    key, eps2 = B.rand(key, B.default_dtype, int(size), 1)

    eps1, eps2 = eps1.squeeze(), eps2.squeeze()
    x = B.expand_dims(eps2 * (xmax - xmin) + xmin, axis=1).squeeze()
    y = x + 0.3 * B.sin(2 * B.pi * (x + eps2)) + 0.3 * B.sin(4 * B.pi * (x + eps2)) + eps1 * 0.02

    return key, x, y, eps1, eps2