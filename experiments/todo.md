## Plotting

- Across training epochs, plot (z, final_yz) for every client in separate client folder
- TODO: plot weight distribution of q

- TODO: plot sampled functions as line, same color, really low alpha

## Multi-client

### Homogeneous partitions
- Sequential PVI
- Stay on toy regression dataset
- Uniform sampling for all the clients
- Use gaps in training dataset, use smaller N per client

### Heterogeneous partitions:
- Partition dataset into left and right half
- Combine one posterior with 2 clients, each having one half of the dataset

Metrics: log-likelihood, held-out RMSEs