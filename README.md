# GI-PVI
Repository for MPhil Thesis: Global Inducing Point Variational Approximations for Federated BNNs using PVI.

The repository is structured as follows. `gi/` contains the implementation of GI-PVI as well as MFVI-PVI. `experiments/` contains everything that is related to training the GI-PVI but independent of the method, including datasets (`dgp.py`), prior specification (`priors.py`). 

## Classification experiment

Arguments are specified via command-line. Precision initialization of the weights is done inside `classification.py` and is currently set to `1e3 - D_in`.

- Command: `python classification.py --q {GI,MFVI} -d {A,B} --prior {neal,std} --server {SYNC,SEQ} --split {A,B} --lr 0.001 --local_iters 20000 --global_iters 10 --batch 256 --num_clients=1`

## Regression experiment

- Specify arguments in `config/ober.py`.
- Command: `python pvi_regression.py` or `python mfvi_regression.py`.