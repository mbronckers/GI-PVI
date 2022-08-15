# GI-PVI
Repository for MPhil Thesis: Global Inducing Point Variational Approximations for Federated BNNs using PVI.

Run classification experiment as follows:
`python classification.py --q {GI,MFVI} -d {A,B} --prior {neal,std} --server {SYNC,SEQ} --split {A,B} --lr 0.001 --local_iters 20000 --global_iters 10 --batch 256 --num_clients=1`

Run regression experiment. Specify arguments in `config/ober.py`.
`python pvi_regression.py`
or
`python mfvi_regression.py`