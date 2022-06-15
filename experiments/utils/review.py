"""
This file gets the most recently modified results folder (i.e. most recent results) and outputs the variable states at the end of training
"""
import os
import sys

def latest_subdir(b="/homes/mojb2/Thesis/GI-PVI/results/"):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        
        if os.path.isdir(bd): result.append(bd)

    return max(result, key=os.path.getmtime)

if __name__ == "__main__":
    import lab.torch as B
    import torch

    from varz import Vars

    results_dir = f"/homes/mojb2/Thesis/GI-PVI/results/" if len(sys.argv) == 1 else sys.argv[1]
    exp_dir = latest_subdir(results_dir) if len(sys.argv) <= 2 else os.path.join(results_dir, sys.argv[2])

    _vs_state_dict = torch.load(exp_dir + '/model/_vs.pt')

    vs: Vars = Vars(torch.float64)

    for idx, name in enumerate(_vs_state_dict.keys()):
        if name.__contains__("output_var") or \
            name.__contains__("nz"):

            vs.positive(_vs_state_dict[name], name=name)
        else:
            vs.unbounded(_vs_state_dict[name], name=name)

    vs.print()