import os

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

    results_dir = f"/homes/mojb2/Thesis/GI-PVI/results/"
    exp_dir = latest_subdir(results_dir)

    _vs_state_dict = torch.load(exp_dir + '/model/_vs.pt')

    vs: Vars = Vars(torch.float64)

    for idx, name in enumerate(_vs_state_dict.keys()):
        if name.__contains__("output_var") or \
            name.__contains__("nz"):

            vs.positive(_vs_state_dict[name], name=name)
        else:
            vs.unbounded(_vs_state_dict[name], name=name)

    vs.print()