import logging
import numpy as np

logger = logging.getLogger(__name__)

def track_change(opt, vs, var_names, i, epoch, olds):
    """ Steps optimizer and reports delta for variables when iteration%epoch == 0 """
    _olds = olds

    opt.step()

    if i%epoch==0:
        # Report deltas
        for var_name in var_names:
            if _olds != {}:
                _delta = vs[var_name].detach().cpu().squeeze() - _olds[var_name]
            else:
                _delta = vs[var_name].detach().cpu().squeeze()

            logger.debug(f"Δ{var_name}: {np.array(_delta)}")

        # Update olds
        for var_name in var_names:
            _old = vs[var_name].detach().cpu().squeeze()
            _olds[var_name] = _old

    return opt, olds

def track(opt, vs, var_names, i, epoch):
    opt.step()

    for var_name in var_names:
        _v = vs[var_name].detach().cpu().squeeze()
        if i%epoch == 0: logger.debug(f"{var_name}: {np.array(_v)}")

    return opt
