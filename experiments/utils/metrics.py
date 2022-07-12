import lab as B

def rmse(y_true, y_pred):
    if len(y_pred.shape) == 3:
        if y_pred.device != y_true.device:
            y_pred = y_pred.to(y_true.device)
        return B.sqrt(B.mean(y_true - y_pred.mean(0))**2)