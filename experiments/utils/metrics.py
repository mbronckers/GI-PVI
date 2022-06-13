import lab as B

def rmse(y_true, y_pred):
    if len(y_pred.shape) == 3:
        return B.sqrt(B.mean(y_true - y_pred.mean(0))**2)