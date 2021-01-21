
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# calculate different error measurements
# TODO: add nash sutcliffe model efficiency coefficient and R^2
# TODO: solve problem with mape either by replacing it with Relative percent difference (RPD) or removing it
def obtain_performance(y_true, guess):
    rmse = np.sqrt((1/len(y_true))*sum((y_true-guess)**2))
    mape= (1/len(y_true))*sum(abs((y_true-guess)/y_true))
    mae = (1/len(y_true))*sum(abs((y_true-guess)))
    r2 = r2_score(y_true, guess)
    return {'MAE':mae, 'MAPE': mape, 'RMSE':rmse, 'R2':r2 }

# plot model fit with vertical black dotted line where the training data ends
def plot_model(pred_list, endDateTrain, plotFile):
    plt.figure(figsize=(13,5))
    models = []
    for preds in pred_list:
        plt.plot(preds, alpha=0.8)
        models.append(list(preds.columns)[0])
    plt.legend(models, loc = 'upper right')
    plt.axvline(x=endDateTrain, color='black', ls='--')
    # save plot
    plt.savefig(plotFile, bbox_inches='tight')
    plt.close()

