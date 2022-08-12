from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pandas as pd


def regres(x, y, estimator, label):
    y_pred = estimator.predict(x)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** (1/2)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    perf_df = pd.DataFrame(
        {'r2': r2, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mae': mae}, index=[label])
    return perf_df


def regres2(x, y, estimator_list, label_list):
    r2_list = []
    mse_list = []
    rmse_list = []
    mape_list = []
    mae_list = []
    for estimator in estimator_list:
        y_pred = estimator.predict(x)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = mse ** (1/2)
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)

        r2_list.append(r2)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mape_list.append(mape)
        mae_list.append(mae)

    perf_df = pd.DataFrame({'r2': r2_list, 'mse': mse_list, 'rmse': rmse_list, 'mape': mape_list, 'mae': mae_list},
                           index=label_list)
    return perf_df
