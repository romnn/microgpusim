import typing
import numpy as np
import pandas as pd
import sktime
import sktime.performance_metrics.forecasting
import sklearn.metrics

EPS = np.finfo(np.float64).eps


def slowdown(baseline, values):
    return values / baseline


def speedup(baseline, values):
    return baseline / values


def geo_mean(values: np.ndarray) -> np.ndarray:
    a = np.array(values)
    return a.prod() ** (1.0 / len(a))


# def geo_mean(values: np.narray):
#     return np.exp(np.log(values).mean())


def bounded_relative_absolute_error(true_values: np.ndarray, values: np.ndarray, **kwargs) -> np.ndarray:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    correct = values == true_values

    # we only deal with positive numbers
    assert np.all(values >= 0.0)
    assert np.all(true_values >= 0.0)

    brae = values.abs() / (values.abs() + true_values.abs())
    brae = brae.fillna(0.0)
    # brae[brae] = 0.0
    brae[brae == 0.0] = 0.0
    return brae


def rel_err(true_values: np.ndarray, values: np.ndarray, eps: typing.Optional[float] = None) -> np.ndarray:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    correct = values == true_values

    # we only deal with positive numbers
    assert np.all(values >= 0.0)
    assert np.all(true_values >= 0.0)

    # because we only use posive numbers, we can safely clip to a small positive epsilon
    # if eps is not None:
    #     values = values + eps
    #     true_values = true_values + eps
    #     # true_values = np.clip(true_values, a_min=eps, a_max=None)
    rel_err = (values - true_values).abs() / true_values
    # rel_err = values.abs() / (values.abs() + true_values.abs())

    # print(values)
    # print(true_values)
    # print(values == true_values)
    rel_err = rel_err.fillna(0.0)
    rel_err[correct] = 0.0
    rel_err[rel_err == 0.0] = 0.0

    return rel_err


def rpd(true_values: np.ndarray, values: np.ndarray):
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    pass
    # rel_err = (values - true_values).abs() / true_values
    # rel_err = rel_err.fillna(0.0)
    # rel_err[rel_err == 0.0] = 0.0
    # return rel_err


def mse(true_values, values) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    return sklearn.metrics.mean_squared_error(true_values, values)


def rmse(true_values, values) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    return ((values - true_values) ** 2).mean() ** 0.5


def rmse_scaled(true_values, values) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    diff = values - true_values
    scale = values.abs() + true_values.abs()
    return (diff / scale).mean()


def abs_err(true_values: np.ndarray, values: np.ndarray) -> np.array:
    # values = values.fillna(0.0)
    # true_values = true_values.fillna(0.0)

    values = np.array(values)
    true_values = np.array(true_values)
    values = np.nan_to_num(values, copy=True, nan=0.0)
    true_values = np.nan_to_num(true_values, copy=True, nan=0.0)

    return np.abs(true_values - values)
    # return (true_values - values).abs()
    # return sklearn.metrics.mean_absolute_error(true_values, values)


def smape(true_values: np.ndarray, values: np.ndarray, raw=False) -> float:
    """SMAPE (symmetric)"""
    values = np.array(values)
    true_values = np.array(true_values)

    values = np.nan_to_num(values, copy=True, nan=0.0)
    true_values = np.nan_to_num(true_values, copy=True, nan=0.0)

    # values = values.fillna(0.0)
    # true_values = true_values.fillna(0.0)

    # strictly positive
    assert (values >= 0).all()
    assert (true_values >= 0).all()

    # denom = np.maximum(values.abs() + true_values.abs(), EPS)
    # smape = (values - true_values).abs() / denom

    denom = np.maximum(np.abs(values) + np.abs(true_values), EPS)
    smape = np.abs(values - true_values) / denom
    smape[values == true_values] = 0.0
    if raw:
        return smape
    else:
        return np.average(smape)


def ermsle(true_values: np.ndarray, values: np.ndarray) -> float:
    """ERMSLE: Exponential root mean square log error"""
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    ratios = values / true_values
    ratios[values == true_values] = 1.0

    log_ratios = np.empty_like(ratios)
    valid_mask = np.isfinite(ratios) & ratios != 0

    # temp
    ratios[~valid_mask] = 1.0
    log_ratios = np.abs(np.log(ratios)) ** 2
    # undo temp
    log_ratios[~valid_mask] = np.nan
    # mean
    rmsle = np.sqrt(np.mean(log_ratios[valid_mask]))
    # exponential
    rmsle = np.abs(np.exp(rmsle))
    return rmsle


def emale(true_values: np.ndarray, values: np.ndarray) -> float:
    """EMALE: Exponential mean absolute log error"""
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    ratios = values / true_values
    ratios[values == true_values] = 1.0
    # print("want", np.array(true_values))
    # print("have", np.array(values))
    # print("emale ratios", np.array(ratios))

    log_ratios = np.empty_like(ratios)
    valid_mask = np.isfinite(ratios) & ratios != 0
    if (ratios == 0.0).any():
        # invalid, cannot compute
        return np.nan

    # temp
    ratios[~valid_mask] = 1.0
    log_ratios = np.abs(np.log(ratios))
    # undo temp
    log_ratios[~valid_mask] = np.nan
    # mean
    male = np.mean(log_ratios[valid_mask])
    # exponential
    emale = np.abs(np.exp(male))
    return emale


def mape(true_values: np.ndarray, values: np.ndarray, raw=False):
    # if not isinstance(true_values, pd.Series):
    values = np.array(values)
    true_values = np.array(true_values)

    if np.isnan(values).all() or np.isnan(true_values).all():
        return np.nan

    values = np.nan_to_num(values, copy=True, nan=0.0)
    true_values = np.nan_to_num(true_values, copy=True, nan=0.0)

    # values = np.array(values).fillna(0.0)
    # true_values = np.array(true_values).fillna(0.0)

    denom = np.maximum(np.abs(true_values), EPS)
    mape = np.abs(values - true_values) / denom
    if raw:
        return mape
    else:
        return np.average(mape, axis=0)

    # return sklearn.metrics.mean_absolute_percentage_error(
    #     true_values, values, multioutput="raw_values" if raw else "uniform_average"
    # )


def rmspe(true_values: np.ndarray, values: np.ndarray, raw=False) -> float:
    # values = values.fillna(0.0)
    # true_values = true_values.fillna(0.0)

    values = np.array(values)
    true_values = np.array(true_values)
    values = np.nan_to_num(values, copy=True, nan=0.0)
    true_values = np.nan_to_num(true_values, copy=True, nan=0.0)

    return sktime.performance_metrics.forecasting.mean_squared_percentage_error(
        y_true=true_values,
        y_pred=values,
        square_root=True,
        symmetric=False,
        multioutput="raw_values" if raw else "uniform_average",
    )


def symmetric_rmspe(true_values: np.ndarray, values: np.ndarray) -> float:
    values = np.array(values)
    true_values = np.array(true_values)
    values = np.nan_to_num(values, copy=True, nan=0.0)
    true_values = np.nan_to_num(true_values, copy=True, nan=0.0)

    # values = values.fillna(0.0)
    # true_values = true_values.fillna(0.0)
    return sktime.performance_metrics.forecasting.mean_squared_percentage_error(
        y_true=true_values, y_pred=values, square_root=True, symmetric=True
    )


def mase(true_values: np.ndarray, values: np.ndarray) -> float:
    values = np.array(values)
    true_values = np.array(true_values)
    values = np.nan_to_num(values, copy=True, nan=0.0)
    true_values = np.nan_to_num(true_values, copy=True, nan=0.0)

    mae = abs_err(true_values=true_values, values=values)
    mean_absolute_deviation = np.mean(np.abs(true_values - values))
    return mae / mean_absolute_deviation
    # mase = np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:])

    # return mase.mean()

    # values = values.fillna(0.0)
    # true_values = true_values.fillna(0.0)
    # return sktime.performance_metrics.forecasting.mean_absolute_scaled_error(
    #     y_true=true_values, y_pred=values, y_train=None
    # )


def correlation(true_values: np.ndarray, values: np.ndarray, atol=None) -> float:
    values = np.array(values)
    true_values = np.array(true_values)
    values = np.nan_to_num(values, copy=True, nan=0.0)
    true_values = np.nan_to_num(true_values, copy=True, nan=0.0)

    # values = values.fillna(0.0)
    # true_values = true_values.fillna(0.0)

    assert len(values) == len(true_values)
    if len(values) <= 1:
        return np.nan
    # print("true values", true_values)
    # print("values", values)
    # print("values sum", values.sum())
    # print("values stddev", values.std())
    # print("true values stddev", true_values.std())
    # if values.sum() > 0 and :
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(true_values))

    # this does not change anything about the std dev
    # values += 1.0
    # true_values += 1.0

    print("correlation have", np.array(values))
    print("correlation want", np.array(true_values))

    if values.std() != 0 and true_values.std() != 0:
        return np.corrcoef(true_values, values)[0][1]
    # else:
    #     return np.nan
    elif atol is not None and np.allclose(
        values,
        # np.amin([values, true_values], axis=0),
        true_values,
        # np.amax([values, true_values], axis=0),
        atol=atol,
    ):
        return 1.0
    else:
        #     assert len(np.amin([values, true_values], axis=0)) == len(values)
        #     a = np.amin([values, true_values], axis=0)
        #     b = np.amax([values, true_values], axis=0)
        #     # print(a, b)
        #     # print(np.abs(a - b))
        return np.nan
