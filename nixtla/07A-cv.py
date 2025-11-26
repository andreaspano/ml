import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF
from siuba import group_by, summarize, _
from utilsforecast.losses import mape
from plotnine import ggplot, aes, geom_line, facet_wrap, geom_density , geom_histogram



#Data
df = AirPassengersDF

M = AutoARIMA(season_length=12, alias = 'Arima') #Forecasting model 
m = 12 #A minimum training size
h = 12 #forecast horizon: h
s = 1 #step size: s
n = 12 #number of windows: n


sf = StatsForecast(
    models=[M],
    freq='ME'
)

#A starting point o: usually the origin
#A loss function: L(y, ŷ) (e.g., RMSE, MAE, MAPE)



# Cross-validation 
df_cv = sf.cross_validation(
        df=df,
        h=h,
        step_size=s,
        n_windows=n,
        time_col = 'ds',
        target_col = 'y',
        id_col = 'unique_id'
    )


def MAPE(y, yhat):
    """
    Mean Absolute Percentage Error avoiding division by zero.

    Args:
      y (array-like): actual values.
      y_hat (array-like): predicted values.

    Returns:
      float: mean absolute percentage error computed only for y != 0.
    """
    mask = y != 0
    return np.mean(np.abs((y[mask] - yhat[mask]) / y[mask]))







df_mape = (df_cv
    >> group_by(_.unique_id, _.cutoff)
    >> summarize(mape = MAPE(_.y, _.Arima))
    >> summarize(mape_avg = np.mean(_.mape), mape_sd = np.std(_.mape), mape_max = np.max(_.mape))
)

df_mape


p = (
    ggplot(df_mape, aes(x="mape_avg"))
    + geom_histogram(bins=20, fill="skyblue", color="black")
    + labs(
        title="Distribution of Average MAPE",
        x="MAPE (average)",
        y="Count"
    )
    + theme_bw()
)
p


import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_bw

def plot_normal(mean, std, n_points=500):
    """
    Plot the normal distribution for a given mean and std using plotnine.

    Args:
        mean (float): Mean of the distribution.
        std (float): Standard deviation of the distribution.
        n_points (int): Number of points to plot.
    Returns:
        plotnine.ggplot object
    """
    x = np.linspace(mean - 4*std, mean + 4*std, n_points)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    df = pd.DataFrame({'x': x, 'density': y})

    p = (
        ggplot(df, aes(x="x", y="density"))
        + geom_line(color="blue")
        + labs(
            title=f"Normal Distribution (mean={mean}, std={std})",
            x="x",
            y="Density"
        )
        + theme_bw()
    )
    return p

# Example usage:
plot_normal(mean=0, std=1)
