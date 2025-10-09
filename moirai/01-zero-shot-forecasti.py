#ref https://www.datasciencewithmarco.com/blog/hands-on-with-moirai-a-foundation-forecasting-model-by-salesforce

# Import
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS, NBEATSx, TSMixerx

from neuralforecast.losses.numpy import mae, mse

# Data
df = pd.read_csv('~/dev/ml/data/medium_views_published_holidays.csv')
df['ds'] = pd.to_datetime(df['ds'])
moirai_df = df.set_index('ds')

# Converts a long-format pandas DataFrame into a Hugging Face dataset with structured columns.
ds = PandasDataset.from_long_dataframe(
    moirai_df,
    target='y',
    item_id='unique_id',
    feat_dynamic_real=["published", "is_holiday"]
)

# Split the data into a training and test set. 
# 168 steps for the test set
# 24 windows with a 7-day horizon. 24*7 = 168
test_size = 168
horizon = 7

train, test_template = split(
    ds, offset=-test_size
)



# test Data set 
test_data = test_template.generate_instances(
    prediction_length=horizon,
    windows=test_size//horizon,
    distance=horizon
)

# To see test_data
#for instance in test_data:
#    print(instance)

#  initialize the Moirai model 
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
    prediction_length=horizon,
    context_length=500,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

# Make prediction
predictor = model.create_predictor(batch_size=32)
forecasts = predictor.predict(test_data.input)
forecasts = list(forecasts)

# function to  format the predictions as a DataFrame and extract a user-defined confidence interval.
def get_median_and_ci(data, 
                      horizon,
                      id,
                      confidence=0.80):

    n_samples, n_timesteps = data.shape
    
    # Calculate the median for each timestep
    medians = np.median(data, axis=0)
    
    # Calculate the lower and upper percentile for the given confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    # Calculate the lower and upper bounds for each timestep
    lower_bounds = np.percentile(data, lower_percentile, axis=0)
    upper_bounds = np.percentile(data, upper_percentile, axis=0)

    # Create a DataFrame with the results
    df = pd.DataFrame({
        'unique_id': id,
        'Moirai': medians,
        f'Moirai-lo-{int(confidence*100)}': lower_bounds,
        f'Moirai-hi-{int(confidence*100)}': upper_bounds
    })
    
    return df

def get_median_and_ci(data, 
                      horizon,
                      id,
                      confidence=0.80):

    n_samples, n_timesteps = data.shape
    
    # Calculate the median for each timestep
    medians = np.median(data, axis=0)
    
    # Calculate the lower and upper percentile for the given confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    # Calculate the lower and upper bounds for each timestep
    lower_bounds = np.percentile(data, lower_percentile, axis=0)
    upper_bounds = np.percentile(data, upper_percentile, axis=0)

    # Create a DataFrame with the results
    df = pd.DataFrame({
        'unique_id': id,
        'Moirai': medians,
        f'Moirai-lo-{int(confidence*100)}': lower_bounds,
        f'Moirai-hi-{int(confidence*100)}': upper_bounds
    })
    
    return df

moirai_preds = [
    get_median_and_ci(
        data=forecasts[i].samples,
        horizon=horizon,
        id=1
    )
    for i in range(24)
]


moirai_preds_df = pd.concat(moirai_preds, axis=0, ignore_index=True)



tst = df.tail(test_size)


prd  = pd.concat([tst.reset_index(drop=True), moirai_preds_df.reset_index(drop=True),], axis = 1)

plt.figure(figsize=(14, 5))

# Actual values (blue line)
plt.plot(prd['ds'], prd['y'], label='Actual (y)', color='steelblue', linewidth=2)

# Moirai median forecast (red line)
plt.plot(prd['ds'], prd['Moirai'], label='Moirai forecast', color='red', linewidth=2)

# 80% prediction interval (shaded red area)
plt.fill_between(prd['ds'],
                 prd['Moirai-lo-80'],
                 prd['Moirai-hi-80'],
                 color='red',
                 alpha=0.2,
                 label='Moirai 80% PI')

plt.title('Actual vs Moirai Forecast')
plt.xlabel('Date')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()
