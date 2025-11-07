import pandas as pd
import numpy as np
import inspect

import statsforecast.models as sfm
from statsforecast.models import (
    HistoricAverage,
    SeasonalNaive,
    AutoARIMA,
    HoltWinters,
    AutoETS,
    CrostonOptimized,
    GARCH,
    Naive,
    AutoTBATS
)
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import  HierarchicalPlot
from hierarchicalforecast.core import HierarchicalReconciliation

from statsforecast.core import StatsForecast
from hierarchicalforecast.utils import aggregate as hf_aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace

from plotnine import ggplot, aes, geom_line, facet_wrap 


# Prepare data
# need to be arranged for each dataset 
def prepare(df): 

    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')

    # Build the hierarchy 
    df['Total'] = 'TOTAL'
    spec = [
        ['Total'],               # top level (TOTAL)
        ['Total', 'id']   # bottoms (each series)
    ]
    # Aggregate Series
    df, S_df, tags = hf_aggregate(df=df, spec=spec)

    return df, S_df, tags


# init models
def init_model(model_dict, freq):

    # make sure every model carries its alias
    for alias, model in model_dict.items():
        model.alias = alias


    ## Instantiate StatsForecast class as sf
    sf = StatsForecast( 
        models=list(model_dict.values()),
        freq=freq, 
        fallback_model = HistoricAverage(),
        n_jobs=-1,
    )

    return sf


# best model
def best_model(df, sf, h, n_windows, step_size):

    ## Cross Validation
    df_cv = sf.cross_validation(
        df=df,
        h=h,
        step_size=1,
        n_windows=n_windows
    )

    # Cross validation Stats
    df_cv_stat = cross_validation_stats(df_cv)

    # Best Model
    df_best_model = (
        df_cv_stat
            .loc[df_cv_stat.groupby('unique_id')['mape_L2'].idxmin()]
            .reset_index(drop=True)
    )

    return df_best_model, df_cv_stat




# Split trn & tst
# Split
def split(df, h):

    ds_split = df['ds'].max() -  pd.DateOffset(months=h)
    df_trn =  df[df['ds'] <= ds_split]
    df_tst =  df[df['ds'] > ds_split]

    return df_trn, df_tst


# MAPE
def MAPE(y, y_hat):
    mask = y != 0
    return np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask]))

# Cross validation kpi
def cross_validation_stats(cv_df):
    
    cv_df = (
        cv_df
        .melt(
            id_vars=['unique_id', 'cutoff', 'y', 'ds'],   # columns to keep fixed
            var_name='model',                       # name of the new “model” column
            value_name='y_hat'                      # name for model forecast values
            )
        )

    
    
    mape_df = (
        cv_df.groupby(['unique_id', 'model', 'cutoff'], as_index=False)
        .apply(lambda g: pd.Series({'mape': MAPE(g['y'], g['y_hat'])}), include_groups=False)
        .reset_index(drop=True)
    )

    mape_stats = (
        mape_df
        .groupby(['unique_id', 'model'], as_index=False)
        .agg(
            mape_mean=('mape', 'mean'),
            mape_std=('mape', 'std'),
            mape_max=('mape', 'max')
        )
    )

    mape_stats = mape_stats.assign(
        
        mape_L2 = np.sqrt(
            mape_stats['mape_mean']**2 + 
            mape_stats['mape_std']**2  
            #0.1*mape_stats['mape_max']**2
            )
        
        
    )

    return mape_stats
    

# make model
def make_model(name: str, model_dict):
    best_model = model_dict[name]
    return best_model
    



# Forecast Best Model
def forecast_best(df_trn , df_best_model, model_dict, h , freq):

    forecasts = []
    fitted_values = []

    for _, row in df_best_model.iterrows():
        uid = row["unique_id"]
        model_name = row["model"]

        # instantiate the correct model
        m = make_model(model_name, model_dict)

        # subset this series 
        ser = df_trn.loc[df_trn["unique_id"] == uid, ["unique_id", "ds", "y"]]
        

        # fit & forecast
        sf = StatsForecast(models=[m], freq=freq, n_jobs=1)
        fcst = sf.forecast(df=ser, h=h, fitted=True).reset_index(drop=True)
        fit = sf.forecast_fitted_values().reset_index(drop=True)

        # find the forecast column name (the model output)
        pred_col = [c for c in fcst.columns if c not in {"unique_id", "ds"}][0]

        fcst = fcst.rename(columns={pred_col: "yhat"})
        fcst["model"] = model_name

        forecasts.append(
            fcst[["unique_id", "ds", "yhat"]]  # we don't need model downstream
        )

        # Fitted values (in-sample) 
        fit_col = [c for c in fit.columns if c not in {"unique_id", "ds", "y"}][0]
        fit = fit.rename(columns={fit_col: "yhat"})
     

        fitted_values.append(
            fit[["unique_id", "ds", "y", "yhat"]]
        )

    # combine results after loop
    df_fct = pd.concat(forecasts, ignore_index=True)      # future forecasts
    df_fit     = pd.concat(fitted_values, ignore_index=True)  # in-sample actuals + fitted

    return  df_fit , df_fct

    


# Reconcile 
def reconcile_best(df_fct, df_fit, S_df, tags):

    #hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method='ols')])
    hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method='mint_shrink')])


    df_fct = hrec.reconcile(
        Y_hat_df=df_fct,  # base forecasts (bottoms; optionally top)
        Y_df=df_fit,
        S_df=S_df,
        tags=tags
    )

    # rename  recocciled column to standard 'yhat'
    col = [c for c in df_fct.columns if c not in {'unique_id', 'ds', 'yhat'}][0]
    df_fct = df_fct.rename(columns={col:'yhat_rec'})

    return df_fct

# Mape on best model
def mape_best (df_fct, df_tst):

    merged_df = pd.merge(
        df_tst,          # test data with actual y
        df_fct,     # forecasted data with y_hat
        on=["unique_id", "ds"],  # join keys
        how="left"       # keep all test rows (or "inner" if you only want matched ones)
    )


    mape_df = (
        merged_df.groupby('unique_id', as_index=False)
        .apply(
            lambda g: pd.Series({
                'mape': MAPE(g['y'], g['yhat']),
                'mape_rec': MAPE(g['y'], g['yhat_rec'])
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    return mape_df

# Plot results
def plot_result(df, df_fct , df_tst, n_tail, total = False):


    # Plot 
    df_merged = pd.merge(
            df_tst,          
            df_fct,     
            on=["unique_id", "ds"],  
            how="left" 
        )


    if total == True:
        df_plot = df[df["unique_id"] != "TOTAL"]
        df_merged = df_merged[df_merged["unique_id"] != "TOTAL"]


    df_plot = df_plot.groupby("unique_id").tail(n_tail)
    

    plt = (
        ggplot(df_plot)  # What data to use
            + aes(x="ds", y="y")  # What variable to use
            + geom_line() 
            + geom_line(aes(x="ds", y="yhat"), data=df_merged, color="red")
            + facet_wrap("unique_id", scales="free_y") # Geometric object to use for drawing
    )

    return(plt)
