import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def datetime_transformer(df, datetime_vars):
    """
    The datetime transformer

    Parameters
    ----------
    df : dataframe
    datetime_vars : the datetime variables
    
    Returns
    ----------
    The dataframe where datetime_vars are transformed into the following 3 datetime types:
    year, month, and day
    """
    
    # The dictionary with key as datetime type and value as datetime type operator
    dict_ = {'year'   : lambda x : x.dt.year,
             'month'  : lambda x : x.dt.month,
             'day'    : lambda x : x.dt.day}
    
    # Make a copy of df
    df_datetime = df.copy(deep=True)
    
    # For each variable in datetime_vars
    for var in datetime_vars:
        # Cast the variable to datetime
        df_datetime[var] = pd.to_datetime(df_datetime[var])
        
        # For each item (datetime_type and datetime_type_operator) in dict_
        for datetime_type, datetime_type_operator in dict_.items():
            # Add a new variable to df_datetime where:
            # the variable's name is var + '_' + datetime_type
            # the variable's values are the ones obtained by datetime_type_operator
            df_datetime[var + '_' + datetime_type] = datetime_type_operator(df_datetime[var])
            
    # Remove datetime_vars from df_datetime
    # df_datetime = df_datetime.drop(columns=datetime_vars)
                
    return df_datetime

def autocorrelation_cal(y,k):
    T=len(y)
    mean_y=np.mean(y)

    numerator=0
    denominator=0
    T_k=0
    
    for t in range(0,T):
        denominator=denominator+(np.square(y[t]-mean_y))

    for t in range(k,T):
        numerator=numerator+((y[t]-mean_y)*(y[t-k]-mean_y))

    T_k=numerator/denominator
    return T_k

def acf_plotter(y,l):
    #acf over y with 100 samples
    lags=[]
    autoCorr=[]
    max_lag=l
    for i in range(0,max_lag):
        lags.append(i)
        autoCorr.append(autocorrelation_cal(y,i))

    #making symmetrical acf plot about y axis
    autoCorr_copy=autoCorr[1:].copy()
    autoCorr_copy.reverse()
    autoCorr_copy=np.concatenate((autoCorr_copy,autoCorr),axis=None)

    lags_rev=lags[1:].copy()
    lags_rev.reverse()
    lags_rev=np.negative(lags_rev)
    lags_rev=np.concatenate((lags_rev,lags),axis=None)

    #plotting acf over y
    plt.stem(lags_rev,autoCorr_copy,use_line_collection=True)
    plt.title("ACF Plot for Sample size {} with {} lags".format(len(y),l))
    plt.show()


def nan_checker(df):
 
    # Get the variables with NaN, their proportion of NaN and dtype
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])
    
    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False)
    
    return df_nan