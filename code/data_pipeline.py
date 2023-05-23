import pandas as pd
import matplotlib.pyplot as plt 
import missingno as msno

from sklearn.feature_selection import VarianceThreshold, f_regression

def data_pipeline(top_features=100,
                  plot=True,
                  lags=27) -> tuple[pd.DataFrame]:
    """
    Function to load, clean and prepare data for modelling.

    Parameters
    ----------
    top_features : int, optional
        Number of top features to plot. The default is 100.
    plot : bool, optional
        Whether to plot the top features. The default is True.
    lags : int, optional
        Number of lags to use. The default is 28.   

    Returns
    -------
    data_lagged : pandas.DataFrame
        Dataframe with lagged features.
    f_features : pandas.DataFrame
        Dataframe with feature importances.
    """

    data = pd.read_csv('../data/data.csv')
    data = data.fillna(0)

    p_data = data[data.group == 'P']
    i_data = data[data.group == 'I']

    for inj in i_data.cat.unique():
        inj_data = i_data[i_data.cat == inj].drop(columns=['cat', 'group', 
                                                        'oil', 'liquid', 
                                                        'is_base'])
        if inj != 'I3':
            inj_data = inj_data.drop(columns=['status'])

        p_data = p_data.merge(inj_data, 
                            on=['start_lag', 'coef', 'date'], 
                            suffixes=('', f'_{inj}'),
                            how='left')

    data_init = p_data.copy()

    baseline = (data_init[data_init.is_base]
                        .drop('is_base', axis=1)
                        .reset_index(drop=True))

    non_baseline = (data_init[~data_init.is_base]
                            .drop('is_base', axis=1)
                            .reset_index(drop=True))

    baseline = baseline.drop(columns=['start_lag', 'coef', 'group', 'status', 'water_I3', 'bhp_I3'])

    non_baseline = non_baseline.merge(baseline, 
                                    on=['cat', 'date'], 
                                    suffixes=('', '_base'), 
                                    how='left')

    non_baseline = non_baseline.drop(columns=['group'])
    non_baseline['cat'] = non_baseline['cat'].apply(lambda x: int(x[1]))

    if plot:
        msno.matrix(non_baseline)
        plt.title('NAN values in data')
        plt.show()

    sel = VarianceThreshold(threshold=0)
    sel.fit(non_baseline.drop(['cat'], axis=1))

    cols2keep = non_baseline.drop(['cat'], axis=1).columns[sel.get_support()]

    data_clean = non_baseline[['cat'] + list(cols2keep)]

    data_lagged = data_clean.copy()

    for i in range(1, lags):
        data_lagged = data_lagged.join(data_clean.groupby(['cat', 'start_lag', 'coef'])
                                                .shift(i)
                                                .add_suffix(f'_{i}d_ago')
                                                ).fillna(0)

    data_lagged['delta_baseline'] = data_lagged['oil'] - data_lagged['oil_base']

    data_lagged = data_lagged.drop(['water_I3', 'bhp_I1', 'bhp_I2', 'bhp_I3'], axis=1)

    f_reg = f_regression(data_lagged.drop(['oil', 'water', 'liquid', 'delta_baseline'], axis=1), 
                         data_lagged[['delta_baseline']])

    f_features = (pd.DataFrame({
                            'feature': (
                                        data_lagged
                                        .drop(['oil', 'water', 'liquid', 'delta_baseline'], axis=1)
                                        .columns
                                        ), 
                            'importance': f_reg[0]
                            })
                            .sort_values('importance', ascending=False)
                )

    if plot:
        plot_features(f_features, top_features)

    return data_lagged, f_features

def plot_features(f_features, top_features):
    plt.figure(figsize=(10, 20))
    plt.barh(f_features[:top_features].feature, f_features[:top_features].importance)
    plt.title('F-regression')
    plt.xlabel('F-value')
    plt.ylabel('Feature')
    plt.show()