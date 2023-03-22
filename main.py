import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv('C:/Users/luana/OneDrive/Documentos/aprendmaq/ml-2023-1-trabalho-1/dataset.csv' )

#pd.set_option('display.max_columns', None)

dt =  data.dropna(axis=1, how='all')

#print(dt[['Creatinine', 'Urea']].describe())

columns = ['Patient ID', 'Patient age quantile', 'SARS-Cov-2 exam result', 'Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)' ]
dt.dropna(how='all', subset=dt.columns.difference(columns), inplace=True)
#print(dt[columns])

colunas = ['Creatinine', 'Urea']
print(dt[colunas])

dt_copy = dt.copy()

for column in colunas:
    i_column = dt_copy.columns.get_loc(column)
    dt_copy[dt.columns[i_column]] = dt_copy[dt_copy.columns[i_column]].str.replace(',', '.').astype(float)

    q1 = dt_copy[column].quantile(0.25)
    q3 = dt_copy[column].quantile(0.75)
    iqr = q3 - q1

    li = q1 - (1.5 * iqr)
    ls = q3 + (1.5 * iqr) 

    print('column:', column)
    print('Limite inferior:', li)
    print('Limite superior:', ls)
    print('IQR:', iqr)

    outliers = (dt_copy[column] < li) | (dt_copy[column] > ls)

    dt.loc[outliers, column] = np.nan

    #print(dt[column])
    #print(outliers)

    dt_copy.loc[:, 'outliers'] = outliers
    inlierColor = 'blue'
    outlierColor = 'red'
    plt.scatter(dt_copy.index, dt_copy[column], c=dt_copy['outliers'].map({True: outlierColor, False: inlierColor}))
    plt.title(f'Gráfico de dispersão da coluna {column}')
    plt.xlabel('Índice da linha')
    plt.ylabel(column)
    plt.show()

    dt_cleanOutliers = dt_copy[~outliers][[column]]
    plt.scatter(dt_cleanOutliers.index, dt_cleanOutliers[column])
    plt.title(f'Gráfico de dispersão da coluna {column} sem outliers')
    plt.xlabel('Índice da linha')
    plt.ylabel(column)
    plt.show()


print('Dados sem outliers:')
print(dt[colunas])
dt.to_csv('dataset_clean.csv', index=False)

