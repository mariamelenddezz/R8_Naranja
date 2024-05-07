import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.stats import spearmanr, kendalltau, pointbiserialr
from sklearn.metrics import matthews_corrcoef
from empiricaldist import Cdf
import pickle
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from yellowbrick.regressor import ResidualsPlot
import os


def graficar_correlaciones(df, target, metric='pearson'):
    """Cálculo de diferentes métricas de correlación de las variables númericas del df con la variable target.

    Args:
        df: df con las variables
        target: variable a predecir
        metric: métrica a calcular. Defaults to 'pearson'.

    Raises:
        ValueError: Debes insertar alguna de estas métricas: 'pearson', 'spearman', 'kendall', 'point_biserial', 'matthews'.

    Returns:
        _type_: Devuelve la figura en la que se muestran como gráfico de barras los diferentes valores de correlación.
    """
    if metric == 'pearson':
        matriz_correlaciones = df.select_dtypes(include=['float', 'int']).corrwith(target)
    elif metric == 'spearman':
        matriz_correlaciones = df.select_dtypes(include=['float', 'int']).apply(lambda x: spearmanr(x, target).correlation)
    elif metric == 'kendall':
        matriz_correlaciones = df.select_dtypes(include=['float', 'int']).apply(lambda x: kendalltau(x, target)[0])
    elif metric == 'point_biserial':
        matriz_correlaciones = df.apply(lambda x: pointbiserialr(x, target)[0] if (x.dtypes == 'float64' or x.dtypes == 'int64') and len(x.unique()) == 2 else None)
    elif metric == 'matthews':
        matriz_correlaciones = df.apply(lambda x: matthews_corrcoef(x, target) if x.dtypes == 'int64' else None)
    else:
        raise ValueError("Métrica no válida. Las opciones son 'pearson', 'spearman', 'kendall', 'point_biserial', 'matthews'.")

    correlaciones_df = pd.DataFrame({
        'Variables': matriz_correlaciones.index,
        'Correlación': matriz_correlaciones.values
    })

    fig = px.bar(correlaciones_df, x='Variables', y='Correlación',
                title=f'Correlaciones ({metric.capitalize()}) con la Variable Externa')

    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=list(range(len(correlaciones_df['Variables']))),
                    ticktext=correlaciones_df['Variables'])
    
    fig.update_traces(marker_color='royalblue', marker_line_color='royalblue',
                  marker_line_width=1.5, opacity=0.6)

    fig.update_layout(plot_bgcolor='white', height=600, width=800, yaxis=dict(showgrid=True, gridcolor='lightgray'))

    fig.update_yaxes(range=[-1.01, 1.01], dtick=0.2)

    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity = 0.6)

    return fig


def grafico_distribucion(y):
    cdf_heating = Cdf.from_seq(y)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15))
    ax1.plot(cdf_heating)
    ax1.set_xlabel('Potencia Medida', fontsize=18)
    ax1.set_ylabel('CDF', fontsize=18)
    ax1.set_title('Función de distribución acumulativa', fontsize=18)

    ax = sns.kdeplot(y, fill=False)  # Cambio shade por fill
    kdeline = ax.lines[0]
    mean = y.mean()
    sdev = y.std()
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(mean, xs, ys)
    left = mean - sdev
    right = mean + sdev
    left, right = np.percentile(y, [25, 75])
    ax2.vlines(mean, 0, height, ls=':')

    # ax.fill_between(xs, 0, ys, facecolor='blue', alpha=0.2)
    ax2.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), interpolate=True, alpha=0.2)
    ax2.text(left-7, 0, "Q1")
    ax2.text(right, 0, "Q3")
    plt.xlabel('Potencia Medida', fontsize=18)
    plt.ylabel('KDE', fontsize=18)
    plt.title('Estimación de densidad de kernel', fontsize=18)
    plt.show()
    
    return None


def evaluar_modelo(modelo, nombre_modelo, compresor, X_train, y_train, X_test, y_test, k_folds=5):
    
    metricas_modelo = {}

    # Creamos un objeto KFold para dividir los datos
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Calcular R^2 train
    cv_r2_scores = cross_val_score(modelo, X_train, y_train, cv=kf, n_jobs=-2, scoring='r2')
    cv_r2_score_promedio = np.mean(cv_r2_scores)

    # Calcular RMSE
    cv_rmse_scores = cross_val_score(modelo, X_train, y_train, cv=kf, n_jobs=-2, scoring=make_scorer(mean_squared_error))
    cv_rmse_score_promedio = np.sqrt(np.mean(cv_rmse_scores))

    # Entrenar el modelo con todos los datos de train
    modelo.fit(X_train, y_train)

    # Guardar el modelo en un archivo
    ruta_modelo = os.path.join(f'Modelos/Comp_{compresor}', f'{nombre_modelo}_modelo.pkl')
    with open(ruta_modelo, 'wb') as modelo_archivo:
        pickle.dump(modelo, modelo_archivo)

    # Realizar predicciones en los datos de prueba
    y_pred = modelo.predict(X_test)
    print(y_pred)
    
    # Calcular el coeficiente de determinación (R^2) en los datos de prueba
    test_r2 = r2_score(y_test, y_pred)
    
    # Calcular el error cuadrático medio (RMSE) en los datos de prueba
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{nombre_modelo} - TRAIN = R^2: {cv_r2_score_promedio}, RMSE: {cv_rmse_score_promedio}\n TEST = R^2: {test_r2}, RMSE: {test_rmse}")

    metricas_modelo = {
        'cv_r2_score': cv_r2_score_promedio,
        'cv_rmse_score': cv_rmse_score_promedio,
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }

    # Agregar análisis de residuos
    visualizer = ResidualsPlot(modelo, hist=False, qqplot=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()

    return metricas_modelo