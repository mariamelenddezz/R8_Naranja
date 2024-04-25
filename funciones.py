import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.stats import spearmanr, kendalltau, pointbiserialr
from sklearn.metrics import matthews_corrcoef



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