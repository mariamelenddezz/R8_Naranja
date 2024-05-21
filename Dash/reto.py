#-------------------------------------------------------------------------------------------------------------

#Cargamos las librerías
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash import dash_table
from dash.dependencies import Input, Output



#-------------------------------------------------------------------------------------------------------------

#Carga de los datos y creación de df único
df_cA = pd.read_csv("../Datos/Originales/Comp_A.csv")
df_cB = pd.read_csv("../Datos/Originales/Comp_B.csv")
df_cC = pd.read_csv("../Datos/Originales/Comp_C.csv")
df_cD = pd.read_csv("../Datos/Originales/Comp_D.csv")

# Agregar una columna de identificador para cada DataFrame
df_cA['Compresor'] = 'CompresorA'
df_cB['Compresor'] = 'CompresorB'
df_cC['Compresor'] = 'CompresorC'
df_cD['Compresor'] = 'CompresorD'

# Concatenar los DataFrames
df_compresores = pd.concat([df_cA, df_cB, df_cC, df_cD], ignore_index=True)

#-------------------------------------------------------------------------------------------------------------

# Definimos hoja de estilo externa
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Le asignamos la hoja de estilo a nuestra app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#-------------------------------------------------------------------------------------------------------------

row_filtro0 = html.Div([
     html.Div([
        html.Label("Selecciona un rango de Potencia:"), 
        dcc.RangeSlider(
            id="filtro-potencia", 
            min = df_compresores["Potencia_Medida"].min(), 
            max = df_compresores["Potencia_Medida"].max(), 
            step = 10, 
            value = [60, 90]
        )
    ])
])

tab1 = html.Div([
    html.Div([
        # Row 1: Filtros
        html.Div([
            html.Div([
                html.Label("Selecciona un rango de temperatura:"),
                dcc.RangeSlider(
                    id="filtro-temp", 
                    min=df_compresores["Temperatura"].min(), 
                    max=df_compresores["Temperatura"].max(),
                    step= 10,
                    value = [10, 30]
                )
            ], className="six columns"),

            html.Div([
                html.Label("Selecciona el compresor:"),
                dcc.Dropdown(
                    id="filtro-compresores", 
                    options = [
                        {"label": "A", "value":"CompresorA"}, 
                        {"label": "B", "value":"CompresorB"},
                        {"label": "C", "value":"CompresorC"},
                        {"label": "D", "value":"CompresorD"},
                    ], 
                    value = "CompresorA"
                )
            ], className="six columns"),
        ], className="row"), 

        # Row 2: Gráficos
        html.Div([
            html.Div([
                dcc.Graph(id="grafico-scatter")
            ], className="seven columns"), 

            html.Div([
                dash_table.DataTable(
                    id = "tabla1", 
                    columns =[{"name":i, "id":i} for i in df_compresores[["Compresor","Frecuencia", "Potencia_Medida", "Presion", "Temperatura"]].columns],
                    data = df_compresores[["Compresor","Frecuencia", "Potencia_Medida", "Presion", "Temperatura"]].to_dict("records"), 
                    filter_action="native", 
                    sort_action="native", 
                    sort_mode="multi", 
                    page_size=10
                )
            ], className = "five columns")
        ], className="row")
    ])
])

tab2 = html.Div([
    html.Div([
        html.Label("Selecciona una variable:"),
        dcc.Dropdown(
            id="filtro-variable", 
            options=[
                {"label": "Frecuencia", "value": "Frecuencia"},
                {"label": "Potencia Medida", "value": "Potencia_Medida"},
                {"label": "Presión", "value": "Presion"},
                {"label": "Temperatura", "value": "Temperatura"}
            ], 
            value="Temperatura"
        )
    ], className="twelve columns"),
    
    html.Div([
        # Primera fila
        html.Div([
            dcc.Graph(id="grafico-boxplots-multiples")
        ], className="twelve columns"), 
        
        # Segunda fila
        html.Div([
            html.Div([
                dcc.Graph(id="grafico-boxplotA")
            ], className="six columns"), 

            html.Div([
                dcc.Graph(id="grafico-histA")
            ], className="six columns")
        ], className="row"),
        
        # Tercera fila
        html.Div([
            html.Div([
                dcc.Graph(id="grafico-boxplotB")
            ], className="six columns"), 

            html.Div([
                dcc.Graph(id="grafico-histB")
            ], className="six columns")
        ], className="row"),

        # Cuarta fila
        html.Div([
            html.Div([
                dcc.Graph(id="grafico-boxplotC")
            ], className="six columns"), 

            html.Div([
                dcc.Graph(id="grafico-histC")
            ], className="six columns")
        ], className="row"),
        
        # Quinta fila
        html.Div([
            html.Div([
                dcc.Graph(id="grafico-boxplotD")
            ], className="six columns"), 

            html.Div([
                dcc.Graph(id="grafico-histD")
            ], className="six columns")
        ], className="row")
    ], className="row")
])




#################################### DEFINICIÓN DE LA ESTRUCTURA O LAYOUT DE LA APLICACIÓN #########################################################

app.layout = html.Div([
    html.H1("Información acerca de los diferentes Compresores"), 
    row_filtro0, 
    html.Br(), 
    dcc.Tabs([
        dcc.Tab(label="Información de manera individual", children = tab1),
        dcc.Tab(label="Comparaciones entre compresores", children=tab2)
    ])
])

#################################### DEFINICIÓN DE LA ESTRUCTURA CALLBACK 1 ##########################################################################

@app.callback([
    Output(component_id="grafico-scatter", component_property="figure"),
    Output(component_id="tabla1", component_property="data")
   ], 
    [
        Input(component_id="filtro-potencia", component_property="value"),
        Input(component_id="filtro-compresores", component_property="value"),
        Input(component_id="filtro-temp", component_property="value")
    ]
)
def actualizar_graficos(potencia, compresor, temp):
    df_filtro = df_compresores.copy()

    df_filtro_tabla = df_filtro[
        (df_filtro["Potencia_Medida"] >= potencia[0]) &
        (df_filtro["Potencia_Medida"] <= potencia[1]) &
        (df_filtro["Compresor"] == compresor) &
        (df_filtro["Temperatura"] >= temp[0]) &
        (df_filtro["Temperatura"] <= temp[1])
    ]

    # Gráfico de dispersión
    fig1 = px.scatter(data_frame=df_filtro_tabla, x="Frecuencia", y="Potencia_Medida", title="Relación entre la Frecuencia y la Potencia")
    
    # Datos para la tabla
    tabla1 = df_filtro_tabla[["Compresor", "Frecuencia", "Potencia_Medida", "Presion", "Temperatura"]].to_dict("records")

    
    return fig1, tabla1

#################################### DEFINICIÓN DE LA ESTRUCTURA CALLBACK 2 ##########################################################################
@app.callback([
    Output(component_id="grafico-boxplots-multiples", component_property="figure"),
    Output(component_id="grafico-boxplotA", component_property="figure"),
    Output(component_id="grafico-histA", component_property="figure"),
    Output(component_id="grafico-boxplotB", component_property="figure"),
    Output(component_id="grafico-histB", component_property="figure"),
    Output(component_id="grafico-boxplotC", component_property="figure"),
    Output(component_id="grafico-histC", component_property="figure"),
    Output(component_id="grafico-boxplotD", component_property="figure"),
    Output(component_id="grafico-histD", component_property="figure")
], 
[
    Input(component_id="filtro-variable", component_property="value"),
    Input(component_id="filtro-potencia", component_property="value")
])
def actualizar_graficos(variable, potencia):
    df_compresores_filtro = df_compresores[
        (df_compresores["Potencia_Medida"] >= potencia[0]) &
        (df_compresores["Potencia_Medida"] <= potencia[1])
    ]

    # Filtrar dataframes específicos de cada compresor
    df_cA_filtro = df_cA[
        (df_cA["Potencia_Medida"] >= potencia[0]) &
        (df_cA["Potencia_Medida"] <= potencia[1])
    ]

    df_cB_filtro = df_cB[
        (df_cB["Potencia_Medida"] >= potencia[0]) &
        (df_cB["Potencia_Medida"] <= potencia[1])
    ]

    df_cC_filtro = df_cC[
        (df_cC["Potencia_Medida"] >= potencia[0]) &
        (df_cC["Potencia_Medida"] <= potencia[1])
    ]

    df_cD_filtro = df_cD[
        (df_cD["Potencia_Medida"] >= potencia[0]) &
        (df_cD["Potencia_Medida"] <= potencia[1])
    ]


    # Crear gráficos para cada DataFrame filtrado
    fig_multiple = px.ecdf(data_frame=df_compresores_filtro, x=variable, title=f"Distribución de {variable} en todos los compresores", facet_col="Compresor", color="Compresor", color_discrete_sequence=["#FFB6C1", "#32CD32", "#4682B4", "#E9967A"])

    fig_boxplotA = px.box(data_frame=df_cA_filtro, y=variable, title=f"Distribución de {variable} en Compresor A", color_discrete_sequence=["#FFB6C1"])
    fig_histA = px.histogram(data_frame=df_cA_filtro, x=variable, title=f"Histograma de {variable} en Compresor A", color_discrete_sequence=["#FFB6C1"])

    fig_boxplotB = px.box(data_frame=df_cB_filtro, y=variable, title=f"Distribución de {variable} en Compresor B", color_discrete_sequence=["#32CD32"])
    fig_histB = px.histogram(data_frame=df_cB_filtro, x=variable, title=f"Histograma de {variable} en Compresor B", color_discrete_sequence=["#32CD32"])

    fig_boxplotC = px.box(data_frame=df_cC_filtro, y=variable, title=f"Distribución de {variable} en Compresor C", color_discrete_sequence=["#4682B4"])
    fig_histC = px.histogram(data_frame=df_cC_filtro, x=variable, title=f"Histograma de {variable} en Compresor C", color_discrete_sequence=["#4682B4"])

    fig_boxplotD = px.box(data_frame=df_cD_filtro, y=variable, title=f"Distribución de {variable} en Compresor D", color_discrete_sequence=["#E9967A"])
    fig_histD = px.histogram(data_frame=df_cD_filtro, x=variable, title=f"Histograma de {variable} en Compresor D", color_discrete_sequence=["#E9967A"])

    return fig_multiple, fig_boxplotA, fig_histA, fig_boxplotB, fig_histB, fig_boxplotC, fig_histC, fig_boxplotD, fig_histD



#-------------------------------------------------------------------------------------------------------------

# Ejecucción de la app
if __name__ == '__main__':
    app.run_server(debug=False)