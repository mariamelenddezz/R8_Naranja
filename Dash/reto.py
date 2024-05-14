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
df_cA = pd.read_csv("Datos/Originales/Comp_A.csv")
df_cB = pd.read_csv("Datos/Originales/Comp_B.csv")
df_cC = pd.read_csv("Datos/Originales/Comp_C.csv")
df_cD = pd.read_csv("Datos/Originales/Comp_D.csv")

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
            value = [60, 80]
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
                    value = [10, 20]
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

        html.Br(),

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
        dcc.Graph(id="grafico-boxplots")

])





#################################### DEFINICIÓN DE LA ESTRUCTURA O LAYOUT DE LA APLICACIÓN #########################################################

app.layout = html.Div([
    html.H1("Información acerca de los diferentes Compresores"), 
    row_filtro0, 
    html.Br(), 
    dcc.Tabs([
        dcc.Tab(label="Información de manera individual", children = tab1),
        dcc.Tab(label="Comparativa entre los diferentes compresores", children=tab2)
    ])
])

#################################### DEFINICIÓN DE LA ESTRUCTURA CALLBACK ##########################################################################

@app.callback([
    Output(component_id="grafico-scatter", component_property="figure"),
    Output(component_id="tabla1", component_property="data"),
    Output(component_id="grafico-boxplots", component_property="figure")], 
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

    df_filtro_boxplot = df_filtro[
        (df_filtro["Potencia_Medida"] >= potencia[0]) &
        (df_filtro["Potencia_Medida"] <= potencia[1])
    ]

    # Gráfico de dispersión
    fig1 = px.scatter(data_frame=df_filtro_tabla, x="Frecuencia", y="Potencia_Medida", title="Relación entre la Frecuencia y la Potencia")
    
    # Datos para la tabla
    tabla1 = df_filtro_tabla[["Compresor", "Frecuencia", "Potencia_Medida", "Presion", "Temperatura"]].to_dict("records")

    # Gráfico de boxplots
    fig2 = px.box(data_frame=df_filtro_boxplot, y="Presion", facet_col="Compresor", color="Compresor", color_discrete_sequence=['#66FF66', '#00CC00', '#009900', '#004C00'])

    return fig1, tabla1, fig2





#-------------------------------------------------------------------------------------------------------------

# Ejecucción de la app
if __name__ == '__main__':
    app.run_server(debug=False)