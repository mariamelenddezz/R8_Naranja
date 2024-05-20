#-------------------------------------------------------------------------------------------------------------

#Cargamos las librerías
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash import dash_table
from dash.dependencies import Input, Output
import funciones as fun
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from empiricaldist import Cdf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, svm
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from yellowbrick.regressor import ResidualsPlot
import pygad
import random
import math
import matplotlib.pyplot as plt
import sql


#-------------------------------------------------------------------------------------------------------------

#Optimización (Carga de modelos)
with open('Modelos/Comp_a/RandomForestRegressor_modelo.pkl', 'rb') as archivo:
    modelo_a = pickle.load(archivo)
with open('Modelos/Comp_b/RandomForestRegressor_modelo.pkl', 'rb') as archivo:
    modelo_b = pickle.load(archivo)
with open('Modelos/Comp_c/RandomForestRegressor_modelo.pkl', 'rb') as archivo:
    modelo_c = pickle.load(archivo)
with open('Modelos/Comp_d/RandomForestRegressor_modelo.pkl', 'rb') as archivo:
    modelo_d = pickle.load(archivo)


with open('Modelos/Comp_a/scaler_model.pkl', 'rb') as archivo:
    scaler_a = pickle.load(archivo)
with open('Modelos/Comp_b/scaler_model.pkl', 'rb') as archivo:
    scaler_b = pickle.load(archivo)
with open('Modelos/Comp_c/scaler_model.pkl', 'rb') as archivo:
    scaler_c = pickle.load(archivo)
with open('Modelos/Comp_d/scaler_model.pkl', 'rb') as archivo:
    scaler_d = pickle.load(archivo)

condiciones_iniciales = [25, 25, 25, 25]

# Generamos la poblacion inicial
random.seed(42) #semilla para que en todas las ejecuciones la poblacion inicial sea la misma
poblacion_inicial = []
poblacion_inicial.append(condiciones_iniciales)
for _ in range(50):
    individuo = []
    for _ in range(4):
        gen = random.uniform(1, 90)
        individuo.append(gen)
    poblacion_inicial.append(individuo)


    #definimos la funcion fitness
def fitness_func(ga_instance, solution, solution_idx):
    capacidad = np.array([100, 90, 95, 110])
    frecuencia_porcentaje = [numero / 100 for numero in solution]
    volumen_aire  = capacidad * frecuencia_porcentaje
    if sum(volumen_aire) >= objetivo:
        comp_a = pd.DataFrame({
        'Presion': [presion_comp_a],
        'Temperatura': [temp_comp_a],
        'Frecuencia': solution[0]
        })
        X = scaler_d.transform(comp_a)
        X = pd.DataFrame(X, columns=comp_a.columns)
        potencia_a = modelo_a.predict(X)
        comp_b = pd.DataFrame({
        'Presion': [presion_comp_b],
        'Temperatura': [temp_comp_b],
        'Frecuencia': solution[1]
        })
        X = scaler_d.transform(comp_b)
        X = pd.DataFrame(X, columns=comp_b.columns)
        potencia_b = modelo_b.predict(X)
        comp_c = pd.DataFrame({
        'Presion': [presion_comp_c],
        'Temperatura': [temp_comp_c],
        'Frecuencia': solution[2]
        })
        X = scaler_d.transform(comp_c)
        X = pd.DataFrame(X, columns=comp_c.columns)
        potencia_c = modelo_c.predict(X)
        comp_d = pd.DataFrame({
        'Presion': [presion_comp_d],
        'Temperatura': [temp_comp_d],
        'Frecuencia': solution[3]
        })
        X = scaler_d.transform(comp_d)
        X = pd.DataFrame(X, columns=comp_d.columns)
        potencia_d = modelo_d.predict(X)
        fitness = (potencia_a + potencia_b + potencia_c + potencia_d)
        penalizacion = sum(volumen_aire) - objetivo
        fitness = (fitness)+math.exp(penalizacion)
        for i in range(len(solution)):
            if solution[i] != condiciones_iniciales[i]:
                fitness +=3
        fitness = -fitness
        fitness = float(fitness[0])
    else:
        fitness = -10e99
    return fitness


num_generations = 100000
num_parents_mating = len(poblacion_inicial)//2
num_genes = len(poblacion_inicial[0])
mutation_percent_genes = 1
mutation_probability = 0.1
keep_parents = 2
gene_space = range(1, 90)
parent_1 = 'ramdom'
crossover_1 = 'two_points'
mutation_1 = 'swap'


def on_stop(ga_instance, last_population_fitness):
    print(f"El algoritmo genético ha finalizado.")
    print("Mejor solución encontrada:", ga_instance.best_solution())

def on_generation(ga_instance):
    print("Generación {generation}: Mejor solucion = {best_solution} Mejor fitness = {best_fitness}".format(generation=ga_instance.generations_completed,
                                                                          best_solution = ga_instance.best_solution()[0],
                                                                          best_fitness = ga_instance.best_solution()[1]))
    
ga_instance_1 = pygad.GA(fitness_func=fitness_func,
                       num_generations=100000, 
                       num_parents_mating=num_parents_mating,
                       num_genes=num_genes,
                       parent_selection_type=parent_1,
                       crossover_type=crossover_1,
                       mutation_type= mutation_1,
                       gene_space=gene_space,
                       mutation_percent_genes = mutation_percent_genes,
                       mutation_probability=mutation_probability,
                       initial_population=poblacion_inicial,
                       gene_type=float,
                       on_stop=on_stop,
                       on_generation=on_generation,
                       save_solutions=True,
                       stop_criteria="saturate_25")
ga_instance_1.run()



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
       
       #Primera Row
       html.Div([
           dcc.Input(type = 'number', placeholder = 'Presion', id = 'filtro_presion')
       ], className="four columns"), 

       html.Div([
           dcc.Input(type="number", placeholder = "Temperatura", id = "filtro_temperatura")
       ], className = "four columns"), 

       html.Div([
           dcc.Input(type="number", placeholder = "Volumen de aire", id="filtro_aire")
       ], className="four columns"), 

       #Segunda Row
       html.Div([
           dcc.Graph(id="compresor_A")
       ], className="six columns"), 

       html.Div([
           dcc.Graph(id="compresor_B")
       ], className="six columns"), 
       
       #Tercera Row
       html.Div([
           dcc.Graph(id="compresor_C")
       ], className="six columns"), 

       html.Div([
           dcc.Graph(id="compresor_D")
       ], className="six columns"), 
       

], className="row"),

       



#################################### DEFINICIÓN DE LA ESTRUCTURA O LAYOUT DE LA APLICACIÓN #########################################################

app.layout = html.Div([
    html.H1("Información acerca de los diferentes Compresores"), 
    row_filtro0, 
    html.Br(), 
    dcc.Tabs([
        dcc.Tab(label="Información de manera individual", children = tab1),
        dcc.Tab(label="Optimización del aire acondicionado", children=tab2)
    ])
])

#################################### DEFINICIÓN DE LA ESTRUCTURA CALLBACK ##########################################################################

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





# callback del segundo tab
@app.callback([
    Output(component_id="compresor_A", component_property="figure"), 
       Output(component_id="compresor_B", component_property="figure"), 
          Output(component_id="compresor_C", component_property="figure"), 
             Output(component_id="compresor_D", component_property="figure"), 
], 
[
    Input(component_id = "fitro_presion", component_property="value"), 
     Input(component_id = "filtro_temperatura", component_property="value"), 
      Input(component_id = "filtro_aire", component_property="value"), 
]
)
def frecuencia(presion, temperatura, aire):
    presion_comp_a = presion
    temp_comp_a = temperatura
    presion_comp_b = presion
    temp_comp_b = temperatura
    presion_comp_c = presion
    temp_comp_c = temperatura
    presion_comp_d = presion
    temp_comp_d = temperatura
    objetivo = aire
    ga_instance_1 = pygad.GA(fitness_func=fitness_func,
                       num_generations=100000, 
                       num_parents_mating=num_parents_mating,
                       num_genes=num_genes,
                       parent_selection_type=parent_1,
                       crossover_type=crossover_1,
                       mutation_type= mutation_1,
                       gene_space=gene_space,
                       mutation_percent_genes = mutation_percent_genes,
                       mutation_probability=mutation_probability,
                       initial_population=poblacion_inicial,
                       gene_type=float,
                       on_stop=on_stop,
                       on_generation=on_generation,
                       save_solutions=True,
                       stop_criteria="saturate_25")
    ga_instance_1.run()
    comp_a = ga_instance_1.best_solution()[0][0]
    comp_b = ga_instance_1.best_solution()[0][1]
    comp_c = ga_instance_1.best_solution()[0][2]
    comp_d = ga_instance_1.best_solution()[0][3]





#-------------------------------------------------------------------------------------------------------------

# Ejecucción de la app
if __name__ == '__main__':
    app.run_server(debug=False)