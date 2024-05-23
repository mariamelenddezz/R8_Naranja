import json
import numpy as np
import pandas as pd
import paho.mqtt.client as paho
from datetime import datetime, timedelta
import logging
from elasticsearch import Elasticsearch

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s > %(name)s > %(levelname)s: %(message)s')

# Configurar conexión a Elasticsearch
elastic_password = '9oUgwZwyFHXO7mR8HZnk'

es = Elasticsearch('https://localhost:9200',
                   ca_certs = '/home/martin/elasticsearch-8.4.3/config/certs/http_ca.crt',
                   basic_auth = ('elastic',elastic_password))


# Generamos diferentes funciones para las consultas y demas
class Handler:

    def __init__(self):
        self.total_data = []

    def on_message(self, client, userdata, message):
        total_data = self.total_data

        if message.topic == "topic_data":
            record = json.loads(message.payload.decode("utf-8"))

            if record not in total_data:
                total_data.append(record)

        if message.topic == "topic_queries":
            consultas(message.payload.decode('utf-8'), total_data)

def consultas(message, total_data):
    if np.size(total_data) == 0:
        print('No hay datos')
    else:
        if message == "/trafico_por_puerto_de_origen_y_destino":
            resultado = trafico_puerto(total_data)
            print(resultado)
            indice = 'trafico_por_puerto'
            logging.info(f'Indexando en el indice {indice} en Elasticsearch.')
            index_data_in_elasticsearch(indice, resultado)

        elif message == "/trafico_por_accion_y_su_duracion":
            resultado = trafico_accion_duracion(total_data)
            print(resultado)
            indice = 'trafico_por_accion'
            logging.info(f'Indexando en el indice {indice} en Elasticsearch.')
            index_data_in_elasticsearch(indice, resultado)

        elif message == "/proporcion_de_bytes_por_registro":
            resultado = proporcion(total_data)
            print(resultado)
            indice = 'proporcion_de_bytes'
            logging.info(f'Indexando en el indice {indice} en Elasticsearch.')
            index_data_in_elasticsearch(indice, resultado)

def trafico_puerto(total_data):
    df_total = pd.DataFrame(total_data)
    data = df_total.copy()

    data_1 = pd.DataFrame(data.groupby(['Source_Port','Destination_Port']).agg({'Bytes': 'sum','Packets': 'sum'}).reset_index())
    print('')
    logging.info('Se muestra el trafico por puerto de origen y destino:')
    print('')
    return data_1

def trafico_accion_duracion(total_data):
    df_total = pd.DataFrame(total_data).reset_index()
    data = df_total.copy()
    data_2 = data.groupby('Action').agg({'Elapsed Time (sec)': 'mean',
                                        'Bytes': 'sum',
                                        'Packets': 'sum'
                                    }).reset_index().sort_values('Bytes')
    print('')
    logging.info('Se muestra el trafico por la accion del firewall, y su duracion:')
    print('')
    return data_2

def proporcion(total_data):
    df_total = pd.DataFrame(total_data)
    data = df_total.copy()
    data['ratio_bytes_enviados'] = data['Bytes_Sent'] / data['Bytes']
    data['ratio_bytes_recibidos'] = data['Bytes_Received'] / data['Bytes']
    print('')
    logging.info('Se muestra la proporcion de bytes enviados y recibidos por registro:')
    print('')
    return data[['ratio_bytes_enviados', 'ratio_bytes_recibidos']]

def index_data_in_elasticsearch(index_name, data):
    records = data.to_dict(orient='records')
    for record in records:
        es.index(index=index_name, body=record)

def index_data_in_elasticsearch(index_name, data):
    # Eliminar el índice existente para sobrescribir los datos porque sino se acumularian
    # en el mismo indice todoas la veces que se hiciesen nuevas consultas.
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # Crear el índice de nuevo 
    es.indices.create(index=index_name)

    # Indexar los nuevos datos
    records = data.to_dict(orient='records')
    for record in records:
        es.index(index=index_name, body=record)

def on_connect(client, userdata, flags, rc, extra_arg = None):
    logging.debug(f'Connected with result code {rc}')
    client.subscribe("topic_data")
    client.subscribe("topic_queries")








