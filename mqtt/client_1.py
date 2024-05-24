# Fichero Firewall
ruta = "../Datos/Originales/log2.csv"

from datetime import datetime
from time import sleep
import json
from functions import *
import paho.mqtt.client as paho
import logging
import pandas as pd


# Crear instancia al publisher 1
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s > %(name)s > %(levelname)s: %(message)s')


# 1. Generamos publisher 1
publisher_1 = paho.Client(paho.CallbackAPIVersion.VERSION2,"client-publish", clean_session=True)


# 2. Conectamos cliente 1 con broker 'localhost'
broker = "localhost"
logging.debug(f'Connecting to broker {broker}')
publisher_1.connect(broker)


# 3. Definimos el 'topic_data'
topic_data = 'topic_data'


# 4. Publicacion de los datos de log2.csv al topic_data
# Leemos los datos que se quieren publicar
file_name = ruta
df_log = pd.read_csv(file_name)
df_log = df_log.rename(columns={'Source Port':'Source_Port'})
df_log = df_log.rename(columns={'Destination Port':'Destination_Port'})
df_log = df_log.rename(columns={'Bytes Sent':'Bytes_Sent'})
df_log = df_log.rename(columns={'Bytes Received':'Bytes_Received'})
df_recortado = df_log.head(5000)

for index, row in df_recortado.iterrows():
    # Agregamos la variable timestamp a la vez que se publican
    row['timestamp'] = str(datetime.now())
    record = row.to_dict()
    
    publisher_1.publish(topic_data, json.dumps(record))
    logging.info(f"Enviando a topic_data: {record}")
    sleep(1)

# Desconectamos el cliente una vez que se han publicado todas las filas
logging.debug('Publicación completada pero el cliente sigue conectado.')