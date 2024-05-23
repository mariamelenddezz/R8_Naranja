import paho.mqtt.client as paho
import logging
from functions import * 

# Crear instancia al publisher 2
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s > %(name)s > %(levelname)s: %(message)s')

# 1. Generamos publisher 2
publisher_2 = paho.Client(paho.CallbackAPIVersion.VERSION2, "P2", clean_session=True)

# 2. Conectamos publisher 2 con broker 'localhost'
broker = "localhost"
logging.debug(f'Connecting to broker {broker}')
publisher_2.connect(broker)

# 3. Definimos el 'topic_queries'
topic_2 = 'topic_queries'

# 4. Publicacion del nombre de la query elegida segun el usuario
while True:

    print("\nConsultas disponibles:\n \n 1- /trafico_por_puerto_de_origen_y_destino \n 2- /trafico_por_accion_y_su_duracion \n" 
        " 3- /proporcion_de_bytes_por_registro \n")
        

    query = input(
                '\nSeleccione una consulta: ')

    message = query

    publisher_2.publish(topic_2, message, qos=1)
    
    logging.info("\nEnviando a transact_queries: " + message)