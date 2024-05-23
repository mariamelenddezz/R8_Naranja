import paho.mqtt.client as paho
import logging
from functions import *

# Crear instancia al subscriber 1
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s > %(name)s > %(levelname)s: %(message)s')

# Guardamos el nombre de los topics a los que suscribirse
topic_1 = "topic_data"
topic_2 = 'topic_queries'

# Definimos handler para poder llamar a las funciones
handler = Handler()

# Conectamos con broker
broker = "localhost"
subscriber_1 = paho.Client(paho.CallbackAPIVersion.VERSION2,"S1")
subscriber_1.on_connect = on_connect
subscriber_1.on_message = handler.on_message
logging.debug(f'Connecting to broker {broker}') 
subscriber_1.connect(broker)

# Se suscribe a los dos topicos
subscriber_1.subscribe(topic_1) 
subscriber_1.subscribe(topic_2)

logging.debug("Subscribed")
logging.debug("Start loop")

subscriber_1.loop_start()

# Mantener el programa principal en ejecución
try:
    while True:
        pass
except KeyboardInterrupt:
    logging.debug('Desconectando...')
    subscriber_1.loop_stop()
    subscriber_1.disconnect()
