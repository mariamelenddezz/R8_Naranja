import sqlite3 as sqlite

base_datos = "compresores.db"

def crear_bbdd_tabla():
    conection = sqlite.connect(base_datos)
    cur = conection.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS compresores(
	temperatura FLOAT NOT NULL,
	presion FLOAT NOT NULL,
    volumen_aire FLOAT NOT NULL,
    compresor_A FLOAT NOT NULL,
    compresor_B FLOAT NOT NULL,
    compresor_C FLOAT NOT NULL,
    compresor_D FLOAT NOT NULL,
    )
    """)
    conection.close()
    return None


def insertar_valores(temperatura:float, presion:float, volumen_aire:float, compresor_A:float, compresor_B:float, compresor_C:float, compresor_D:float):
    con = sqlite.connect(base_datos) # Abrir conexión
    cur = con.cursor() # Generamos un CURSOR. Necesario para ejecutar sentencias SQL
    cur.execute("""INSERT INTO compresores VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (temperatura, presion, volumen_aire, compresor_A, compresor_B, compresor_C, compresor_D))
    con.commit() # Necesario hacer COMMIT para que se guarden los cambios
    con.close()
    return None


def consultar_todos():
    con = sqlite.connect(base_datos) # Abrir conexión
    cur = con.cursor() # Generamos un CURSOR. Necesario para ejecutar sentencias SQL
    cur.execute("SELECT * FROM compresores")
    resultados = cur.fetchall()
    con.close()
    return resultados