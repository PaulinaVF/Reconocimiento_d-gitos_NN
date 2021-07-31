# Probar funcionamiento red neuronal
# Paulina Vara 2020
import math
# Como se guardaron los pesos al entrenar la red, ahora se necesita solo leerlos
def leerPesos (archivo):
    linesInFile = archivo.readlines()  # saves the data from the opened
    weights = []
    for line in linesInFile:
        line = list(map(float, line.split()))
        weights.append(line)
    return (weights)
# Se tiene una base de datos en .txt para probar el roconocimiento:
def leerBD_Numeros (archivo):
    numeros = []
    etiquetas = []
    linesInFile = archivo.readlines()  # saves the data from the opened

    def split(word):
        return [char for char in word]

        # A cada línea que se leyó se le aplica el siguiente procedimiento
    archivo_linea = 0
    numero_linea = 0
    numeros_leidos = 0
    for line in linesInFile:
        if(numero_linea == 0):
            numeros.append([])  # Creamos la "lista de listas" y en cada una de esas listas:
        if(numero_linea == 32):
            etiqueta = [0 for i in range(10)]
            etiqueta[int(line)] = 1
            etiquetas.append(etiqueta)
            numeros_leidos += 1
            numero_linea = 0
        else:
            list_chars = split(line)  # Se separan los caracteres dentro de la lista
            list_chars.pop()
            numeros[numeros_leidos].extend(list(map(int, list_chars)))  # Convierte todos los caracteres leídos a enteros
            archivo_linea += 1
            numero_linea +=1
    return (numeros, etiquetas)

numeros, etiquetas = leerBD_Numeros(open("Prueba.txt", 'r'))
testingSamples = []
testingSamples.append(numeros)
testingSamples.append(etiquetas)
n_in = 1024 # Pues son imágenes de 32 x 32
n_hid = 20 # Número de capas ocultas
n_out = 10 # Número de salidas
# Se leen los valores de pesos guardados
w_hidInput = leerPesos(open('weights to hidden layer.txt', 'r'))
w_outInput = leerPesos(open('weights to output layer.txt', 'r'))

tam_muestreo = len(testingSamples[1])

# Se utilizará una matriz de confusión para ver los resultados
matrizConf = []
for j in range(10):
    muestra = [0 for i in range(10)]
    matrizConf.append(muestra)
# En esta matriz [x][y] .. x representa el número que esperaba & y el número obtenido

# Variables para medir la acertividad total:
cantidad_aciertos = 0
porcentaje_acierto = 0.0
for i_sample in range(tam_muestreo):
    # Se crea el espacio para los cálculos que requiere la red
    f_hid, o_hid = [0.0 for i in range(n_hid)], [0.0 for i in range(n_hid)]
    f_out, o_out = [0.0 for i in range(n_out)], [0.0 for i in range(n_out)]
    # Se aplica solo el proceso "forward" pues solo va a reconocer
    for i in range(len(f_hid)):
        f_hid[i] += (-1 * w_hidInput[0][i])
        j = 0
        while (j < n_in):
            f_hid[i] += testingSamples[0][i_sample][j] * w_hidInput[j + 1][i]
            j += 1
    for i in range(len(o_hid)):
        o_hid[i] = 1 / (1 + math.exp(-f_hid[i]))

    for i in range(len(f_out)):
        f_out[i] += (-1 * w_outInput[0][i])
        j = 0
        while (j < n_hid):
            f_out[i] += o_hid[j] * w_outInput[j + 1][i]
            j += 1
    for i in range(len(o_out)):
        o_out[i] = 1 / (1 + math.exp(-f_out[i]))

    # Comparar los resultados obtenidos con los esperados:
    solucionReal = testingSamples[1][i_sample].index(max(testingSamples[1][i_sample]))
    solucionPropuesta = o_out.index(max(o_out))
    if (solucionPropuesta == solucionReal):
        cantidad_aciertos += 1
    matrizConf[solucionReal][solucionPropuesta] += 1

porcentaje_acierto = round(cantidad_aciertos / tam_muestreo * 100, 2)
print('Clasificados correctamente: ' + str(porcentaje_acierto) + '%')
print('Matriz de confusión:\n - Renglón = Número correcto\n - Columna = Número identificado')
for line in matrizConf:
    print(line)