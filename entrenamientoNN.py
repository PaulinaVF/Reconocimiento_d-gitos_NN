# Red neuronal entrenamiento
# Paulina Vara 2020
import math
import random
from tkinter.filedialog import asksaveasfile

errorFound = True

def leerBD_Numeros (archivo):
    numeros = []
    etiquetas = []
    linesInFile = archivo.readlines()

    def split(word):
        return [char for char in word]

    # A cada línea que se leyó se le aplica el siguiente procedimiento
    archivo_linea = 0
    numero_linea = 0
    numeros_leidos = 0
    for line in linesInFile:
        if(numero_linea == 0):
            numeros.append([])  # Crea espacio para un número
        if(numero_linea == 32): # Cuando termina de leer el número (de 32x32 pixeles) lee la etiqueta
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

# Se genera un conjunto de entrenamiento con una BD de 1934 números
numeros, etiquetas = leerBD_Numeros(open("Entrenamiento.txt", 'r'))
numerosPropios, etiquetasPropios = leerBD_Numeros(open("En formato 0s y 1s.txt","r"))
numeros.extend(numerosPropios)
etiquetas.extend(etiquetasPropios)

trainingSamples = []
trainingSamples.append(numeros)
trainingSamples.append(etiquetas)
coef_Aprendizaje = 0.1
n_in = 1024 # Pues son imágenes de 32 x 32
n_hid = 20 # Número de neuronas en la capa oculta
n_out = 10 # Número de salidas (cada una representa un número 0-9)

# Los pesos se van a guardar en dos listas diferentes, la que afecta la entrada de la capa oculta y la que afecta la entrada de la capa de salida
# Para respetar los índices y no perder de dónde a dónde se considera el peso se trabajarán como "matrices" para representar los índices i y j
w_hidInput = [[0.0 for i in range(n_hid)] for j in range(n_in+1)]
w_outInput = [[0.0 for i in range(n_out)] for j in range(n_hid+1)]

# Se generan pesos aleatorios entre -1 y 1 para la red neuronal
for i in range(len(w_hidInput)):
    for j in range(len(w_hidInput[i])):
        w_hidInput[i][j] = random.uniform(-1, 1)
for i in range(len(w_outInput)):
    for j in range(len(w_outInput[i])):
        w_outInput[i][j] = random.uniform(-1, 1)


while(errorFound):
    tam_muestreo = len(trainingSamples[1])
    # Variables para medir la acertividad
    cantidad_aciertos = 0
    porcentaje_acierto = 0.0
    # Se comienza asumiendo que no hay error y al encontrar error la variable se volverá verdadera
    errorFound = False
    for i_sample in range(tam_muestreo):
        # Se crea el espacio para los cálculos que requiere la red
        f_hid, o_hid = [0.0 for i in range(n_hid)], [0.0 for i in range(n_hid)]
        f_out, o_out= [0.0 for i in range(n_out)], [0.0 for i in range(n_out)]
        # Forward
        for i in range(len(f_hid)):
            f_hid[i] += (-1*w_hidInput[0][i])
            j = 0
            while (j < n_in):
                f_hid[i] += trainingSamples[0][i_sample][j]*w_hidInput[j+1][i]
                j += 1
        for i in range(len(o_hid)):
            o_hid[i] = 1/(1+math.exp(-f_hid[i]))

        for i in range(len(f_out)):
            f_out[i] += (-1 * w_outInput[0][i])
            j = 0
            while (j < n_hid):
                f_out[i] += o_hid[j] * w_outInput[j+1][i]
                j += 1
        for i in range(len(o_out)):
            o_out[i] = 1/(1+math.exp(-f_out[i]))

        # Comprobar si hubo error
        solucionReal = trainingSamples[1][i_sample].index(max(trainingSamples[1][i_sample]))
        redondeoOuts = []
        for elemento in o_out:
            redondeoOuts.append(round(elemento))
        if(sum(redondeoOuts) != 1):
            solucionPropuesta = -1
        else:
            solucionPropuesta = redondeoOuts.index(1)

        if(solucionPropuesta == solucionReal):
            esteOutBien = True
            cantidad_aciertos += 1
        else:
            errorFound = True
            esteOutBien = False
            # Backward:
            error_out = [0.0 for i in range(n_out)]
            error_hid = [0.0 for i in range(n_hid)]

            for i in range(n_out):
                error_out[i] = o_out[i]*(1-o_out[i])*(trainingSamples[1][i_sample][i]-o_out[i])

            for i in range(n_hid):
                sum_pesoError = 0.0
                for j in range(n_out):
                    sum_pesoError += (w_outInput[i+1][j] * error_out[j])
                error_hid[i] = o_hid[i]*(1-o_hid[i])*sum_pesoError

            for j in range(n_hid):
                w_hidInput[0][j] += (coef_Aprendizaje*error_hid[j]*(-1))
                for i in range(n_in):
                    w_hidInput[i+1][j] += (coef_Aprendizaje * error_hid[j] * trainingSamples[0][i_sample][i])

            for j in range(n_out):
                w_outInput[0][j] += (coef_Aprendizaje*error_out[j]*-1)
                for i in range(n_hid):
                    w_outInput[i+1][j] += (coef_Aprendizaje * error_out[j] * o_hid[i])

    print(str(cantidad_aciertos)+' de '+str(tam_muestreo))
    porcentaje_acierto = round(cantidad_aciertos / tam_muestreo * 100, 2)
    print(str(porcentaje_acierto)+'%')
    if(porcentaje_acierto > 85):
        errorFound = False

# Para guardar los valores de w
def guardarLista(lista):
    file = asksaveasfile(mode='w', defaultextension=".txt")
    if (file is not None):
        for row in lista:
            for element in row:
                file.write(str(element) + ' ')  # Guarda valor por valor de una fila separándolo por espacios
            file.write('\n')  # Cambia de linea
        file.close()

# Así al terminar permite guardar los pesos que se dan de la capa de entrada a la oculta y de la oculta a la de salida
guardarLista(w_hidInput)
guardarLista(w_outInput)