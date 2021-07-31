# Paulina Vara Figueroa 2020
# Reconocimiento de dígitos con red neuronal
from scipy import ndimage
import tkinter as tk
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import PIL
import math

def filterImage(imgOrigFormat):  # , imgName, imgRoute):
    global img
    # Primero se convierte la imagen del formato leído al formato de opencv para poder trabajar con ella
    pil_image = PIL.Image.open(imgOrigFormat)
    img = np.array(pil_image)
    # Se reduce el tamaño para trabajar mejor con la imagen (utilizando escala para no afectar la forma del número)
    w, h = pil_image.size
    if (h >= w):
        scale_percent = 150 / h * 100
        width = int(w * scale_percent / 100)
        height = 150
        dsize = (width, height)
        img = cv2.resize(img, dsize)
    else:
        scale_percent = 150 / w * 100
        width = 150
        height = int(h * scale_percent / 100)
        dsize = (width, height)
        img = cv2.resize(img, dsize)
    # Limieza de imagen:
    # - 1: Convertir a escala de grises
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # - 2: Aplicar filtrado
    img = cv2.bilateralFilter(img, 7 , 75, 75)
    # - 3: Dilatar la imagen para eliminar ruido
    #      - 3.1: El programa está considerando la imagen invertida, da un efecto de "erosión" al número
    kernel = np.ones((3, 3), np.uint8)
    dilatation_dst = cv2.dilate(img, kernel)
    eroNumber = dilatation_dst
    # - 4: Erosionar la imagen para recuperar información
    #      - 4.1: El programa está considerando la imagen invertida, da un efecto de "dilatación" al número
    kernel = np.ones((5 , 5), np.uint8)
    img = cv2.erode(dilatation_dst, kernel)
    dilNumber = cv2.erode(dilatation_dst, kernel)
    # - 5: Convertir a una imagen binaria
    ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV, dst=img)
    firstBinary = img
    img_dim = 32  # Se busca imagen de 32x32
    # - 6: Obtener el centro de masa de la imagen
    M = cv2.moments(thresh)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # - 7: Obtener las dimensionas aproximadas solo del número
    y, x, w, h = cv2.boundingRect(thresh)
    # Utilizando el centro de masa y las dimensiones del número se extrae solo el número de la imagen
    y = cy - math.ceil(h / 2)
    if (y < 0):
        h += y
        y -= y
    x = cx - math.ceil(w / 2)
    if (x < 0):
        w += x
        x -= x
    crop_img = img[y + 1:y + h, x + 1:x + w]  # Número ya recortado
    # Ahora se busca una imagen de 32x32 con el número centrado y abarcando la altura, para eso se siguen los pasos:
    # - 1: se ajusta el número que se recortó (crop_img) a que ninguna de sus dimensiones supere 32 (img_dim = 32)
    #     - 1.1: no se hace un cambio de tamaño directo a 32x32 para no afectar la forma del número
    #     - 1.2: entonces se escala el número conforme al lado (h ó w) haciendo que este tome el valor de 32
    if (h >= w):
        scale_percent = img_dim / h * 100
        # calcula el cabio de tamaño a escala para no afectar la forma del número
        width = int(crop_img.shape[1] * scale_percent / 100)
        height = img_dim
        dsize = (width, height)
        output = cv2.resize(crop_img, dsize)
    else:
        scale_percent = img_dim / w * 100
        width = img_dim
        # calcula el cabio de tamaño a escala para no afectar la forma del número
        height = int(crop_img.shape[0] * scale_percent / 100)
        dsize = (width, height)
        crop_img = cv2.resize(crop_img, dsize)
        dsize = (crop_img.shape[1], img_dim)
        output = cv2.resize(crop_img, dsize, interpolation=cv2.INTER_AREA)

    # 2 - se binariza la imagen para trabajar solos con 0 y 255
    ret, thresh = cv2.threshold(output, 100, 255, cv2.THRESH_BINARY)

    # 3 - si la imagen quedó rectangular, se rellenan los extremos con 0's para hacerla cuadrada:
    newThresh = []
    for j in range(len(thresh)):
        tam_relleno = img_dim - len(thresh[j])
        tam_rellenoIzq = math.floor(tam_relleno / 2)
        tam_rellenoDer = tam_relleno - tam_rellenoIzq
        lineaConRelleno = []
        for i in range(tam_rellenoIzq):
            lineaConRelleno.extend([0])
        numeric = list(map(int, list(thresh[j])))
        lineaConRelleno.extend(numeric)
        for i in range(tam_rellenoDer):
            lineaConRelleno.extend([0])
        newThresh.append(lineaConRelleno)

    # 4 - la imagen en el formato deseado se encuentra en "newThresh" y esta es la salida de la función
    return newThresh, dilNumber, eroNumber, firstBinary

def cargarImagen(prevWind):
    global ax, ax2, ax3, ax4, filteredImg, dilImg, eroImg, firstBinary
    img = askopenfilename()
    try:
        pil_image = PIL.Image.open(img)
    except:
        doNothing = True
    else:
        prevWind.destroy()
        loadWind = tk.Tk()
        loadWind.title('Reconocimiento')
        # Se abre la imagen original y se envía a ser filtrada
        originalImg = np.array(pil_image)
        filteredImg, dilImg, eroImg, firstBinary = filterImage(img)

        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.set_title('Erosión')
        ax2 = fig.add_subplot(222)
        ax2.set_title('Dilatación')
        ax3 = fig.add_subplot(223)
        ax3.set_title('Imagen binaria')
        ax4 = fig.add_subplot(224)
        ax4.set_title('Ajuste por centro de masa')
        ax.imshow(eroImg, cmap=plt.cm.gray)
        ax2.imshow(dilImg, cmap=plt.cm.gray)
        ax3.imshow(firstBinary, cmap=plt.cm.gray)
        ax4.imshow(filteredImg, cmap=plt.cm.gray)
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05, hspace=0.3, wspace=0)
        canvas = FigureCanvasTkAgg(fig, loadWind)
        canvas.draw()
        fig.canvas.draw()

        def abrirNueva():
            global ax, ax2, ax3, ax4, originalImg, filteredImg
            img = askopenfilename()
            try:
                pil_image = PIL.Image.open(img)
            except:
                doNothing = True
            else:
                originalImg = np.array(pil_image)
                filteredImg, dilImg, eroImg, firstBinary = filterImage(img)
                ax.imshow(eroImg, cmap=plt.cm.gray)
                ax2.imshow(dilImg, cmap=plt.cm.gray)
                ax3.imshow(firstBinary, cmap=plt.cm.gray)
                ax4.imshow(filteredImg, cmap=plt.cm.gray)
                fig.canvas.draw()

        botonesFrame = tk.Frame(loadWind)
        tk.Label(botonesFrame, width=20).pack(side=tk.LEFT)
        b_Otra = tk.Button(botonesFrame, text="CARGAR NUEVA\nIMAGEN", font='Helvetica 12', height=2, width=17,
                           command=abrirNueva)
        b_Otra.configure(relief='groove', bg='black', fg='white', activebackground='dimgray', activeforeground='white')
        b_Otra.pack(side=tk.LEFT)
        b_Reconocer = tk.Button(botonesFrame, text="RECONOCER\nIMAGEN", font='Helvetica 12', height=2, width=17,
                                command=lambda: recognizeImage(filteredImg))
        b_Reconocer.configure(relief='groove', bg='black', fg='white', activebackground='dimgray',
                              activeforeground='white')
        tk.Label(botonesFrame, width=20).pack(side=tk.RIGHT)
        b_Reconocer.pack(side=tk.RIGHT)
        fig.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
        botonesFrame.pack(side=tk.BOTTOM, fill=tk.BOTH, pady=10)

        def on_closing():
            plt.close(fig)
            loadWind.destroy()

        loadWind.protocol("WM_DELETE_WINDOW", on_closing)  # Para definir qué pasa al cerrar la ventana
        loadWind.mainloop()

def accessWebCam(webCamIntegrada, ipAdd):
    global webRead, frame
    mainWind.destroy()
    webcamWind = tk.Tk()
    webcamWind.title('Detector')
    # Se apoya de pyplot (plt) para mostrar la imagen filtrada al tomar la foto
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.set_title('Erosión')
    ax2 = fig.add_subplot(222)
    ax2.set_title('Dilatación')
    ax3 = fig.add_subplot(223)
    ax3.set_title('Imagen binaria')
    ax4 = fig.add_subplot(224)
    ax4.set_title('Ajuste por centro de masa')
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05, hspace=0.3, wspace=0)
    canvas = FigureCanvasTkAgg(fig, webcamWind)
    canvas.draw()

    fig.canvas.draw()

    def reconocerImg(filename):
        # Antes de hacer el reconocimiento, filtra la imagen muestra ese resultado
        newThresh, dilImg, eroImg, firstBinary = filterImage(filename)
        ax.imshow(eroImg, cmap=plt.cm.gray)
        ax2.imshow(dilImg, cmap=plt.cm.gray)
        ax3.imshow(firstBinary, cmap=plt.cm.gray)
        ax4.imshow(newThresh, cmap=plt.cm.gray)
        fig.canvas.draw()
        recognizeImage(newThresh)

    def capturarImg():
        global webRead, frame
        if webRead == True:
            # Guarda la imagen para trabajar con ella desde reconocerImg
            cv2.imwrite("cap.jpg", frame)
            reconocerImg("cap.jpg")
        else:
            messagebox.showerror("Error", "No se pudo capturar la imagen")

    # Se conecta con "cámara web"
    # "miDireccion" va a cambiar dependiendo de la ipWebcam que se utilice (por eso se solicita dependiendo del dispositivo)
    # Si se desea utilizar la webcam integrada al equipo, en lugar de "miDireccion" como argumento se utiliza un 0
    if(webCamIntegrada):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    else:
        miDireccion = 'http://'+str(ipAdd)+'/video'
        cam = cv2.VideoCapture(miDireccion)
    # La imagen es capturada al precionar la tecla de espacio
    webcamWind.bind("<space>", lambda e: capturarImg())
    # Para ver la imagen de la cámara se trata como una "etiqueta"
    webcm = tk.Label(webcamWind)
    # Se incluye en la ventana tanto la cámara como el "plot" que mostrará la imagen filtrada
    webcm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    fig.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def show_frame():
        global webRead, frame, thisAfter
        webRead, frame = cam.read()
        # Se gira la imagen porque la cámara la ve diferente
        # Estos giros son diferentes al trabajar con la cámara integrada al equipo
        if not (webCamIntegrada):
            frame = cv2.flip(frame, 1)
            frame = cv2.flip(frame, 2)
            frame = ndimage.rotate(frame, 270)
        # Pasar a formato que permita ver la imagen normal
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        webcm.imgtk = imgtk
        webcm.configure(image=imgtk)
        thisAfter = webcm.after(5, show_frame)

    show_frame()

    def on_closing():
        # Si se cierra la ventana es importante cancelar cualquier "after" que "show_frame()" haya iniciado
        global thisAfter
        webcamWind.after_cancel(thisAfter)
        plt.close(fig)
        try:
            webcamWind.destroy()
        except:
            webcamWind.destroy()

    webcamWind.protocol("WM_DELETE_WINDOW", on_closing)  # Para definir qué pasa al cerrar la ventana
    webcamWind.mainloop()

def opcionesDeCaptura():
    def pedirIpAdd ():
        # Para usar IP Webcam es necesario utilizar la dirección que muestra la aplicación en el dispositivo
        global ipAdd1, ipAdd2, ipAdd3, ipAdd4, puerto
        def continuar ():
            # Antes de continuar hay que validar que la dirección es válida
            global ipAdd1, ipAdd2, ipAdd3, ipAdd4, puerto
            try:
                # Verificar que los datos ingresados hayan sido numéricos
                ip1 = int(ipAdd1.get())
                ip2 = int(ipAdd2.get())
                ip3 = int(ipAdd3.get())
                ip4 = int(ipAdd4.get())
                port = int(puerto.get())
                checkConvertion = ip1+ip2+ip3+ip4+port
            except ValueError:
                tk.messagebox.showerror('Error', 'Asegúrate de ingresar una dirección válida')
            else:
                # Verificar que los números concuerden con valores de IP (entre 0 y 255)
                if (ip1>255 or ip1<0 or ip2>255 or ip2<0 or ip3 >255 or ip3<0 or ip4>255 or ip4<0):
                    tk.messagebox.showerror('Error','Asegúrate de ingresar una dirección válida')
                else:
                    ipAddress = str(ip1)+'.'+str(ip2)+'.'+str(ip3)+'.'+str(ip4)+':'+str(port)
                    try:
                        # También hay que verificar que se puede establecer conexión
                        miDireccion = 'http://' + str(ipAddress)+'/video'
                        cam = cv2.VideoCapture(miDireccion)
                    except:
                        tk.messagebox.showerror('Error', 'No se puede establecer conexión')
                    else:
                        # Si es válida y estableció bien conexión se llama a la ventana en que se muestra
                        v_pedirIP.destroy()
                        accessWebCam(False,ipAddress)

        v_opCapt.destroy()
        v_pedirIP = tk.Tk()
        v_pedirIP.geometry('290x145')
        v_pedirIP.resizable(width=False, height=False)
        v_pedirIP.title('IP')
        tk.Label(v_pedirIP, text='Indica la dirección IP y número de\npuerto que aparece en tu IP Webcam',
                 font='Helvetica 12').place(x=10, y=10)
        ipAdd1 = ttk.Entry(v_pedirIP, width=4, font='Helvetica 12', justify=tk.CENTER)
        ipAdd2 = ttk.Entry(v_pedirIP, width=4, font='Helvetica 12', justify=tk.CENTER)
        ipAdd3 = ttk.Entry(v_pedirIP, width=4, font='Helvetica 12', justify=tk.CENTER)
        ipAdd4 = ttk.Entry(v_pedirIP, width=4, font='Helvetica 12', justify=tk.CENTER)
        puerto = ttk.Entry(v_pedirIP, width=5, font='Helvetica 12', justify=tk.CENTER)
        ipAdd1.place(x=10, y=55)
        tk.Label(v_pedirIP,text='.',font='Arial 14').place(x=53,y=53)
        ipAdd2.place(x=65, y=55)
        tk.Label(v_pedirIP, text='.', font='Arial 14').place(x=108, y=53)
        ipAdd3.place(x=120, y=55)
        tk.Label(v_pedirIP, text='.', font='Arial 14').place(x=163, y=53)
        ipAdd4.place(x=175, y=55)
        tk.Label(v_pedirIP, text=':', font='Arial 14').place(x=218, y=53)
        puerto.place(x=230, y=55)
        tk.Label(v_pedirIP, text='Ejemplo: 192.168.1.208:8080', font='Helvetica 12 italic',
                 foreground='dimgray').place(x=20, y=80)
        b_continuar=tk.Button(v_pedirIP, text='Continuar', bg='greenyellow', fg='black',
                      activebackground='yellowgreen', activeforeground='black', command=continuar)
        b_continuar.place(x=120,y=112)

    def webCamDirecta ():
        try:
            # Verifica si se puede establecer conexión con la webcam
            cam = cv2.VideoCapture(0)
        except:
            tk.messagebox.showerror('Error', 'No se puede establecer conexión')
        else:
            v_opCapt.destroy()
            accessWebCam(True, '')
    # Se da la opción de capturar la imagen desde la webcam integrada o desde una IP Webcam
    v_opCapt = tk.Tk()
    v_opCapt.geometry('240x150')
    v_opCapt.resizable(width=False, height=False)
    v_opCapt.title('Captura')
    tk.Label(v_opCapt, text='Indica el dispositivo de captura:', font='Helvetica 12').place(x=10, y=10)
    b_integrada = tk.Button(v_opCapt, text='Webcam integrada', bg='greenyellow', fg='black',
                      activebackground='yellowgreen', activeforeground='black', command=webCamDirecta)
    b_integrada.place(x=45, y=40, width=150, height=40)

    b_ipWEB = tk.Button(v_opCapt, text='IP Webcam', bg='greenyellow', fg='black',
                      activebackground='yellowgreen', activeforeground='black',  command=pedirIpAdd)
    b_ipWEB.place(x=45, y=90, width=150, height=40)

def recognizeImage(thresh):
    # Para esto ya se debe haber entrenado a la red y guardado los pesos, ahora solo se leen:
    def leerPesos(archivo):
        linesInFile = archivo.readlines()
        weights = []
        for line in linesInFile:
            line = list(map(float, line.split()))
            weights.append(line)
        return (weights)

    # Convertir información de thresh a 0's y 1's como enteros
    numericList = []
    for i in range(len(thresh)):
        numeric = list(map(int, list(thresh[i])))
        numericList.extend(numeric)

    for i in range(len(numericList)):
        if (numericList[i] == 255):
            numericList[i] = 1

    # Especificar neuronas de entrada, salida y escondidas
    n_in = 1024  # Entran 1024 (32x32)
    n_hid = 20  # Se utilizaron 20 en la capa escondida
    n_out = 10  # Hay 10 salidas (cada una representa a un número)

    # Recuperar la información de los pesos
    try:
        w_hidInput = leerPesos(open('weights to hidden layer.txt', 'r'))
        w_outInput = leerPesos(open('weights to output layer.txt', 'r'))
    except:
        tk.messagebox.showerror('Error', 'Asegúrese de que el archivo de pesos esté disponible y vuelva a intentar')
    else:
        # Se crea el espacio para los cálculos que requiere la red
        f_hid, o_hid = [0.0 for i in range(n_hid)], [0.0 for i in range(n_hid)]
        f_out, o_out = [0.0 for i in range(n_out)], [0.0 for i in range(n_out)]

        # Se aplica solo el proceso "forward" pues solo va a reconocer
        for i in range(len(f_hid)):
            f_hid[i] += (-1 * w_hidInput[0][i])
            j = 0
            while (j < n_in):
                f_hid[i] += numericList[j] * w_hidInput[j + 1][i]
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

        # Mostrará la solución en un mensaje emergente
        tk.messagebox.showinfo('Reconocido', 'El numero es: ' + str(o_out.index(max(o_out))))

# Crea la ventana inicial del programa
mainWind = tk.Tk()
mainWind.title('Inicio')
mainWind.geometry('190x175')
mainWind.resizable(width=False, height=False)
mainWind.title('IM')
tk.Label(mainWind, text='Abrir imagen\npara reconocimiento', font='Helvetica 12').place(x=20, y=10)
b_LoadPic = tk.Button(mainWind, text='Cargar Imagen', bg='DodgerBlue2', fg='black',
                  activebackground='DodgerBlue3', activeforeground='black', command=lambda: cargarImagen(mainWind))
b_LoadPic.place(x=20, y=60, width=150, height=40)
b_TakePic = tk.Button(mainWind, text='Capturar Imagen', bg='DodgerBlue2', fg='black',
                  activebackground='DodgerBlue3', activeforeground='black',  command=opcionesDeCaptura)
b_TakePic.place(x=20, y=110, width=150, height=40)

mainWind.mainloop()