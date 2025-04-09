import os
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from ipywidgets import interact, widgets

"""##**DATA EXPLORATION**"""

import h5py
def explore_hdf5(file_path):
    with h5py.File(file_path, "r") as hdf:
        def print_attrs(name, obj):
            """Imprime los atributos y arreglos dentro del archivo HDF5."""
            print(f"\nObjeto: {name}")
            # Mostrar atributos si existen
            if obj.attrs:
                print("  Atributos:")
                for attr, value in obj.attrs.items():
                    print(f"    {attr}: {value}")
            # Mostrar datos si es un dataset
            #if isinstance(obj, h5py.Dataset):
            #    print("  Datos (forma):", obj.shape)
            #    print("  Tipo de datos:", obj.dtype)
            #    print("  Ejemplo de datos:", obj[()] if obj.size < 10 else obj[:5])  # Muestra hasta 5 elementos
        hdf.visititems(print_attrs)  # Recorre todos los grupos y datasets

"""# **READ HDF5 ATTRIBUTES**"""

def read_atributos(filename):
    with h5py.File(filename, "r") as obj:
        var_path = "Data/data_spc"

        # Detectar automáticamente los nombres de los canales dentro de "Data/data_spc"
        channel_names = sorted(obj[var_path].keys())  # Lista con nombres de los canales
        num_channels = len(channel_names)
        print(f"🔍 Se encontraron {num_channels} canales: {channel_names}")

        # Marca de tiempo
        utc_time = np.array(obj["Data/utctime"])

        # Leer datos de todos los canales en un solo arreglo
        data_list = [np.array(obj[f"{var_path}/{channel}"]) for channel in channel_names]
        data_arr = np.stack(data_list, axis=0)  # Crea un arreglo con los datos en (4, 20, 1000, 936)

        # Reordenar dimensiones: mover la segunda dimensión (20) a la primera posición
        data_arr = np.moveaxis(data_arr, 1, 0)  # Ahora tiene forma (20, 4, 1000, 936)

        # Leer todos los metadatos automáticamente
        metadata = {}
        for key in obj["Metadata"].keys():
            metadata[key] = np.array(obj[f"Metadata/{key}"])

        # Se extraen las variables necesarias de la metadata
        code=metadata.get("code")
        nProfiles = metadata.get("nProfiles", 1)       # Número de perfiles, puede ser 1 si no está definido
        nIncohInt = metadata.get("nIncohInt", 1)       # Intervalo de incoherencia
        nCohInt = metadata.get("nCohInt", 1)           # Intervalo de coherencia
        windowOfFilter = metadata.get("windowOfFilter", 1)  # Ventana de filtro, si está presente
        pwcode = 1  # Valor inicial de pwcode, se ajustará según el cálculo

        # Calcular pwcode basado en los datos de entrada si flagDecodeData es True
        if metadata.get("flagDecodeData", False):
            pwcode = np.sum(code[0] ** 2)  # Ejemplo de cálculo de pwcode (suma de los cuadrados de los datos)

        # Cálculo del factor de normalización
        normFactor = nProfiles * nIncohInt * nCohInt * pwcode * windowOfFilter

        return data_arr, metadata, channel_names,utc_time,normFactor

"""# **FREQUENCY-VELOCITY RANGE**"""

def freq_vel_Range(ipp, freq, nFFTPoints):
    """Calcula el rango de frecuencias y velocidades utilizando los atributos del HDF5."""
    PRF = 1 / ipp  # Frecuencia de repetición de pulso
    fmax = PRF / 2  # Máxima frecuencia
    C = 3.0e8  # Velocidad de la luz en m/s
    _lambda_ = C / (freq * 1e6)  # Longitud de onda
    vmax = fmax * _lambda_ / 2.0  # Velocidad máxima

    deltafreq = fmax / nFFTPoints
    freqrange = 2 * deltafreq * (np.arange(nFFTPoints) - nFFTPoints / 2.0)

    deltavel = vmax / nFFTPoints
    velrange = 2 * deltavel * (np.arange(nFFTPoints) - nFFTPoints / 2.0)

    return freqrange, velrange

"""# **Additional Methods**

# **HILDEBRAND SEKHON**
"""

def hildebrand_sekhon(data, navg):
    data = data.copy()
    sortdata = np.sort(data,axis=None)
    lenOfData = len(sortdata)
    nums_min = lenOfData/10
    if (lenOfData/10) > 2:
      nums_min = lenOfData/10
    else:
      nums_min = 2
    sump = 0.
    sumq = 0.
    j = 0
    cont = 1
    while((cont==1)and(j<lenOfData)):
      sump += sortdata[j]
      sumq += sortdata[j]**2
      j += 1
      if j > nums_min:
        rtest = float(j)/(j-1) + 1.0/navg
        if ((sumq*j) > (rtest*sump**2)):
          j = j - 1
          sump  = sump - sortdata[j]
          sumq =  sumq - sortdata[j]**2
          cont = 0
    lnoise = sump /j
    stdv = np.sqrt((sumq - lnoise**2)/(j - 1))
    return lnoise

def getNoisebyHildebrand(data_spc):
    #print("VAMOS AHI")
    noise    = np.zeros(data_spc.shape[0],dtype='f')
    #print("SHAPE",data_spc.shape)
    #data_spc = data_spc.reshape(4,data_spc.shape[0],1000)
    for channel in range(data_spc.shape[0]):
        daux = data_spc[channel,:,:]
        noise[channel] = hildebrand_sekhon(daux, 1)
    return noise

def integracion_incoherente(data, block_size):
    """
    Realiza la integración incoherente de los datos promediando bloques de la primera dimensión de 'data'.

    :param data: ndarray de forma (20, 4, 1000, 936)
    :param block_size: tamaño de los bloques a promediar
    :return: ndarray con los datos promediados
    """
    # Número de bloques que se pueden formar
    num_blocks = data.shape[0] // block_size

    # Iniciar un arreglo vacío para almacenar los resultados de la integración incoherente
    integrated_data = []

    for i in range(num_blocks):
        # Seleccionar el bloque de datos
        block = data[i * block_size: (i + 1) * block_size]

        # Promediar a lo largo de la primera dimensión (bloques de 5 elementos)
        integrated_block = np.nansum(block, axis=0)

        # Almacenar el bloque promediado
        integrated_data.append(integrated_block)

    # Convertir la lista de bloques promediados a un arreglo numpy
    integrated_data = np.array(integrated_data)

    return integrated_data

"""# **PLOT**"""

# Función para graficar el espectro
def spectraPlot(fdata__, metadata, channel_names, utc_time,normFactor, j=0, int_inc=None, power_min=None, power_max=None):
    # Leer atributos del archivo
    #data__, metadata, channel_names, utc_time = read_atributos(filename)
    print("Dimensión de data__:", data__.shape)  # (20, 4, 936, 128)
    print("int_inc", int_inc)
    data = integracion_incoherente(data__, int_inc)

    print("Dimensión de data:", data.shape)  # (20, 4, 936, 128)

    data_spc = data[j]  # Extraer datos para el tiempo j
    data_utc_time = utc_time[j] - 18000  # Ajuste de zona horaria

    # Convertir UTC a fecha legible
    fecha_legible = datetime.utcfromtimestamp(data_utc_time).strftime("%Y-%m-%d %H:%M:%S")

    power = data_spc.copy()/normFactor  # Copia de los datos originales
    nFFTPoints = power.shape[1]

    print("Dimensión de power:", power.shape)  # (4, 936, 128)

    # Configurar el rango de frecuencias y velocidades
    ippSeconds = metadata.get("ippSeconds", 1)
    frequency = metadata.get("freq", 49.92)  # Ajustar
    nFFTPoints = metadata.get("nFFTPoints", 256)
    height_list = metadata.get("heightList", 1)

    freq_range, vel_range = freq_vel_Range(ippSeconds, frequency, nFFTPoints)
    RangeMin, RangeMax = height_list[0], height_list[-1]
    MinVel, MaxVel = vel_range[0], vel_range[-1]

    # Crear subgráficos para los 4 canales
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Channell {i+1}" for i in range(4)])

    power_db_all_channels = []

    for channel in range(4):  # Iterar sobre los 4 canales
        power_db = 10 * np.log10(power[channel].T)  # Convertir a dB
        power_db = np.nan_to_num(power_db, nan=np.nanmin(power_db), posinf=np.nanmax(power_db), neginf=np.nanmin(power_db))

        power_db_all_channels.append(power_db) # Agregar la potencia de este canal a la lista para encontrar el rango común

        # Agregar heatmap a la figura
        fig.add_trace(go.Heatmap(
            z=power_db.astype(float),
            x=vel_range,
            y=height_list,
            colorscale="Jet",
            colorbar=dict(title="Potencia (dB)"),
            showscale=True  # Habilitar barra de color para el gráfico
        ), row=(channel // 2) + 1, col=(channel % 2) + 1)

    # Determinar el rango de potencia común
    all_power_db = np.concatenate([p.flatten() for p in power_db_all_channels])
    if power_min is None:
        power_min = np.min(all_power_db)
    if power_max is None:
        power_max = np.max(all_power_db)

    # Actualizar el layout de la figura
    fig.update_layout(
        title=f"Espectrogramas | Fecha: {fecha_legible}",
        height=800, width=900,
        coloraxis=dict(colorscale="Jet", colorbar=dict(title="Potencia (dB)", tickvals=[power_min, power_max])),
        xaxis_title="Velocidad (m/s)",
        yaxis_title="Altura (m)",
        showlegend=True  # Mostrar leyenda global para todos los subgráficos
    )

    # Asegurarse de que todos los subgráficos compartan el eje X
    fig.update_xaxes(title="Velocidad (m/s)", row=1, col=1, showticklabels=True)
    fig.update_xaxes(title="Velocidad (m/s)", row=1, col=2, showticklabels=True)
    fig.update_xaxes(title="Velocidad (m/s)", row=2, col=1, showticklabels=True)
    fig.update_xaxes(title="Velocidad (m/s)", row=2, col=2, showticklabels=True)

    # Actualizar los valores mínimos y máximos de potencia
    fig.update_traces(zmin=power_min, zmax=power_max)

    # Mostrar gráfico
    fig.show()


"""# **CODE- SPECTRUM VIEWER**"""

from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


"""# **SPECTRUM VIEWER**"""

filename = "/home/soporte/Downloads/D2025086000.hdf5"       # Path to the input HDF5 file containing radar data

#filename = ''
data__, metadata, channel_names, utc_time,normFactor = read_atributos(filename)

spectraPlot( # The index of the data slice to plot. Default is 0.
            data__,
            metadata,
            channel_names,
            utc_time,
            normFactor,      # A normalization factor calculated from metadata, used to scale the data.
            j=15,
            int_inc=1,       # Interval increment for incoherent integration. Default is None.
            power_min=17.5,  # Minimum value for the power scale. Default is None.
            power_max= 25)   # Maximum value for the power scale. Default is None.