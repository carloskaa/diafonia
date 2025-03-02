# Importa las bibliotecas necesarias
import streamlit as st  # Framework para crear aplicaciones web
import numpy as np      # Para operaciones numéricas y manejo de arrays
import matplotlib.pyplot as plt  # Para creación de gráficos
from utils import *     # Importa funciones personalizadas del archivo utils.py

# Configura el título principal de la aplicación
st.title("Análisis de Acoplamiento y Diafonía")

# Sección de parámetros geométricos en la barra lateral
st.sidebar.header("Parámetros Geométricos")
# Inputs numéricos para los parámetros de altura y separación
h_G = st.sidebar.number_input("Altura conductor generador (m)", value=12)
h_R = st.sidebar.number_input("Altura conductor victima (m)", value=10)
x = st.sidebar.number_input("Separación X (m)", value=5)
# Inputs para radios de conductores con formato científico
r_R = st.sidebar.number_input("Radio conductor generador (m)", value=11.28e-3, format="%e")
r_G = st.sidebar.number_input("Radio conductor victima (m)", value=11.28e-3, format="%e")
LongL = st.sidebar.number_input("Longitud de los conductores (m)", value=20000)
y = h_G - h_R  # Calcula diferencia de alturas

# Botón para mostrar el diagrama geométrico
if st.sidebar.button("Ver Diagrama"):
    # Configuración del gráfico
    st.subheader("Distribución Geométrica de Conductores")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-2, x + 2)
    ax.set_ylim(0, h_G + 2)
    ax.set_xlabel("Distancia (m)")
    ax.set_ylabel("Altura (m)")
    ax.set_title("Distribución Geométrica de Conductores")

    # Dibuja los conductores y el suelo con escalado visual
    escala_radio = 20  # Factor para visualización de radios
    ax.add_patch(plt.Circle((0, h_G), r_G * escala_radio, color='b', alpha=0.5, label="Generador"))
    ax.add_patch(plt.Circle((x, h_R), r_R * escala_radio, color='r', alpha=0.5, label="Víctima"))
    ax.add_patch(plt.Rectangle((-5, 0), x + 10, 0.5, color='k', alpha=0.3, label="Suelo"))
    
    # Etiquetas de texto para los conductores
    ax.text(0, h_G + 1, "Generador", ha='center', fontsize=10, fontweight='bold')
    ax.text(x, h_R + 1, "Víctima", ha='center', fontsize=10, fontweight='bold')
    st.pyplot(fig)  # Muestra el gráfico en Streamlit

s = np.sqrt(x ** 2 + y ** 2)  # Calcula distancia entre conductores

# Sección de parámetros del rayo en la barra lateral
st.sidebar.header("Parámetros de Simulación del Rayo")
# Inputs para parámetros temporales y de corriente
fs = st.sidebar.number_input("Frecuencia de muestreo (Hz)", value=100e6, format="%e")
tr = st.sidebar.number_input("Tiempo de frente (s)", value=1.2e-6, format="%e")
tf = st.sidebar.number_input("Tiempo de cola (s)", value=50e-6, format="%e")
I0 = st.sidebar.number_input("Corriente pico (A)", value=35000)

# Botón para visualizar el rayo
if st.sidebar.button("Visualizar Rayo en el Tiempo"):
    T = 1 / fs  # Calcula período de muestreo
    t = np.arange(0, (3 / tf) * T, T)  # Crea vector de tiempo
    i_t = ray_impulse(t, I0, tr, tf)  # Genera forma de onda del rayo (utils)

    # Configura y muestra gráfico del rayo
    st.subheader("Rayo en el Tiempo")
    fig, ax = plt.subplots()
    ax.plot(t, i_t, 'b')
    ax.set_title("Rayo en el Tiempo")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid()
    st.pyplot(fig)

# Sección de parámetros eléctricos
st.sidebar.header("Parámetros del Sistema Eléctrico")
# Inputs para valores de resistencias e inductancias
R_NE = float(st.sidebar.text_input("Resistencia NE (Ω)", value="10000000"))
R_FE = float(st.sidebar.text_input("Resistencia FE (Ω)", value="10"))
R_FE2 = float(st.sidebar.text_input("Inductancia FE (H)", value="0.0438"))

R_S = float(st.sidebar.text_input("Resistencia RS (Ω)", value="5.236"))
R_L = float(st.sidebar.text_input("Resistencia RL (Ω)", value="10"))
R_L2 = float(st.sidebar.text_input("Inductancia RL (H)", value="0.0438"))
Z_ray = float(st.sidebar.text_input("Resistencia del rayo (Ω)", value="1"))

# Convierte valores a números complejos (impedancias)
R_FE = complex(float(R_FE),float(R_FE2))
R_L = complex(float(R_L),float(R_L2))

# Botón principal de simulación
if st.sidebar.button("Simular"):
    st.subheader("Resultados de la Simulación")
    T = 1 / fs
    t = np.arange(0, (3 / tf) * T, T)
    i_t = ray_impulse(t, I0, tr, tf)  # Genera señal del rayo

    # Extensión de la señal para FFT
    t_ext = np.concatenate((-np.flip(t), t))
    x_ext = np.concatenate((np.flip(i_t), i_t))

    # Transformada de Fourier
    X_f = np.fft.fftshift(np.fft.fft(x_ext))
    f = np.fft.fftshift(np.fft.fftfreq(len(t_ext), T))

    # Cálculos de parámetros electromagnéticos
    LM = inductancia_mutua(h_G, h_R, s) * LongL  # Inductancia mutua
    lr = calcular_inductancia(h_R, r_R) * LongL   # Inductancia conductor víctima
    lg = calcular_inductancia(h_G, r_G) * LongL   # Inductancia conductor generador
    CM = calcular_capacitancia(lg, LM, lr) * LongL # Capacitancia mutua

    # Modelado de circuitos y cálculos de voltajes
    VS = X_f / Z_ray
    I_G = VS / (R_S + R_L)
    V_G = (R_L / (R_S + R_L)) * VS
    AcopleIn = (R_NE / (R_NE + R_FE)) * LM * I_G * 2 * np.pi * f
    AcopleCa = (R_NE * R_FE / (R_NE + R_FE)) * CM * V_G * 2 * np.pi * f

    # Combinación de términos para voltajes finales
    term1 = (R_NE / (R_NE + R_FE)) * LM * (1 / (R_S + R_L))
    term2 = (R_NE * R_FE / (R_NE + R_FE)) * CM * (R_L / (R_S + R_L))
    V_ne_vs = (term1 + term2) * 1j * 2 * np.pi * f * VS

    termino1 = - (R_FE / (R_NE + R_FE)) * (LM / (R_S + R_L))
    termino2 = (R_NE * R_FE / (R_NE + R_FE)) * (CM * (R_L / (R_S + R_L)))
    V_fe_vs = (termino1 + termino2) * 1j * 2 * np.pi * f * VS

    # Gráfica del espectro modulado
    st.subheader("Espectro Modulado")
    fig, ax = plt.subplots()
    ax.plot(f, np.abs(V_fe_vs), 'm')
    ax.set_title("Espectro Modulado")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("V_fe_vs")
    ax.grid()
    st.pyplot(fig)

    # Reconstrucción de señales temporales
    V_fe_vs_reconstruida_shifted = np.fft.ifft(np.fft.ifftshift(V_fe_vs))
    V_recortada = V_fe_vs_reconstruida_shifted[len(t):]

    # Gráfica de señal reconstruida V_fe_vs
    st.subheader("Señal Reconstruida V_fe_vs")
    fig, ax = plt.subplots()
    ax.plot(t_ext, V_fe_vs_reconstruida_shifted, 'g')
    ax.set_title("Señal Reconstruida V_fe_vs")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_xlim([0, 5e-6])
    ax.legend(["x_{reconstruida}"])
    ax.grid()
    st.pyplot(fig)

    # Reconstrucción y gráfica de V_ne_vs
    V_ne_vs_reconstruida_shifted = np.fft.ifft(np.fft.ifftshift(V_ne_vs))
    V_ne_vs_recortada = V_ne_vs_reconstruida_shifted[len(t):]

    st.subheader("Señal Reconstruida V_ne_vs")
    fig, ax = plt.subplots()
    ax.plot(t, V_ne_vs_recortada, 'g')
    ax.set_title("Señal Reconstruida V_ne_vs")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_xlim([0, 5e-6])
    ax.legend(["x_{reconstruida}"])
    ax.grid()
    st.pyplot(fig)