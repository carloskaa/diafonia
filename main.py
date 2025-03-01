import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import re
# Título de la aplicación
st.title("Análisis de Acoplamiento y Diafonía")

# Parametros geometricos de la simulación
st.sidebar.header("Parámetros Geométricos")
h_G = st.sidebar.number_input("Altura conductor generador (m)", value=12)
h_R = st.sidebar.number_input("Altura conductor victima (m)", value=10)
x = st.sidebar.number_input("Separación X (m)", value=5)
r_R = st.sidebar.number_input("Radio conductor generador (m)", value=11.28e-3, format="%e")
r_G = st.sidebar.number_input("Radio conductor victima (m)", value=11.28e-3, format="%e")
LongL = st.sidebar.number_input("Longitud de los conductores (m)", value=20000)
y = h_G - h_R
if st.sidebar.button("Ver Diagrama"):
    # Crear la figura
    st.subheader("Distribución Geométrica de Conductores")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-2, x + 2)
    ax.set_ylim(0, h_G + 2)
    ax.set_xlabel("Distancia (m)")
    ax.set_ylabel("Altura (m)")
    ax.set_title("Distribución Geométrica de Conductores")

    # Dibujar conductores con un tamaño más grande para visibilidad
    escala_radio = 20  # Factor de escala para aumentar el tamaño visual de los conductores
    ax.add_patch(plt.Circle((0, h_G), r_G * escala_radio, color='b', alpha=0.5, label="Generador"))
    ax.add_patch(plt.Circle((x, h_R), r_R * escala_radio, color='r', alpha=0.5, label="Víctima"))
    ax.add_patch(plt.Rectangle((-5, 0), x + 10, 0.5, color='k', alpha=0.3, label="Suelo"))
    # Agregar etiquetas de nombres sobre los conductores
    ax.text(0, h_G + 1, "Generador", ha='center', fontsize=10, fontweight='bold')
    ax.text(x, h_R + 1, "Víctima", ha='center', fontsize=10, fontweight='bold')
    # Mostrar gráfica en Streamlit
    st.pyplot(fig)
s = np.sqrt(x ** 2 + y ** 2)

# Parámetros de simulación del rayo
st.sidebar.header("Parámetros de Simulación del Rayo")
fs = st.sidebar.number_input("Frecuencia de muestreo (Hz)", value=100e6, format="%e")
tr = st.sidebar.number_input("Tiempo de frente (s)", value=1.2e-6, format="%e")
tf = st.sidebar.number_input("Tiempo de cola (s)", value=50e-6, format="%e")
I0 = st.sidebar.number_input("Corriente pico (A)", value=35000)

if st.sidebar.button("Visualizar Rayo en el Tiempo"):
    T = 1 / fs  # Período de muestreo
    t = np.arange(0, (3 / tf) * T, T)
    i_t = ray_impulse(t, I0, tr, tf)

    st.subheader("Rayo en el Tiempo")
    fig, ax = plt.subplots()
    ax.plot(t, i_t, 'b')
    ax.set_title("Rayo en el Tiempo")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid()
    st.pyplot(fig)

# Parámetros eléctricos
st.sidebar.header("Parámetros del Sistema Eléctrico")

R_NE = parse_complex(st.sidebar.text_input("Impedancia NE (Ω)", value="100"))
R_FE = parse_complex(st.sidebar.text_input("Impedancia FE (Ω)", value="50"))
R_S = parse_complex(st.sidebar.text_input("Impedancia RS (Ω)", value="200"))
R_L = parse_complex(st.sidebar.text_input("Impedancia RL (Ω)", value="300"))
Z_ray = parse_complex(st.sidebar.text_input("Impedancia del rayo (Ω)", value="1"))


# Botón para ejecutar la simulación
if st.sidebar.button("Simular"):
    st.subheader("Resultados de la Simulación")
    T = 1 / fs  # Período de muestreo
    t = np.arange(0, (3 / tf) * T, T)
    i_t = ray_impulse(t, I0, tr, tf)

    # Extender la señal por simetría
    t_ext = np.concatenate((-np.flip(t), t))
    x_ext = np.concatenate((np.flip(i_t), i_t))

    # Transformada de Fourier
    X_f = np.fft.fftshift(np.fft.fft(x_ext))
    f = np.fft.fftshift(np.fft.fftfreq(len(t_ext), T))

    LM = inductancia_mutua(h_G, h_R, s) * LongL
    lr = calcular_inductancia(h_R, r_R) * LongL
    lg = calcular_inductancia(h_G, r_G) * LongL
    CM = calcular_capacitancia(lg, LM, lr) * LongL

    VS = X_f / Z_ray
    I_G = VS / (R_S + R_L)
    V_G = (R_L / (R_S + R_L)) * VS

    AcopleIn = (R_NE / (R_NE + R_FE)) * LM * I_G * 2 * np.pi * f
    AcopleCa = (R_NE * R_FE / (R_NE + R_FE)) * CM * V_G * 2 * np.pi * f

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

    # Recuperación de la señal
    V_fe_vs_reconstruida_shifted = np.fft.ifft(np.fft.ifftshift(V_fe_vs))
    V_recortada = V_fe_vs_reconstruida_shifted[len(t):]

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
