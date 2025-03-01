import numpy as np
import matplotlib.pyplot as plt
from utils import *
# Parámetros de la señal
fs = 100e6  # Frecuencia de muestreo (Hz)
T = 1 / fs  # Período de muestreo

tr = 1.2e-6  # Tiempo de frente
tf = 50e-6   # Tiempo de cola
I0 = 35000
t = np.arange(0, (3 / tf) * T, T)  # Paso de T hasta (3/tf) * T
i_t = ray_impulse(t, I0, tr, tf)

# Extender la señal por simetría
t_ext = np.concatenate((-np.flip(t), t))
x_ext = np.concatenate((np.flip(i_t), i_t))

# Transformada de Fourier
X_f = np.fft.fftshift(np.fft.fft(x_ext))
f = np.fft.fftshift(np.fft.fftfreq(len(t_ext), T))

# Parámetros eléctricos
R_NE = 100
R_FE = 50
R_S = 200
R_L = 300
Z_ray = 1
LongL = 20000
h_G = 12
h_R = 10
s = 5.3
r_R = 11.28e-3
r_wG = 11.28e-3

LM = inductancia_mutua(h_G, h_R, s) * LongL
lr = calcular_inductancia(h_R, r_R) * LongL
lg = calcular_inductancia(h_G, r_wG) * LongL
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

plt.figure()
plt.plot(f, np.abs(V_fe_vs), 'm')
plt.title('Espectro Modulado')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('V_fe_vs')
plt.grid()
plt.show()

# Recuperación de la señal
V_fe_vs_reconstruida_shifted = np.fft.ifft(np.fft.ifftshift(V_fe_vs))
V_recortada = V_fe_vs_reconstruida_shifted[len(t):]

plt.figure()
plt.plot(t_ext, V_fe_vs_reconstruida_shifted, 'g')
plt.title('Señal Reconstruida V_fe_vs')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.xlim([0, 5e-6])
plt.legend(['x_{reconstruida}'])
plt.grid()
plt.show()

V_ne_vs_reconstruida_shifted = np.fft.ifft(np.fft.ifftshift(V_ne_vs))
V_ne_vs_recortada = V_ne_vs_reconstruida_shifted[len(t):]

plt.figure()
plt.plot(t, V_ne_vs_recortada, 'g')
plt.title('Señal Reconstruida V_ne_vs')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.xlim([0, 5e-6])
plt.legend(['x_{reconstruida}'])
plt.grid()
plt.show()
