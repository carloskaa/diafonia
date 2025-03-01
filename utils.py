import numpy as np

def ray_impulse(t, I0, tr, tf):
    # Cálculo de constantes a y b
    a = tr / np.log(9)
    b = tf / np.log(2)

    # Cálculo de la onda de rayo
    i_t_raw = I0 * (np.exp(-t / b) - np.exp(-t / a))
    return (I0 / np.max(i_t_raw)) * i_t_raw  # Normalización

# Cálculo de inductancias y capacitancias
def inductancia_mutua(h_G, h_R, s):
    mu_0 = 4 * np.pi * 1e-7
    return (mu_0 / (4 * np.pi)) * np.log(1 + 4 * (h_G * h_R) / (s ** 2))

def calcular_inductancia(h_G, r_wG):
    mu_0 = 4 * np.pi * 1e-7
    return (mu_0 / (2 * np.pi)) * np.log((2 * h_G) / r_wG)

def calcular_capacitancia(lg, lm, lr):
    ind = np.array([[lg, lm], [lm, lr]])
    cap = np.linalg.inv(ind) * (1 / (3e8 * 3e8))
    return -cap[0, 1]

def parse_complex(value):
    if 'j' in value:
        return complex(re.sub(r'(\d+)j([+-]?\d+)', r'\2+\1j', value))
    else:    
        return complex(value)