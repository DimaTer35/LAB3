import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024 

def extract_fft_features(noise_chunk):
    """Извлекает энергию по частотным полосам с помощью FFT"""
    windowed_noise = noise_chunk * np.hanning(len(noise_chunk))
    fft_spectrum = np.abs(np.fft.rfft(windowed_noise))
    fft_freqs = np.fft.rfftfreq(len(noise_chunk), 1 / SAMPLE_RATE)
    
    band_subbass = np.sum(fft_spectrum[(fft_freqs >= 0) & (fft_freqs < 80)])
    band_bass = np.sum(fft_spectrum[(fft_freqs >= 80) & (fft_freqs < 300)])
    band_mids = np.sum(fft_spectrum[(fft_freqs >= 300) & (fft_freqs < 2000)])
    band_highs = np.sum(fft_spectrum[fft_freqs >= 2000])
    
    total_energy = band_subbass + band_bass + band_mids + band_highs + 1e-6
    
    return [
        band_subbass / total_energy, 
        band_bass / total_energy, 
        band_mids / total_energy, 
        band_highs / total_energy, 
        np.var(noise_chunk)
    ]

def generate_data(noise_type, samples=BLOCK_SIZE):
    t = np.linspace(0, samples/SAMPLE_RATE, samples, endpoint=False)
    
    if noise_type == 'engine':
        noise = (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)) * 0.1
        target_mu = 0.05
    elif noise_type == 'road':
        noise = np.random.normal(0, 0.03, samples)
        target_mu = 0.005
    elif noise_type == 'wind':
        # Генерируем ветер (Фильтр высоких частот через разность отсчетов)
        white_noise = np.random.normal(0, 0.05, samples + 1)
        noise = np.diff(white_noise) * 0.5
        target_mu = 0.001 # Экстремально малый шаг для защиты от нестабильности!
    else:
        # Смешанный шум
        noise = (np.sin(2 * np.pi * 100 * t)) * 0.05 + np.random.normal(0, 0.01, samples)
        target_mu = 0.02

    features = extract_fft_features(noise)
    return features, target_mu

print("Генерация спектральных данных (с учетом шума ветра)...")
X_train, y_train = [], []

# Создаем расширенный датасет
for _ in range(500):
    for n_type in ['engine', 'road', 'wind', 'mixed']:
        features, mu = generate_data(n_type)
        X_train.append(features)
        y_train.append(mu)

print("Обучение сети на новых FFT-признаках...")
model = MLPRegressor(hidden_layer_sizes=(15, 10), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'anc_mu_model.pkl')
print("Успех! Обновленная модель сохранена в anc_mu_model.pkl.")
