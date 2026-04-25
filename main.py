import sounddevice as sd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
TAPS = 64

print("Загрузка спектральной нейросети...")
try:
    nn_model = joblib.load('anc_mu_model.pkl')
    print("Нейросеть успешно подключена!")
except FileNotFoundError:
    print("ОШИБКА: Сначала запустите train_nn.py")
    exit(1)

anc_active = False
engine_active = True
road_active = False
current_mu = 0.01

weights = np.zeros(TAPS)
x_buffer = np.zeros(TAPS)
t_global = 0

def extract_fft_features(noise_chunk):
    """Точно такая же функция, как при обучении"""
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

def generate_engine_noise(frames, current_time):
    t = (current_time + np.arange(frames)) / SAMPLE_RATE
    noise = (np.sin(2 * np.pi * 100 * t) + 
             0.5 * np.sin(2 * np.pi * 200 * t) + 
             0.2 * np.sin(2 * np.pi * 50 * t)) * 0.1 
    return noise

def generate_road_noise(frames):
    return np.random.normal(0, 0.03, frames)

def audio_callback(indata, outdata, frames, time_info, status):
    global t_global, weights, x_buffer, current_mu
    global anc_active, engine_active, road_active
    
    if status:
        print(f"Сбой потока: {status}")

    voice = indata[:, 0] 
    
    engine_part = generate_engine_noise(frames, t_global) if engine_active else np.zeros(frames)
    road_part = generate_road_noise(frames) if road_active else np.zeros(frames)
    
    total_noise = engine_part + road_part
    t_global += frames

    # =======================================================
    # АНАЛИЗ СЕЙЧАС ИДЕТ ПО СПЕКТРУ (FFT)
    # =======================================================
    if anc_active and (engine_active or road_active):
        # Извлекаем 5 спектральных признаков
        features = extract_fft_features(total_noise)
        # Скармливаем нейросети
        predicted_mu = nn_model.predict([features])[0]
        current_mu = np.clip(predicted_mu, 0.001, 0.1)
    # =======================================================

    out_audio = np.zeros(frames)

    for i in range(frames):
        x_val = total_noise[i]
        v_val = voice[i]

        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = x_val

        if anc_active:
            y = np.dot(weights, x_buffer)
        else:
            y = 0.0
            weights.fill(0.0)

        e = x_val + v_val - y 

        if anc_active:
            norm = np.dot(x_buffer, x_buffer) + 1e-6
            weights += current_mu * e * x_buffer / norm

        out_audio[i] = e

    outdata[:, 0] = out_audio
    outdata[:, 1] = out_audio

# --- ИНТЕРФЕЙС УПРАВЛЕНИЯ ---
print("=" * 50)
print("🎛️ ПУЛЬТ УПРАВЛЕНИЯ ANC (Spectral-NN NLMS)")
print("=" * 50)
print(" [1] - Вкл/Выкл ANC")
print(" [2] - Вкл/Выкл Мотор")
print(" [3] - Вкл/Выкл Дорогу")
print(" [q] - Выход")
print("=" * 50)

try:
    with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=(1, 2), callback=audio_callback):
        while True:
            status_anc = "🟢 ВКЛ" if anc_active else "🔴 ВЫКЛ"
            status_eng = "🔊 ВКЛ" if engine_active else "🔇 ВЫКЛ"
            status_road = "🔊 ВКЛ" if road_active else "🔇 ВЫКЛ"
            
            mu_display = f"{current_mu:.5f}" if anc_active else "---"
            
            print(f"\nСтатус: ANC[{status_anc}] | Мотор[{status_eng}] | Дорога[{status_road}] | NN μ: {mu_display}")
            cmd = input("Ваш выбор (1, 2, 3 или q): ")
            
            if cmd.lower() == 'q':
                break
            elif cmd == '1':
                anc_active = not anc_active
            elif cmd == '2':
                engine_active = not engine_active
            elif cmd == '3':
                road_active = not road_active

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
