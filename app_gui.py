import customtkinter as ctk
import sounddevice as sd
import numpy as np
import joblib
import warnings
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings("ignore")

# ==========================================
# НАСТРОЙКИ СТИЛЯ ИНТЕРФЕЙСА (СВЕТЛАЯ ТЕМА)
# ==========================================
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue") 

COLOR_BG = "#f3f4f6"        
COLOR_PANEL = "#ffffff"     
COLOR_TEXT_MAIN = "#2c3e50" 
COLOR_TEXT_SUB = "#7f8c8d"  
COLOR_ON = "#2ecc71"        
COLOR_OFF = "#e74c3c"       
COLOR_SRC_ON = "#3498db"    
COLOR_SRC_OFF = "#bdc3c7"   
COLOR_NN = "#9b59b6"        

# ==========================================
# 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ И ПАМЯТЬ
# ==========================================
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
TAPS = 64

state = {
    'anc_active': False,
    'engine_active': True,
    'road_active': False,
    'wind_active': False,  # Добавили состояние ветра
    'nn_active': True,    
    'current_mu': 0.01,
    'fixed_mu': 0.01      
}

weights = np.zeros(TAPS)
x_buffer = np.zeros(TAPS)
t_global = 0
gui_plot_buffer = np.zeros(BLOCK_SIZE)

# ==========================================
# 2. МАТЕМАТИЧЕСКОЕ И АУДИО ЯДРО
# ==========================================
try:
    nn_model = joblib.load('anc_mu_model.pkl')
except FileNotFoundError:
    print("ОШИБКА: Файл anc_mu_model.pkl не найден! Запустите train_nn.py")
    exit(1)

def extract_fft_features(noise_chunk):
    windowed_noise = noise_chunk * np.hanning(len(noise_chunk))
    fft_spectrum = np.abs(np.fft.rfft(windowed_noise))
    fft_freqs = np.fft.rfftfreq(len(noise_chunk), 1 / SAMPLE_RATE)
    
    band_subbass = np.sum(fft_spectrum[(fft_freqs >= 0) & (fft_freqs < 80)])
    band_bass = np.sum(fft_spectrum[(fft_freqs >= 80) & (fft_freqs < 300)])
    band_mids = np.sum(fft_spectrum[(fft_freqs >= 300) & (fft_freqs < 2000)])
    band_highs = np.sum(fft_spectrum[fft_freqs >= 2000])
    
    total_energy = band_subbass + band_bass + band_mids + band_highs + 1e-6
    return [band_subbass/total_energy, band_bass/total_energy, 
            band_mids/total_energy, band_highs/total_energy, np.var(noise_chunk)]

def generate_engine_noise(frames, current_time):
    t = (current_time + np.arange(frames)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)) * 0.1 

def generate_road_noise(frames):
    return np.random.normal(0, 0.03, frames)

def generate_wind_noise(frames):
    """Генератор аэродинамического шума (высокие частоты)"""
    white_noise = np.random.normal(0, 0.05, frames + 1)
    wind = np.diff(white_noise) * 0.5
    return wind

def audio_callback(indata, outdata, frames, time_info, status):
    global t_global, weights, x_buffer, gui_plot_buffer
    
    voice = indata[:, 0] 
    
    # Собираем все шумы
    engine_part = generate_engine_noise(frames, t_global) if state['engine_active'] else np.zeros(frames)
    road_part = generate_road_noise(frames) if state['road_active'] else np.zeros(frames)
    wind_part = generate_wind_noise(frames) if state['wind_active'] else np.zeros(frames)
    
    total_noise = engine_part + road_part + wind_part
    t_global += frames

    if state['anc_active'] and (state['engine_active'] or state['road_active'] or state['wind_active']):
        if state['nn_active']:
            features = extract_fft_features(total_noise)
            predicted_mu = nn_model.predict([features])[0]
            state['current_mu'] = np.clip(predicted_mu, 0.0005, 0.1) # Снизили нижний лимит для ветра
        else:
            state['current_mu'] = state['fixed_mu']

    out_audio = np.zeros(frames)

    for i in range(frames):
        x_val = total_noise[i]
        v_val = voice[i]
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = x_val

        if state['anc_active']:
            y = np.dot(weights, x_buffer)
        else:
            y = 0.0
            weights.fill(0.0)

        e = x_val + v_val - y 

        if state['anc_active']:
            norm = np.dot(x_buffer, x_buffer) + 1e-6
            weights += state['current_mu'] * e * x_buffer / norm

        out_audio[i] = e

    outdata[:, 0] = out_audio
    outdata[:, 1] = out_audio
    np.copyto(gui_plot_buffer, out_audio)

# ==========================================
# 3. ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС (GUI)
# ==========================================
class ANCApplication(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart ANC System Pro (Light Edition)")
        self.geometry("1200x850") # Слегка расширили окно
        self.configure(fg_color=COLOR_BG)
        
        self.setup_ui()
        
        self.stream = sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=(1, 2), callback=audio_callback)
        self.stream.start()
        self.update_plot()

    def setup_ui(self):
        top_frame = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=15, border_width=1, border_color="#e5e7eb")
        top_frame.pack(side="top", fill="x", padx=20, pady=20)

        font_btn = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        font_title = ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
        font_mu = ctk.CTkFont(family="Consolas", size=16, weight="bold")

        # --- ИСТОЧНИКИ ШУМА ---
        src_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        src_frame.pack(side="left", padx=20, pady=15)
        
        ctk.CTkLabel(src_frame, text="ИСТОЧНИКИ ШУМА", font=font_title, text_color=COLOR_TEXT_SUB).pack(pady=(0, 10))
        
        self.btn_eng = ctk.CTkButton(src_frame, text="Мотор (2)", font=font_btn, width=100, height=40,
                                     text_color="white", fg_color=COLOR_SRC_ON, hover_color="#2980b9", command=self.toggle_eng)
        self.btn_eng.pack(side="left", padx=5)

        self.btn_road = ctk.CTkButton(src_frame, text="Дорога (3)", font=font_btn, width=100, height=40,
                                      text_color="#555555", fg_color=COLOR_SRC_OFF, hover_color="#95a5a6", command=self.toggle_road)
        self.btn_road.pack(side="left", padx=5)

        self.btn_wind = ctk.CTkButton(src_frame, text="Ветер (5)", font=font_btn, width=100, height=40,
                                      text_color="#555555", fg_color=COLOR_SRC_OFF, hover_color="#95a5a6", command=self.toggle_wind)
        self.btn_wind.pack(side="left", padx=5)

        ctk.CTkFrame(top_frame, width=2, height=60, fg_color="#e5e7eb").pack(side="left", padx=10)

        # --- LMS КОНТРОЛЛЕР ---
        algo_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        algo_frame.pack(side="left", padx=20, pady=15)
        
        ctk.CTkLabel(algo_frame, text="LMS КОНТРОЛЛЕР", font=font_title, text_color=COLOR_TEXT_SUB).pack(pady=(0, 10))

        self.btn_anc = ctk.CTkButton(algo_frame, text="ANC ВЫКЛ (1)", font=font_btn, width=120, height=40,
                                     text_color="white", fg_color=COLOR_OFF, hover_color="#c0392b", command=self.toggle_anc)
        self.btn_anc.pack(side="left", padx=5)

        self.btn_nn = ctk.CTkButton(algo_frame, text="Нейросеть ВКЛ (4)", font=font_btn, width=140, height=40,
                                    text_color="white", fg_color=COLOR_NN, hover_color="#8e44ad", command=self.toggle_nn)
        self.btn_nn.pack(side="left", padx=5)

        # --- Индикатор и Выход ---
        right_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        right_frame.pack(side="right", padx=20, pady=15)

        self.lbl_mu = ctk.CTkLabel(right_frame, text="Управление μ: ---", font=font_mu, text_color=COLOR_TEXT_SUB)
        self.lbl_mu.pack(side="top", pady=(0, 10), anchor="e")

        ctk.CTkButton(right_frame, text="ВЫХОД", font=font_btn, width=100, height=30,
                      text_color="white", fg_color="#95a5a6", hover_color="#7f8c8d", command=self.on_closing).pack(side="top", anchor="e")

        # --- НАСТРОЙКА ГРАФИКОВ ---
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor(COLOR_BG) 

        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_facecolor(COLOR_PANEL)
        self.ax1.set_title("Остаточный шумовой сигнал e(n)", color=COLOR_TEXT_MAIN, fontsize=12, fontweight='bold', pad=10)
        self.ax1.set_ylabel("Амплитуда", color=COLOR_TEXT_SUB)
        self.ax1.tick_params(colors=COLOR_TEXT_SUB)
        self.ax1.set_ylim(-0.5, 0.5) # Расширил амплитуду, так как ветер громкий
        self.ax1.grid(True, color='#ecf0f1', linestyle='-', linewidth=1.5, alpha=0.7)
        for spine in self.ax1.spines.values(): spine.set_color('#bdc3c7')
        self.line1, = self.ax1.plot([], [], lw=2, color=COLOR_OFF) 

        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_facecolor(COLOR_PANEL)
        self.ax2.set_title("Динамика шага адаптации μ", color=COLOR_TEXT_MAIN, fontsize=12, fontweight='bold', pad=10)
        self.ax2.set_xlabel("Время (последние кадры)", color=COLOR_TEXT_SUB)
        self.ax2.set_ylabel("Значение μ", color=COLOR_TEXT_SUB)
        self.ax2.tick_params(colors=COLOR_TEXT_SUB)
        self.ax2.set_ylim(0, 0.06)
        self.ax2.grid(True, color='#ecf0f1', linestyle='-', linewidth=1.5, alpha=0.7)
        for spine in self.ax2.spines.values(): spine.set_color('#bdc3c7')
        self.line2, = self.ax2.plot([], [], lw=2.5, color=COLOR_NN)

        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True, padx=20, pady=(0, 20))

        self.mu_history = np.zeros(100)

        self.bind("1", lambda event: self.toggle_anc())
        self.bind("2", lambda event: self.toggle_eng())
        self.bind("3", lambda event: self.toggle_road())
        self.bind("4", lambda event: self.toggle_nn())
        self.bind("5", lambda event: self.toggle_wind())

    def toggle_anc(self):
        state['anc_active'] = not state['anc_active']
        if state['anc_active']:
            self.btn_anc.configure(text="ANC ВКЛ (1)", fg_color=COLOR_ON, hover_color="#27ae60")
            self.line1.set_color('#00b894') 
        else:
            self.btn_anc.configure(text="ANC ВЫКЛ (1)", fg_color=COLOR_OFF, hover_color="#c0392b")
            self.line1.set_color(COLOR_OFF)

    def toggle_eng(self):
        state['engine_active'] = not state['engine_active']
        is_on = state['engine_active']
        self.btn_eng.configure(fg_color=COLOR_SRC_ON if is_on else COLOR_SRC_OFF,
                               text_color="white" if is_on else "#555555")

    def toggle_road(self):
        state['road_active'] = not state['road_active']
        is_on = state['road_active']
        self.btn_road.configure(fg_color=COLOR_SRC_ON if is_on else COLOR_SRC_OFF,
                                text_color="white" if is_on else "#555555")

    def toggle_wind(self):
        state['wind_active'] = not state['wind_active']
        is_on = state['wind_active']
        self.btn_wind.configure(fg_color=COLOR_SRC_ON if is_on else COLOR_SRC_OFF,
                                text_color="white" if is_on else "#555555")

    def toggle_nn(self):
        state['nn_active'] = not state['nn_active']
        if state['nn_active']:
            self.btn_nn.configure(text="Нейросеть ВКЛ (4)", fg_color=COLOR_NN, hover_color="#8e44ad")
            self.line2.set_color(COLOR_NN) 
        else:
            self.btn_nn.configure(text="Нейросеть ВЫКЛ (4)", fg_color="#e67e22", hover_color="#d35400") 
            self.line2.set_color('#e67e22')

    def update_plot(self):
        data = np.copy(gui_plot_buffer)
        self.line1.set_data(np.arange(len(data)), data)
        self.ax1.set_xlim(0, len(data))
        
        self.mu_history = np.roll(self.mu_history, -1) 
        
        if state['anc_active'] and (state['engine_active'] or state['road_active'] or state['wind_active']):
            current_mu_val = state['current_mu']
            self.mu_history[-1] = current_mu_val
            
            if state['nn_active']:
                self.lbl_mu.configure(text=f"NN (Умный μ): {current_mu_val:.5f}", text_color=COLOR_NN)
            else:
                self.lbl_mu.configure(text=f"LMS (Фикс. μ): {current_mu_val:.5f}", text_color="#e67e22")
        else:
            self.mu_history[-1] = 0.0
            self.lbl_mu.configure(text="Управление μ: ---", text_color=COLOR_TEXT_SUB)

        self.line2.set_data(np.arange(len(self.mu_history)), self.mu_history)
        self.ax2.set_xlim(0, len(self.mu_history))

        self.canvas.draw_idle()
        self.after(35, self.update_plot)

    def on_closing(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.destroy()

if __name__ == "__main__":
    app = ANCApplication()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
