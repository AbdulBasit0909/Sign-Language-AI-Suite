import customtkinter as ctk
import cv2
import PIL.Image, PIL.ImageTk
import threading
import subprocess
import joblib
import numpy as np
import os
import sys
import queue
import time
import random
import winsound
import pyautogui
import webbrowser
import speech_recognition as sr
from collections import deque, Counter
from textblob import TextBlob
from datetime import datetime
from deep_translator import GoogleTranslator

# --- NEW: CHATGPT LIBRARY ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)

from src.capture.mediapipe_extractor import LandmarkExtractor
from src.train.train_landmark import train_model_function

# --- CONFIG ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue") 

# --- API KEY (PASTE YOUR KEY HERE IF YOU HAVE ONE) ---
# If you leave this blank, the app will use "Demo Mode" (Fake AI) for your presentation.
OPENAI_API_KEY = "" 

# --- MAPPINGS ---
GRAMMAR_MAP = {
    "me": "I am", "i": "I am", "you": "you are", "we": "we are",
    "name": "my name is", "hungry": "hungry for food", "go": "going to",
    "want": "want to", "thanks": "thank you", "please": "please can I have",
    "yes": "yes, I agree", "no": "no, I disagree", "what": "what is"
}

EMOJI_MAP = {
    "Hello": "👋", "Yes": "✅", "No": "❌", "Thanks": "🙏",
    "I": "😊", "Love": "❤", "You": "👉", "Happy": "😄",
    "Sad": "😢", "Please": "🥺", "Clap": "👏", "House": "🏠",
    "Up": "🔊", "Down": "🔉", "Open": "🌐", "Shot": "📸",
    "Youtube": "📺", "Google": "🔍", "Wiki": "📚"
}

CONTROL_MAP = {
    "Up": "volumeup", "Down": "volumedown", "Shot": "screenshot",
    "Open": "google", "Youtube": "youtube", "Wiki": "wikipedia"
}

LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Hindi": "hi", "Urdu": "ur", "Chinese": "zh-CN"
}

class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Sign Language Assistant (ChatGPT Integrated)")
        self.geometry("1400x950")

        # --- VARS ---
        self.sentence = []
        self.last_stable_label = ""
        self.model = None
        self.label_encoder = None
        self.audio_queue = queue.Queue()
        self.prediction_history = deque(maxlen=15)
        
        self.game_active = False
        self.score = 0
        self.target_word = ""
        self.time_left = 0
        self.game_classes = []

        self.is_recording = False
        self.record_counter = 0
        self.new_data_buffer = []
        self.last_action_time = 0 

        # Load AI
        threading.Thread(target=self.audio_worker, daemon=True).start()
        self.init_ai()

        # --- LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.tab_trans = self.tab_view.add("🗣 Translator & AI")
        self.tab_quiz = self.tab_view.add("🎮 Quiz Mode")
        self.tab_teach = self.tab_view.add("➕ Teach AI")

        self.setup_translator_tab()
        self.setup_quiz_tab()
        self.setup_teach_tab()

        self.cap = cv2.VideoCapture(0)
        self.extractor = LandmarkExtractor()
        
        self.update_camera()

    def init_ai(self):
        try:
            model_path = os.path.join(root_dir, "model", "sign_model.pkl")
            label_path = os.path.join(root_dir, "model", "label_encoder.pkl")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load(label_path)
                self.game_classes = list(self.label_encoder.classes_)
                print("✅ AI Model Loaded")
            else:
                print("⚠ No model found. Please use Teach Mode.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    # ================= UI SETUP =================
    def setup_translator_tab(self):
        self.tab_trans.grid_columnconfigure(1, weight=1)
        self.tab_trans.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self.tab_trans, width=250, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        ctk.CTkLabel(sidebar, text="SMART CONTROLS", font=("Arial", 16, "bold")).pack(pady=10)
        
        # --- SECTION 1: AI & TRANSLATION ---
        ctk.CTkLabel(sidebar, text="Language & AI:").pack(pady=(5,0))
        self.lang_var = ctk.StringVar(value="English")
        self.combo_lang = ctk.CTkOptionMenu(sidebar, variable=self.lang_var, values=list(LANGUAGES.keys()))
        self.combo_lang.pack(pady=5)
        
        # CHATGPT BUTTON
        self.btn_ai = ctk.CTkButton(sidebar, text="🤖 Ask AI (ChatGPT)", fg_color="#10a37f", command=self.ask_chatgpt)
        self.btn_ai.pack(pady=5)

        ctk.CTkButton(sidebar, text="🌍 Translate Text", fg_color="#FF9900", command=self.translate_sentence).pack(pady=5)
        
        # --- SECTION 2: INPUTS ---
        ctk.CTkLabel(sidebar, text="Input Methods:").pack(pady=(15,0))
        self.btn_listen = ctk.CTkButton(sidebar, text="🎙 Voice Input", fg_color="#E0aaff", text_color="black", command=self.listen_to_speech)
        self.btn_listen.pack(pady=5)
        
        # --- SECTION 3: TOGGLES ---
        self.voice_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(sidebar, text="Auto-Speak", variable=self.voice_var).pack(pady=10)

        self.control_var = ctk.BooleanVar(value=False)
        self.switch_control = ctk.CTkSwitch(sidebar, text="💻 PC Control", variable=self.control_var, progress_color="cyan")
        self.switch_control.pack(pady=5)

        # --- SECTION 4: UTILS ---
        ctk.CTkButton(sidebar, text="✨ Smart Fix", fg_color="#6A0DAD", command=self.smart_fix_sentence).pack(pady=(20, 5))
        ctk.CTkButton(sidebar, text="🗑 Clear", fg_color="firebrick", command=self.clear_text).pack(pady=5)
        ctk.CTkButton(sidebar, text="📄 Save Report", fg_color="gray", command=self.generate_report).pack(pady=5)

        # MAIN DISPLAY
        main_area = ctk.CTkFrame(self.tab_trans, fg_color="transparent")
        main_area.grid(row=0, column=1, sticky="nsew", padx=10)

        self.cam_label_trans = ctk.CTkLabel(main_area, text="Camera Loading...")
        self.cam_label_trans.pack(fill="both", expand=True)

        self.lbl_conf = ctk.CTkLabel(main_area, text="Confidence: 0%")
        self.lbl_conf.pack(anchor="w")
        self.progress_bar = ctk.CTkProgressBar(main_area)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=5)

        self.output_box = ctk.CTkTextbox(main_area, height=120, font=("Roboto", 24))
        self.output_box.pack(fill="x", pady=5)
        self.output_box.insert("0.0", "Start signing or type a question...")

    def setup_quiz_tab(self):
        self.tab_quiz.grid_columnconfigure(0, weight=1)
        self.tab_quiz.grid_rowconfigure(1, weight=1)
        top_frame = ctk.CTkFrame(self.tab_quiz, height=80)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.lbl_score = ctk.CTkLabel(top_frame, text="SCORE: 0", font=("Impact", 24), text_color="#00FF00")
        self.lbl_score.pack(side="left", padx=30)
        self.lbl_target = ctk.CTkLabel(top_frame, text="PRESS START", font=("Arial", 30, "bold"))
        self.lbl_target.pack(side="left", expand=True)
        self.lbl_timer = ctk.CTkLabel(top_frame, text="⏳ 0s", font=("Arial", 24))
        self.lbl_timer.pack(side="right", padx=30)
        self.cam_label_quiz = ctk.CTkLabel(self.tab_quiz, text="Camera Loading...")
        self.cam_label_quiz.grid(row=1, column=0, sticky="nsew", padx=10)
        ctk.CTkButton(self.tab_quiz, text="START QUIZ", height=50, font=("Arial", 20), command=self.start_game).grid(row=2, column=0, pady=20)

    def setup_teach_tab(self):
        self.tab_teach.grid_columnconfigure(1, weight=1)
        self.tab_teach.grid_rowconfigure(0, weight=1)
        teach_panel = ctk.CTkFrame(self.tab_teach, width=250)
        teach_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(teach_panel, text="TEACH NEW SIGN", font=("Arial", 18, "bold")).pack(pady=20)
        self.entry_sign_name = ctk.CTkEntry(teach_panel, placeholder_text="Enter Sign Name")
        self.entry_sign_name.pack(pady=10, padx=10)
        self.btn_record = ctk.CTkButton(teach_panel, text="🔴 Record Samples", fg_color="red", command=self.toggle_recording)
        self.btn_record.pack(pady=10)
        self.lbl_sample_count = ctk.CTkLabel(teach_panel, text="Samples: 0/50")
        self.lbl_sample_count.pack(pady=5)
        self.btn_train = ctk.CTkButton(teach_panel, text="⚡ Train Model", state="disabled", command=self.start_training)
        self.btn_train.pack(pady=20)
        self.lbl_train_status = ctk.CTkLabel(teach_panel, text="Status: Ready", text_color="gray")
        self.lbl_train_status.pack(pady=10)
        self.cam_label_teach = ctk.CTkLabel(self.tab_teach, text="Camera View")
        self.cam_label_teach.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    # ================= LOGIC MODULES =================
    
    # --- NEW: CHATGPT LOGIC ---
    def ask_chatgpt(self):
        """Sends current text to AI and speaks response"""
        question = self.output_box.get("0.0", "end").strip()
        if not question or "Start" in question: return
        
        self.btn_ai.configure(text="🤔 Thinking...", state="disabled")
        
        def _ai_thread():
            response_text = ""
            
            # 1. CHECK IF API KEY EXISTS (Real Mode)
            if OPENAI_AVAILABLE and OPENAI_API_KEY:
                try:
                    client = openai.OpenAI(api_key=OPENAI_API_KEY)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Answer concisely: {question}"}]
                    )
                    response_text = response.choices[0].message.content
                except Exception as e:
                    response_text = "Error connecting to OpenAI. Check API Key."
            
            # 2. DEMO MODE (If no key is provided)
            else:
                # Simulating intelligence for the demo
                lower_q = question.lower()
                if "python" in lower_q: response_text = "Python is a high-level programming language known for its simplicity."
                elif "hello" in lower_q: response_text = "Hello there! How can I assist you with Sign Language today?"
                elif "weather" in lower_q: response_text = "I cannot check live weather, but it looks like a nice day inside!"
                elif "name" in lower_q: response_text = "I am your AI Sign Language Assistant."
                else: response_text = f"That is an interesting question about '{question}'. In a full version, I would define it using GPT."

            # 3. UPDATE UI & SPEAK
            self.output_box.delete("0.0", "end")
            self.output_box.insert("0.0", f"AI: {response_text}")
            self.audio_queue.put(response_text)
            self.btn_ai.configure(text="🤖 Ask AI (ChatGPT)", state="normal")

        threading.Thread(target=_ai_thread).start()

    # --- TRANSLATION ---
    def translate_sentence(self):
        text = self.output_box.get("0.0", "end").strip()
        target_lang_name = self.lang_var.get()
        target_code = LANGUAGES[target_lang_name]
        if not text or "Start" in text: return
        try:
            translated = GoogleTranslator(source='auto', target=target_code).translate(text)
            self.output_box.delete("0.0", "end")
            self.output_box.insert("0.0", f"[{target_lang_name}]: {translated}")
            if target_code == 'en': self.audio_queue.put(translated)
            else: winsound.Beep(1000, 200)
        except: self.output_box.insert("end", "\n[Error]")

    # --- VOICE COMMANDS ---
    def listen_to_speech(self):
        def _listen():
            self.btn_listen.configure(text="🎤 Listening...", fg_color="red")
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    winsound.Beep(600, 100)
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio).lower()
                    self.output_box.delete("0.0", "end")
                    self.output_box.insert("0.0", f"{text}")
                    self.process_voice_command(text)
            except:
                self.output_box.delete("0.0", "end")
                self.output_box.insert("0.0", "❌ Didn't catch that.")
            self.btn_listen.configure(text="🎙 Voice Input", fg_color="#E0aaff")
        threading.Thread(target=_listen).start()

    def process_voice_command(self, text):
        performed = False
        if "youtube" in text:
            webbrowser.open("https://www.youtube.com")
            performed = True
        elif "google" in text:
            webbrowser.open("https://www.google.com")
            performed = True
        elif "wiki" in text:
            webbrowser.open("https://www.wikipedia.org")
            performed = True
        elif "screenshot" in text:
            pyautogui.screenshot(f"voice_shot_{int(time.time())}.png")
            performed = True
        if performed: winsound.Beep(1000, 100)

    # --- UTILS ---
    def generate_report(self):
        text = self.output_box.get("0.0", "end").strip()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Report_{timestamp}.txt"
        with open(filename, "w") as f: f.write(f"AI REPORT\n{datetime.now()}\n\n{text}")
        self.output_box.insert("end", f"\n\n[✅ Saved: {filename}]")
        winsound.Beep(1000, 100)

    def get_stable_prediction(self, new_label):
        self.prediction_history.append(new_label)
        if len(self.prediction_history) < 15: return None
        counter = Counter(self.prediction_history)
        most_common_label, count = counter.most_common(1)[0]
        if count >= 12: return most_common_label
        return None

    def trigger_system_action(self, label):
        current_time = time.time()
        if current_time - self.last_action_time < 1.0: return
        if label in CONTROL_MAP:
            action = CONTROL_MAP[label]
            if action == "google": webbrowser.open("https://google.com")
            elif action == "youtube": webbrowser.open("https://youtube.com")
            elif action == "wikipedia": webbrowser.open("https://wikipedia.org")
            elif action == "screenshot": 
                pyautogui.screenshot(f"shot_{int(current_time)}.png")
                winsound.Beep(1000, 50)
            else: pyautogui.press(action)
            self.last_action_time = current_time

    def audio_worker(self):
        while True:
            text = self.audio_queue.get()
            if text is None: break
            try:
                safe_text = text.replace("'", "")
                cmd = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{safe_text}\');"'
                subprocess.run(cmd, shell=True)
            except: pass
            self.audio_queue.task_done()

    # --- TEACH ---
    def toggle_recording(self):
        name = self.entry_sign_name.get().strip()
        if not name:
            self.lbl_train_status.configure(text="⚠ Enter Name First!", text_color="orange")
            return
        if not self.is_recording:
            self.is_recording = True
            self.new_data_buffer = []
            self.record_counter = 0
            self.btn_record.configure(text="Stop Recording")
            self.lbl_train_status.configure(text="Recording...", text_color="yellow")
        else:
            self.is_recording = False
            self.btn_record.configure(text="🔴 Record Samples")

    def start_training(self):
        self.btn_train.configure(state="disabled")
        self.lbl_train_status.configure(text="Training...", text_color="blue")
        dataset_path = os.path.join(root_dir, "dataset.npy")
        new_data = np.array(self.new_data_buffer, dtype=object)
        try:
            if os.path.exists(dataset_path):
                existing = np.load(dataset_path, allow_pickle=True)
                final = np.vstack([existing, new_data])
            else: final = new_data
            np.save(dataset_path, final)
        except Exception as e:
            self.lbl_train_status.configure(text=f"Error: {e}", text_color="red")
            return
        threading.Thread(target=self.run_training_process, daemon=True).start()

    def run_training_process(self):
        acc, msg = train_model_function()
        self.lbl_train_status.configure(text=f"Done! Acc: {acc*100:.1f}%", text_color="green")
        self.init_ai()
        self.new_data_buffer = []
        self.record_counter = 0
        self.lbl_sample_count.configure(text="Samples: 0/50")
        self.entry_sign_name.delete(0, 'end')

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vec = self.extractor.extract(frame)
            hand_found = np.any(vec)

            if self.is_recording and hand_found:
                label = self.entry_sign_name.get().strip()
                if label:
                    self.new_data_buffer.append(np.append(vec, label))
                    self.record_counter += 1
                    self.lbl_sample_count.configure(text=f"Samples: {self.record_counter}")
                    if self.record_counter >= 50:
                        self.is_recording = False
                        self.btn_record.configure(text="🔴 Record Samples")
                        self.btn_train.configure(state="normal")
                        winsound.Beep(1000, 200)

            if self.model and not self.is_recording and hand_found:
                features = np.array(vec).reshape(1, -1)
                probs = self.model.predict_proba(features)[0]
                idx = np.argmax(probs)
                conf = probs[idx]

                self.progress_bar.set(conf)
                self.lbl_conf.configure(text=f"Confidence: {int(conf*100)}%")
                threshold = 0.85

                if conf > threshold:
                    raw_label = self.label_encoder.inverse_transform([idx])[0]
                    final_label = self.get_stable_prediction(raw_label)
                    
                    if final_label:
                        if self.control_var.get():
                            self.trigger_system_action(final_label)
                            cv2.putText(rgb_frame, f"CMD: {final_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else:
                            emoji = EMOJI_MAP.get(final_label, "")
                            display = f"{final_label} {emoji}"
                            cv2.putText(rgb_frame, f"Sign: {display}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            if final_label != self.last_stable_label:
                                self.sentence.append(final_label)
                                self.output_box.delete("0.0", "end")
                                self.output_box.insert("0.0", " ".join(self.sentence))
                                if self.voice_var.get() and not self.game_active: 
                                    self.audio_queue.put(final_label)
                                self.last_stable_label = final_label
                        
                        if self.game_active: self.check_game_answer(final_label)
            
            elif not hand_found:
                cv2.putText(rgb_frame, "Show Hands", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.last_stable_label = ""

            img = PIL.Image.fromarray(rgb_frame)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            current_tab = self.tab_view.get()
            if current_tab == "🗣 Translator & AI": self.cam_label_trans.configure(image=ctk_img)
            elif current_tab == "🎮 Quiz Mode": self.cam_label_quiz.configure(image=ctk_img)
            elif current_tab == "➕ Teach AI": self.cam_label_teach.configure(image=ctk_img)

        self.after(10, self.update_camera)
    
    # --- HELPER FUNCS START HERE (Copy from previous if needed, mostly included above) ---
    # (Note: All main helper functions like start_game, smart_fix_sentence, clear_text are included in the class above)
    def start_game(self):
        if not self.game_classes: return
        self.score = 0
        self.game_active = True
        self.next_round()

    def next_round(self):
        if not self.game_active: return
        self.target_word = random.choice(self.game_classes)
        self.lbl_target.configure(text=f"SHOW ME: {self.target_word.upper()}")
        self.audio_queue.put(f"Show me {self.target_word}")
        self.time_left = 10
        self.update_timer()

    def update_timer(self):
        if not self.game_active: return
        self.lbl_timer.configure(text=f"⏳ {self.time_left}s")
        if self.time_left > 0:
            self.time_left -= 1
            self.after(1000, self.update_timer)
        else:
            winsound.Beep(500, 500)
            self.lbl_target.configure(text=f"TIME UP! {self.target_word}")
            self.game_active = False

    def check_game_answer(self, label):
        if self.game_active and label == self.target_word:
            self.score += 10
            self.lbl_score.configure(text=f"SCORE: {self.score}")
            self.lbl_target.configure(text="✅ CORRECT!")
            winsound.Beep(1000, 200)
            self.game_active = False
            self.after(1500, lambda: [setattr(self, 'game_active', True), self.next_round()])

    def smart_fix_sentence(self):
        raw = self.output_box.get("0.0", "end").strip()
        if not raw: return
        words = raw.split()
        new_s = []
        for w in words:
            if w.lower() in GRAMMAR_MAP: new_s.append(GRAMMAR_MAP[w.lower()])
            else: new_s.append(str(TextBlob(w).correct()))
        final = " ".join(new_s).capitalize() + "."
        self.output_box.delete("0.0", "end")
        self.output_box.insert("0.0", final)
        if self.voice_var.get(): self.audio_queue.put(final)

    def clear_text(self):
        self.sentence = []
        self.output_box.delete("0.0", "end")

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()