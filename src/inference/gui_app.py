


import customtkinter as ctk         # Modern UI framework
import cv2                          # OpenCV for camera handling
import PIL.Image, PIL.ImageTk       # Image processing for GUI display
import threading                    # For running background tasks (Audio/AI)
import subprocess                   # For running Windows system commands
import joblib                       # For loading the trained AI model
import numpy as np                  # For numerical calculations
import os                           # Operating system interactions
import sys                          # System path handling
import queue                        # Thread-safe communication
import time                         # Time tracking for cooldowns
import random                       # Randomization for Quiz mode
import winsound                     # System beep sounds
import pyautogui                    # Automation (Keyboard/Mouse control)
import webbrowser                   # Opening websites
import speech_recognition as sr     # Voice-to-Text
from collections import deque, Counter # Data structures for stability
from textblob import TextBlob       # NLP for grammar fixing
from datetime import datetime       # Timestamping for reports
from deep_translator import GoogleTranslator # Multi-language text translation
from gtts import gTTS               # Google Text-to-Speech (Foreign Audio)
import pygame                       # Audio Player for gTTS

# --- FIX SYSTEM PATHS ---
# Ensures the application can find the 'src' folder no matter where you run it from
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)

# Import custom modules developed for this project
from src.capture.mediapipe_extractor import LandmarkExtractor
from src.train.train_landmark import train_model_function

# --- UI CONFIGURATION ---
ctk.set_appearance_mode("Dark")        # Modern Dark Theme
ctk.set_default_color_theme("blue")    # Blue accent color

# =============================================================================
# 📚 DATA DICTIONARIES (THE KNOWLEDGE BASE)
# =============================================================================

# 1. GRAMMAR FIXER MAP
# Converts "Gloss" (Sign Language structure) to proper English.
GRAMMAR_MAP = {
    "me": "I am", "i": "I am", "you": "you are", "we": "we are",
    "name": "my name is", "hungry": "hungry for food", "go": "going to",
    "want": "want to", "thanks": "thank you", "please": "please can I have",
    "yes": "yes, I agree", "no": "no, I disagree"
}

# 2. EMOJI MAP
# Adds visual flair to the output.
EMOJI_MAP = {
    "Hello": "👋", "Yes": "✅", "No": "❌", "Thanks": "🙏",
    "I": "😊", "Love": "❤", "You": "👉", "Happy": "😄",
    "Sad": "😢", "Please": "🥺", "Clap": "👏", "House": "🏠",
    "Up": "🔊", "Down": "🔉", "Open": "🌐", "Shot": "📸",
    "Youtube": "📺", "Google": "🔍", "Wiki": "📚"
}

# 3. CONTROL MAP (The "Iron Man" Features)
# Maps a specific hand sign to a specific computer keystroke or command.
CONTROL_MAP = {
    "Up": "volumeup",       # Increases PC Volume
    "Down": "volumedown",   # Decreases PC Volume
    "Shot": "screenshot",   # Takes Screenshot
    "Open": "google",       # Custom keyword to open Google
    "Youtube": "youtube",   # Custom keyword to open YouTube
    "Wiki": "wikipedia"     # Custom keyword to open Wikipedia
}

# 4. SUPPORTED LANGUAGES
# List of languages available for real-time translation.
LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Hindi": "hi", "Urdu": "ur", "Chinese": "zh-CN"
}

# =============================================================================
# 🖥️ MAIN APPLICATION CLASS
# =============================================================================
class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- WINDOW SETUP ---
        self.title("AI Sign Language Suite (Offline Edition)")
        self.geometry("1400x950")

        # --- STATE VARIABLES ---
        self.sentence = []              # Stores the history of recognized signs
        self.last_stable_label = ""     # The last confirmed sign
        self.model = None               # The Random Forest AI Model
        self.label_encoder = None       # Decodes Numbers back to Text
        
        # Audio Queue: Stores text to be spoken so the UI doesn't freeze
        self.audio_queue = queue.Queue()
        
        # Prediction Stabilizer:
        # A buffer that stores the last 15 frames to prevent flickering.
        self.prediction_history = deque(maxlen=15)
        
        # Game Variables
        self.game_active = False        # Is the user playing the quiz?
        self.score = 0                  # Current score
        self.target_word = ""           # The word the user must sign
        self.time_left = 0              # Timer for the round
        self.game_classes = []          # List of all signs known by the AI

        # Training Variables (For "Teach AI" Mode)
        self.is_recording = False       # Are we currently saving data?
        self.record_counter = 0         # How many samples collected
        self.new_data_buffer = []       # Temporary storage for new data
        self.last_action_time = 0       # Cooldown timer to prevent spamming controls

        # --- INITIALIZATION ---
        
        # 1. Start the Audio Worker Thread (For Fast English System Voice)
        threading.Thread(target=self.audio_worker, daemon=True).start()
        
        # 2. Load the AI Brain
        self.init_ai()

        # --- LAYOUT CONFIGURATION ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create the Tab View (The main navigation container)
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Add Tabs
        self.tab_trans = self.tab_view.add("🗣 Translator & AI")
        self.tab_quiz = self.tab_view.add("🎮 Quiz Mode")
        self.tab_teach = self.tab_view.add("➕ Teach AI")

        # Build the UI for each tab
        self.setup_translator_tab()
        self.setup_quiz_tab()
        self.setup_teach_tab()

        # --- CAMERA SETUP ---
        # Initialize OpenCV to capture video from the default webcam (Index 0)
        self.cap = cv2.VideoCapture(0)
        
        # Initialize MediaPipe Extractor (The "Eyes" of the AI)
        self.extractor = LandmarkExtractor()
        
        # Start the Main Application Loop
        self.update_camera()

    def init_ai(self):
        """Loads the trained Random Forest model from disk."""
        try:
            model_path = os.path.join(root_dir, "model", "sign_model.pkl")
            label_path = os.path.join(root_dir, "model", "label_encoder.pkl")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load(label_path)
                self.game_classes = list(self.label_encoder.classes_)
                print("✅ AI Model Loaded Successfully")
            else:
                print("⚠ No model found. Please use 'Teach AI' tab first.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    # =========================================================================
    # 🎨 UI CONSTRUCTION
    # =========================================================================

    def setup_translator_tab(self):
        """Builds the Main Interface (Translator, AI, Controls)"""
        self.tab_trans.grid_columnconfigure(1, weight=1)
        self.tab_trans.grid_rowconfigure(0, weight=1)

        # Sidebar (Controls)
        sidebar = ctk.CTkFrame(self.tab_trans, width=250, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        ctk.CTkLabel(sidebar, text="SMART CONTROLS", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Language Selector
        ctk.CTkLabel(sidebar, text="Output Language:").pack(pady=(5,0))
        self.lang_var = ctk.StringVar(value="English")
        self.combo_lang = ctk.CTkOptionMenu(sidebar, variable=self.lang_var, values=list(LANGUAGES.keys()))
        self.combo_lang.pack(pady=5)
        
        # AI Button (Offline Demo)
        self.btn_ai = ctk.CTkButton(sidebar, text="🤖 Ask AI Assistant", fg_color="#10a37f", command=self.ask_demo_ai)
        self.btn_ai.pack(pady=5)

        # Translate Button
        ctk.CTkButton(sidebar, text="🌍 Translate Text", fg_color="#FF9900", command=self.translate_sentence).pack(pady=5)
        
        # Voice Input Button
        ctk.CTkLabel(sidebar, text="Input Methods:").pack(pady=(15,0))
        self.btn_listen = ctk.CTkButton(sidebar, text="🎙 Voice Input", fg_color="#E0aaff", text_color="black", command=self.listen_to_speech)
        self.btn_listen.pack(pady=5)
        
        # Toggles
        self.voice_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(sidebar, text="Auto-Speak", variable=self.voice_var).pack(pady=10)

        self.control_var = ctk.BooleanVar(value=False)
        self.switch_control = ctk.CTkSwitch(sidebar, text="💻 PC Control", variable=self.control_var, progress_color="cyan")
        self.switch_control.pack(pady=5)

        # Utilities
        ctk.CTkButton(sidebar, text="✨ Smart Fix", fg_color="#6A0DAD", command=self.smart_fix_sentence).pack(pady=(20, 5))
        ctk.CTkButton(sidebar, text="🗑 Clear", fg_color="firebrick", command=self.clear_text).pack(pady=5)
        ctk.CTkButton(sidebar, text="📄 Report", fg_color="gray", command=self.generate_report).pack(pady=5)

        # Main Camera & Text Area
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
        """Builds the Game Interface"""
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
        """Builds the Training Interface"""
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

    # =========================================================================
    # 🧠 INTELLIGENT MODULES (LOGIC & AI)
    # =========================================================================
    
    # --- 1. DEMO AI ASSISTANT (Stable Version - NO INTERNET NEEDED) ---
    def ask_demo_ai(self):
        """
        Simulates an intelligent conversation without needing an internet API key.
        Great for presentations where you want reliability.
        """
        question = self.output_box.get("0.0", "end").strip().lower()
        if not question or "start" in question: return
        
        self.btn_ai.configure(text="🤔 Thinking...", state="disabled")
        
        def _ai_thread():
            time.sleep(1) # Fake "thinking" delay to look realistic
            
            # Logic to match keywords to answers
            if "hello" in question or "hi" in question:
                response = "Hello there! How can I assist you with Sign Language today?"
            elif "name" in question:
                response = "I am your AI Sign Language Assistant."
            elif "python" in question:
                response = "Python is a powerful programming language used for AI and Web Development."
            elif "weather" in question:
                response = "I cannot check live weather, but I hope it is sunny outside!"
            elif "sign" in question:
                response = "Sign Language is a visual way to communicate using hand gestures."
            elif "thank" in question:
                response = "You are very welcome!"
            else:
                response = f"That is an interesting topic about '{question}'. In a full version, I would explain it in detail."

            # Update UI and Speak
            self.output_box.delete("0.0", "end")
            self.output_box.insert("0.0", f"AI: {response}")
            self.audio_queue.put(response)
            
            self.btn_ai.configure(text="🤖 Ask AI Assistant", state="normal")

        threading.Thread(target=_ai_thread).start()

    # --- 2. MULTI-LANGUAGE TRANSLATOR ---
    def translate_sentence(self):
        """Translates the text in the box to the selected language"""
        text = self.output_box.get("0.0", "end").strip()
        target_lang_name = self.lang_var.get()
        target_code = LANGUAGES[target_lang_name]
        
        if not text or "Start" in text: return
        try:
            # Call Google Translator API
            translated = GoogleTranslator(source='auto', target=target_code).translate(text)
            
            self.output_box.delete("0.0", "end")
            self.output_box.insert("0.0", f"[{target_lang_name}]: {translated}")
            
            # Note: We only speak English for simplicity in this demo version
            if target_code == 'en': self.audio_queue.put(translated)
            else: 
                # Foreign language? Use Google TTS
                self.speak_foreign_language(translated, target_code)
                
        except: 
            self.output_box.insert("end", "\n[Error: Check Internet]")

    def speak_foreign_language(self, text, lang_code):
        """Uses Google TTS to download and play audio for non-English languages."""
        def _speak_thread():
            try:
                # 1. Generate MP3
                tts = gTTS(text=text, lang=lang_code)
                filename = f"temp_audio_{int(time.time())}.mp3"
                tts.save(filename)
                
                # 2. Play Audio
                pygame.mixer.init()
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # 3. Cleanup
                pygame.mixer.quit()
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                print(f"Audio Error: {e}")
                winsound.Beep(1000, 200)

        threading.Thread(target=_speak_thread).start()

    # --- 3. VOICE COMMAND PROCESSOR ---
    def listen_to_speech(self):
        """Activates microphone to listen for commands"""
        def _listen():
            self.btn_listen.configure(text="🎤 Listening...", fg_color="red")
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    winsound.Beep(600, 100) # Ready beep
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio).lower()
                    
                    self.output_box.delete("0.0", "end")
                    self.output_box.insert("0.0", f"{text}")
                    
                    # Analyze text for control commands
                    self.process_voice_command(text)
            except:
                self.output_box.delete("0.0", "end")
                self.output_box.insert("0.0", "❌ Didn't catch that.")
            self.btn_listen.configure(text="🎙 Voice Input", fg_color="#E0aaff")
        threading.Thread(target=_listen).start()

    def process_voice_command(self, text):
        """Matches voice input to actions"""
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

    # --- 4. UTILITIES (Report & Prediction Stability) ---
    def generate_report(self):
        """Saves a text file with the current session data"""
        text = self.output_box.get("0.0", "end").strip()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Report_{timestamp}.txt"
        with open(filename, "w") as f: f.write(f"AI REPORT\n{datetime.now()}\n\n{text}")
        self.output_box.insert("end", f"\n\n[✅ Saved: {filename}]")
        winsound.Beep(1000, 100)

    def get_stable_prediction(self, new_label):
        """
        Prevents the AI from glitching. 
        It waits until the SAME sign is detected 12 times in the last 15 frames 
        before confirming it.
        """
        self.prediction_history.append(new_label)
        if len(self.prediction_history) < 15: return None
        
        counter = Counter(self.prediction_history)
        most_common_label, count = counter.most_common(1)[0]
        
        # 80% Consistency required
        if count >= 12: return most_common_label
        return None

    # --- 5. PC CONTROL (HCI) ---
    def trigger_system_action(self, label):
        """Triggers Mouse/Keyboard events based on hand signs"""
        current_time = time.time()
        
        # 1. Cooldown Check (Prevent spamming the command)
        if current_time - self.last_action_time < 1.0: return
        
        if label in CONTROL_MAP:
            action = CONTROL_MAP[label]
            if action == "google": webbrowser.open("https://google.com")
            elif action == "youtube": webbrowser.open("https://youtube.com")
            elif action == "wikipedia": webbrowser.open("https://wikipedia.org")
            elif action == "screenshot": 
                pyautogui.screenshot(f"shot_{int(current_time)}.png")
                winsound.Beep(1000, 50)
            else: pyautogui.press(action) # Press keyboard key (e.g., Volume Up)
            
            self.last_action_time = current_time

    # --- 6. AUDIO ENGINE (Thread-Safe) ---
    def audio_worker(self):
        """
        Continuously checks the queue for text and speaks it.
        Uses PowerShell to avoid Python library freezing issues.
        """
        while True:
            text = self.audio_queue.get() # Waits here for text
            if text is None: break
            try:
                # Sanitize text
                safe_text = text.replace("'", "").replace('"', '')
                # Call Windows TTS
                cmd = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{safe_text}\');"'
                subprocess.run(cmd, shell=True)
            except: pass
            self.audio_queue.task_done()

    # --- 7. TEACHING & TRAINING LOGIC ---
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
        
        # Save new data to file
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
        
        # Run training script in background
        threading.Thread(target=self.run_training_process, daemon=True).start()

    def run_training_process(self):
        acc, msg = train_model_function()
        self.lbl_train_status.configure(text=f"Done! Acc: {acc*100:.1f}%", text_color="green")
        self.init_ai() # Reload new model immediately
        self.new_data_buffer = []
        self.record_counter = 0
        self.lbl_sample_count.configure(text="Samples: 0/50")
        self.entry_sign_name.delete(0, 'end')

    # =========================================================================
    # 🎥 MAIN LOOP (FRAME-BY-FRAME PROCESSING)
    # =========================================================================
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Extract Hand Landmarks
            vec = self.extractor.extract(frame)
            hand_found = np.any(vec)

            # --- CASE A: RECORDING NEW DATA ---
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

            # --- CASE B: PREDICTING SIGNS ---
            if self.model and not self.is_recording and hand_found:
                features = np.array(vec).reshape(1, -1)
                probs = self.model.predict_proba(features)[0]
                idx = np.argmax(probs)
                conf = probs[idx]

                # Update Confidence Bar
                self.progress_bar.set(conf)
                self.lbl_conf.configure(text=f"Confidence: {int(conf*100)}%")
                threshold = 0.85

                if conf > threshold:
                    raw_label = self.label_encoder.inverse_transform([idx])[0]
                    final_label = self.get_stable_prediction(raw_label)
                    
                    if final_label:
                        # 1. Check for PC Control
                        if self.control_var.get():
                            self.trigger_system_action(final_label)
                            cv2.putText(rgb_frame, f"CMD: {final_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        # 2. Standard Translation
                        else:
                            emoji = EMOJI_MAP.get(final_label, "")
                            display = f"{final_label} {emoji}"
                            cv2.putText(rgb_frame, f"Sign: {display}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            if final_label != self.last_stable_label:
                                self.sentence.append(final_label)
                                self.output_box.delete("0.0", "end")
                                self.output_box.insert("0.0", " ".join(self.sentence))
                                
                                # Speak only if NOT playing a game
                                if self.voice_var.get() and not self.game_active: 
                                    self.audio_queue.put(final_label)
                                
                                self.last_stable_label = final_label
                        
                        # 3. Check Quiz Answer
                        if self.game_active: self.check_game_answer(final_label)
            
            elif not hand_found:
                cv2.putText(rgb_frame, "Show Hands", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.last_stable_label = ""

            # 3. Render Frame to GUI
            img = PIL.Image.fromarray(rgb_frame)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            
            current_tab = self.tab_view.get()
            if current_tab == "🗣 Translator & AI": self.cam_label_trans.configure(image=ctk_img)
            elif current_tab == "🎮 Quiz Mode": self.cam_label_quiz.configure(image=ctk_img)
            elif current_tab == "➕ Teach AI": self.cam_label_teach.configure(image=ctk_img)

        self.after(10, self.update_camera)
    
    # --- HELPER FUNCS ---
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

# --- ENTRY POINT ---
if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()