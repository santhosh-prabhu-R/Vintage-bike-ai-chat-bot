import customtkinter as ctk
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import os
import nltk
from nltk.tokenize import sent_tokenize

# --- PRE-LOAD NLTK DATA ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"NLTK Download error: {e}")

# -------------------------------
# DATABASE SETUP
# -------------------------------
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, bot TEXT)")
conn.commit()

# -------------------------------
# KNOWLEDGE & IMAGE MAPPING
# -------------------------------
def load_knowledge_base(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if script_dir.endswith('.vscode') else script_dir
    full_path = os.path.join(project_root, file_name)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if len(s) > 10]
    except FileNotFoundError:
        return ["Error: knowledge.txt not found."]

knowledge_sentences = load_knowledge_base("knowledge.txt")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(knowledge_sentences)

bike_image_map = {
    "rx100": "rx100.jpg",
    "rd350": "rd350.jpg",
    "jawa": "jawa.jpg",
    "yezdi": "yezdi.jpg",
    "kb100": "kb100.jpg",
    "samurai": "samurai.jpg"
}

# -------------------------------
# RESPONSE & IMAGE LOGIC
# -------------------------------
def get_response(user_input):
    user_input_clean = user_input.lower().strip()
    
    # 1. FRIENDLY GREETING LAYER (Rule-Based)
    greetings = {
        "hi": "Hey! Great to see you. Ready to dive into some vintage 2-stroke history? 🏍️",
        "hello": "Hello there! I was just thinking about the RD350. What's on your mind?",
        "hey": "Hey! Always happy to chat with a fellow enthusiast. What can I help you find?",
        "thanks": "No problem at all! I'm here whenever you need a technical deep-dive.",
        "thank you": "You're very welcome! It’s my pleasure to help out.",
        "bye": "Catch you later! Keep that engine revving and ride safe.",
        "who are you": "I'm your AI collaborator, specializing in all things two-stroke and vintage!"
    }
    
    if user_input_clean in greetings:
        return greetings[user_input_clean]

    # 2. ML LAYER (Retrieval-Based)
  # 2. ML LAYER (Document Retrieval)
    user_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vec, X)
    index = np.argmax(similarity)
    
    # This helps you see why a question might be failing in the terminal
    print(f"Debug Score: {similarity[0][index]:.2f} for query: {user_input_clean}")
    
    # Lowering the threshold to 0.10 makes it easier to match longer questions
    if similarity[0][index] > 0.10: 
        return knowledge_sentences[index]
    
    return "I'm not quite sure about that one. Try asking about the RX100, Yezdi, or engine mechanics!"

def update_preview_image(user_input):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if script_dir.endswith('.vscode') else script_dir
    img_folder = os.path.join(project_root, "images")
    user_input_lower = user_input.lower()
    
    for bike_key, filename in bike_image_map.items():
        if bike_key in user_input_lower:
            img_path = os.path.join(img_folder, filename)
            if os.path.exists(img_path):
                raw_img = Image.open(img_path)
                ctk_img = ctk.CTkImage(light_image=raw_img, dark_image=raw_img, size=(280, 200))
                photo_display.configure(image=ctk_img, text=f"Viewing: {bike_key.upper()}")
                return

# -------------------------------
# UI INITIALIZATION
# -------------------------------
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.title("2-Stroke Vintage AI Hub")
app.geometry("1150x750")
VINTAGE_ORANGE = "#E67E22"

main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.pack(fill="both", expand=True)

left_panel = ctk.CTkFrame(main_frame, fg_color="transparent")
left_panel.pack(side="left", fill="both", expand=True, padx=20)

# Centered Header Section
header_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
header_frame.pack(pady=20, fill="x")
center_container = ctk.CTkFrame(header_frame, fg_color="transparent")
center_container.pack(anchor="center")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if script_dir.endswith('.vscode') else script_dir
logo_path = os.path.join(project_root, "images", "logo.jpg")

if os.path.exists(logo_path):
    logo_img = ctk.CTkImage(Image.open(logo_path), size=(70, 70))
    ctk.CTkLabel(center_container, image=logo_img, text="").pack(side="left", padx=15)

ctk.CTkLabel(center_container, text="THE 2-STROKE COLLECTIVE", font=("Impact", 38), text_color=VINTAGE_ORANGE).pack(side="left")

# Chat Area with tag-based styling
chat_area = ctk.CTkTextbox(left_panel, font=("Consolas", 15), border_width=2, border_color=VINTAGE_ORANGE, wrap="word")
chat_area.pack(pady=10, fill="both", expand=True)
chat_area.tag_config("user_tag", foreground=VINTAGE_ORANGE)
chat_area.tag_config("bot_tag", foreground="#3498DB")
chat_area.configure(state="disabled")

input_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
input_frame.pack(pady=20, fill="x")
user_entry = ctk.CTkEntry(input_frame, placeholder_text="Ask about a bike...", height=50, border_color=VINTAGE_ORANGE)
user_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

# Right Sidebar for Image Preview
right_panel = ctk.CTkFrame(main_frame, width=320, fg_color="#2A2A2A", corner_radius=15)
right_panel.pack(side="right", fill="both", padx=20, pady=20)
ctk.CTkLabel(right_panel, text="BIKE PREVIEW", font=("Impact", 20), text_color=VINTAGE_ORANGE).pack(pady=20)
photo_display = ctk.CTkLabel(right_panel, text="Mention a bike\nto see it here!", compound="top")
photo_display.pack(expand=True, padx=20)

def send_message(event=None):
    user_text = user_entry.get().strip()
    if user_text:
        update_preview_image(user_text)
        bot_text = get_response(user_text)
        chat_area.configure(state="normal")
        chat_area.insert("end", "👤 YOU: ", "user_tag")
        chat_area.insert("end", f"{user_text}\n")
        chat_area.insert("end", "🔧 BOT: ", "bot_tag")
        chat_area.insert("end", f"{bot_text}\n\n")
        chat_area.configure(state="disabled")
        chat_area.see("end")
        cursor.execute("INSERT INTO chats (user, bot) VALUES (?, ?)", (user_text, bot_text))
        conn.commit()
        user_entry.delete(0, "end")

ctk.CTkButton(input_frame, text="SEND", fg_color=VINTAGE_ORANGE, command=send_message, width=120, height=50).pack(side="right")
app.bind('<Return>', send_message)
app.mainloop()
