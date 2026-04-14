import tkinter as tk
from tkinter import scrolledtext
import json
import pickle
import numpy as np
import nltk
import random
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# --- LOGICA DEL MODELO (Igual a la tuya) ---
lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json', encoding='utf-8'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x, reverse=True)
    return classes[results]

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "No entiendo la pregunta."

# --- INTERFAZ GRÁFICA (La Aplicación) ---
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Asistente de Inventario IA")
        self.root.geometry("450x600")
        self.root.resizable(width=False, height=False)

        # Colores
        self.bg_color = "#2C3E50"
        self.text_color = "#ECF0F1"
        self.bot_msg_color = "#34495E"
        self.user_msg_color = "#1ABC9C"

        # Título
        self.label = tk.Label(root, text="Chatbot Inteligente", font=("Arial", 14, "bold"), bg=self.bg_color, fg=self.text_color)
        self.label.pack(pady=10)

        # Área de chat
        self.chat_window = scrolledtext.ScrolledText(root, bd=0, bg=self.bg_color, fg=self.text_color, font=("Arial", 11), state=tk.DISABLED)
        self.chat_window.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Campo de entrada de texto
        self.entry_box = tk.Entry(root, bd=0, bg="#FFFFFF", fg="#333333", font=("Arial", 12))
        self.entry_box.pack(padx=10, pady=10, fill=tk.X)
        self.entry_box.bind("<Return>", self.send) # Enviar con la tecla Enter

        # Botón enviar
        self.send_button = tk.Button(root, text="Enviar Mensaje", font=("Arial", 10, "bold"), bg=self.user_msg_color, fg="white", command=self.send)
        self.send_button.pack(pady=5)

        self.insert_bot_msg("Bot: ¡Hola! Soy tu asistente. ¿Cómo puedo ayudarte hoy?")

    def send(self, event=None):
        message = self.entry_box.get().strip()
        if message:
            self.entry_box.delete(0, tk.END)
            self.insert_user_msg(f"Tú: {message}")
            
            # Procesar respuesta
            tag = predict_class(message)
            response = get_response(tag)
            
            self.insert_bot_msg(f"Bot: {response}")

    def insert_user_msg(self, msg):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, msg + "\n\n")
        self.chat_window.config(state=tk.DISABLED)
        self.chat_window.yview(tk.END)

    def insert_bot_msg(self, msg):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, msg + "\n\n")
        self.chat_window.config(state=tk.DISABLED)
        self.chat_window.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()