from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential, load_model

app = Flask(__name__)
CORS(app)
lemmatizer = WordNetLemmatizer()

# Cargar archivos con UTF-8
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    # Bolsa de palabras con ceros de Numpy
    bag = np.zeros(len(words))
    for s_word in sentence_words:
        for i, w in enumerate(words):
            if w == s_word:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)
    return classes[np.argmax(res)]

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Lo siento, no entiendo esa pregunta."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        mensaje = data.get('mensaje', '')
        intent = predict_class(mensaje)
        respuesta = get_response(intent)
        return jsonify({"respuesta": respuesta})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"respuesta": "Error en el servidor"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)