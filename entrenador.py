import json
import numpy as np
import random
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD

# Recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# 1. Cargar dataset con UTF-8
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# 2. Procesar datos
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenizar
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        # Guardar (palabras, etiqueta)
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar y limpiar vocabulario
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Guardar vocabularios
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# 3. Preparar datos de entrenamiento
training = []
output_empty = np.zeros(len(classes))

for patron_palabras, etiqueta in documents:
    bag = []

    pattern_words_lem = [lemmatizer.lemmatize(w.lower()) for w in patron_palabras]

    # Bolsa de palabras
    for w in words:
        bag.append(1 if w in pattern_words_lem else 0)

    # Salida (one-hot)
    output_row = list(output_empty)
    output_row[classes.index(etiqueta)] = 1

    training.append([bag, output_row])

# Mezclar datos
random.shuffle(training)
training = np.array(training, dtype=object)

# Separar X y Y
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# 🔥 IMPORTANTE: dimensión correcta de entrada
input_size = len(train_x[0])
output_size = len(classes)

# 4. Crear modelo
model = Sequential([
    Input(shape=(input_size,)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(output_size, activation='softmax')
])

# Compilar modelo
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("\n--- Iniciando Entrenamiento ---")

# Entrenar modelo
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Guardar modelo
model.save('chatbot_model.h5')

print("\n✅ Modelo generado con éxito.")