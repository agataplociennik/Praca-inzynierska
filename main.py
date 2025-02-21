# Importowanie niezbędnych bibliotek
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Funkcja do wczytywania danych z pliku CSV
def load_data(file):
    df = pd.read_csv(file)
    return df


# Ścieżka do pliku CSV
csv_path = '/Users/agataplociennik/PycharmProjects/PROJEKT/bazadanych.csv'

# Wczytywanie danych
data = pd.read_csv(csv_path, sep=';')
audio_paths = data['fname'].tolist()  # ścieżki do plików audio
labels = data['label'].tolist()  # etykiety dla plików audio


# Funkcja do wczytywania plików WAV w formacie mono o próbkowaniu 16kHz
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    return tf.squeeze(wav, axis=-1), sample_rate


# Funkcja do przetwarzania spektrogramów
def preprocess_spectrogram(file_path):
    wav, sample_rate = load_wav_16k_mono(file_path)
    desired_length = 48000
    wav_length = tf.size(wav)
    padding_amount = desired_length - wav_length
    # Dopasowywanie długości sygnału
    if padding_amount > 0:
        zero_padding = tf.zeros([padding_amount], dtype=tf.float32)
        wav = tf.concat([wav, zero_padding], 0)
    else:
        wav = wav[:desired_length]
    # Tworzenie spektrogramu
    spect = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spect = tf.abs(spect)
    spect = tf.expand_dims(spect, axis=2)
    return spect


# Przetwarzanie wszystkich ścieżek audio w spektrogramy
spectrogram_data = [preprocess_spectrogram(fp) for fp in audio_paths]

# Podział na zestawy treningowy i walidacyjny
x_train, x_test, y_train, y_test = train_test_split(spectrogram_data, labels, test_size=0.2, random_state=42)

x_train = np.array(x_train)
x_test = np.array(x_test)

# Kodowanie etykiet jako liczby całkowite
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Liczba unikalnych klas
number_of_classes = len(np.unique(y_train_encoded))

# Kodowanie etykiet w formacie "one-hot"
y_train_encoded = tf.one_hot(y_train_encoded, depth=number_of_classes)
y_test_encoded = tf.one_hot(y_test_encoded, depth=number_of_classes)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Budowanie modelu sieci konwolucyjnej
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(x_train.shape[1], x_train.shape[2], 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(number_of_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# Trenowanie modelu
model.fit(x_train, y_train_encoded, epochs=10, batch_size=16, validation_data=(x_test, y_test_encoded))

# Zapis modelu
model.save('sound_model.h5')

# Ewaluacja modelu na zestawie treningowym
loss, accuracy = model.evaluate(x_train, y_train_encoded)
print(f'Dokładność na zestawie testowym: {accuracy * 100:.2f}%')