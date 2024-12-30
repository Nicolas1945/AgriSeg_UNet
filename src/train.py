import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
from src.utils import load_data
from src.config import BATCH_SIZE

import os
from datetime import datetime

def train_unet(image_dir_train, mask_dir_train, image_dir_val, mask_dir_val, img_size, epochs):
    print("\n--- Carregando os dados de treino e validação ---")
    images_train, masks_train = load_data(image_dir_train, mask_dir_train, img_size)
    images_val, masks_val = load_data(image_dir_val, mask_dir_val, img_size)

    print("\n--- Criando o modelo U-Net ---")
    model = unet_model((img_size, img_size, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    print("\n--- Iniciando o treinamento ---")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        model.fit(
            images_train, masks_train,
            validation_data=(images_val, masks_val),
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=1,
            callbacks=callbacks
        )

    print("\n--- Salvando o modelo treinado ---")

    # Diretório para salvar os modelos
    models_dir = os.path.join("D:\\Projetos\\bemagro\\gihub", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"unet_model_{timestamp}.keras")

    # Salvar no formato recomendado pelo Keras
    model.save(model_path)
    print(f"Modelo treinado e salvo em: {model_path}")


def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Bottleneck
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Decoder
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(p2)
    u1 = layers.concatenate([u1, c2], axis=-1)  # Combina com a camada anterior do encoder
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u2 = layers.concatenate([u2, c1], axis=-1)  # Combina com a camada inicial do encoder
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)

    return models.Model(inputs, outputs)
