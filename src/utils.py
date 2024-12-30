import os
import cv2
import numpy as np

def create_directory(directory):
    """
    Cria um diretório se ele não existir.

    Args:
        directory (str): Caminho do diretório.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(image_dir, mask_dir, img_size):
    """
    Carrega imagens e máscaras, redimensiona e normaliza os dados.

    Args:
        image_dir (str): Diretório contendo as imagens.
        mask_dir (str): Diretório contendo as máscaras.
        img_size (int): Tamanho das imagens de saída (assume quadradas).

    Returns:
        tuple: Arrays numpy contendo as imagens e as máscaras.
    """
    images, masks = [], []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file in image_files:
        img_name, _ = os.path.splitext(img_file)  # Nome base da imagem
        # Encontrar a máscara correspondente, ignorando a extensão
        mask_file = next((m for m in mask_files if os.path.splitext(m)[0] == img_name), None)

        if not mask_file:
            print(f"Máscara correspondente não encontrada para: {img_file}")
            continue

        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Erro ao carregar imagem ou máscara para: {img_file}")
            continue

        # Redimensionar
        image = cv2.resize(image, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        images.append(image)
        masks.append(mask)

    # Normalizar os valores
    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0
    masks = np.expand_dims(masks, axis=-1)

    print(f"Total de imagens carregadas: {len(images)}")
    print(f"Total de máscaras carregadas: {len(masks)}")

    return np.array(images), np.array(masks)


def normalize_image(image):
    """
    Normaliza os valores de uma imagem para o intervalo [0, 1].

    Args:
        image (numpy.ndarray): Imagem a ser normalizada.

    Returns:
        numpy.ndarray: Imagem normalizada.
    """
    return image / 255.0

def resize_image(image, size):
    """
    Redimensiona a imagem para um tamanho específico.

    Args:
        image (numpy.ndarray): Imagem a ser redimensionada.
        size (tuple): Novo tamanho (largura, altura).

    Returns:
        numpy.ndarray: Imagem redimensionada.
    """
    return cv2.resize(image, size)
