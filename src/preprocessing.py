import os
import cv2
import numpy as np
import random
import shutil  # Certifique-se de que esta linha está presente

def create_directory(directory):
    """
    Cria um diretório se ele não existir.

    Args:
        directory (str): Caminho do diretório.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_images(image_dir, output_dir, tile_size):
    """
    Recorta imagens em tiles de tamanho especificado.

    Args:
        image_dir (str): Diretório com as imagens originais.
        output_dir (str): Diretório de saída para os tiles.
        tile_size (int): Tamanho dos tiles (altura e largura).
    """
    create_directory(output_dir)
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.png', '.tif', '.tiff')):
            filepath = os.path.join(image_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"Erro ao carregar a imagem: {filename}")
                continue

            h, w, _ = image.shape
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    tile = image[y:y+tile_size, x:x+tile_size]
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        # Preencher o tile com zeros (preto)
                        tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        tile_padded[:tile.shape[0], :tile.shape[1], :] = tile
                        tile = tile_padded

                    tile_name = f"{os.path.splitext(filename)[0]}_{x}_{y}.jpg"
                    cv2.imwrite(os.path.join(output_dir, tile_name), tile)
    print(f"Imagens recortadas com tiles de {tile_size}x{tile_size}!")

def split_images_and_generate_masks(image_dir, output_image_dir, output_mask_dir, tile_size, binarize_threshold=128):
    """
    Recorta imagens e gera máscaras binárias automaticamente.

    Args:
        image_dir (str): Diretório com as imagens originais.
        output_image_dir (str): Diretório de saída para os tiles das imagens.
        output_mask_dir (str): Diretório de saída para os tiles das máscaras.
        tile_size (int): Tamanho dos tiles (altura e largura).
        binarize_threshold (int): Valor de limiar para binarização (simula as máscaras).
    """
    create_directory(output_image_dir)
    create_directory(output_mask_dir)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.png', '.tif', '.tiff')):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Erro ao carregar a imagem: {filename}")
                continue

            h, w, _ = image.shape

            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    # Recorte da imagem
                    image_tile = image[y:y+tile_size, x:x+tile_size]

                    # Garantir que o tile tenha o tamanho correto
                    if image_tile.shape[0] < tile_size or image_tile.shape[1] < tile_size:
                        image_tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        image_tile_padded[:image_tile.shape[0], :image_tile.shape[1], :] = image_tile
                        image_tile = image_tile_padded

                    # Gerar a máscara binária baseada em um canal ou valor médio
                    gray_tile = cv2.cvtColor(image_tile, cv2.COLOR_BGR2GRAY)
                    _, mask_tile = cv2.threshold(gray_tile, binarize_threshold, 255, cv2.THRESH_BINARY)

                    # Salvar os tiles de imagem e máscara
                    image_tile_name = f"{os.path.splitext(filename)[0]}_{x}_{y}.jpg"
                    mask_tile_name = f"{os.path.splitext(filename)[0]}_{x}_{y}.png"

                    cv2.imwrite(os.path.join(output_image_dir, image_tile_name), image_tile)
                    cv2.imwrite(os.path.join(output_mask_dir, mask_tile_name), mask_tile)

    print(f"Imagens e máscaras recortadas com tiles de {tile_size}x{tile_size}!")

def split_train_val(source_image_dir, source_mask_dir, train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, train_val_split=0.8):
    """
    Divide os dados em conjuntos de treinamento e validação.

    Args:
        source_image_dir (str): Diretório contendo as imagens originais recortadas.
        source_mask_dir (str): Diretório contendo as máscaras originais recortadas.
        train_image_dir (str): Diretório para salvar as imagens de treino.
        train_mask_dir (str): Diretório para salvar as máscaras de treino.
        val_image_dir (str): Diretório para salvar as imagens de validação.
        val_mask_dir (str): Diretório para salvar as máscaras de validação.
        train_val_split (float): Proporção de dados para treino (ex: 0.8 = 80% treino, 20% validação).
    """
    create_directory(train_image_dir)
    create_directory(train_mask_dir)
    create_directory(val_image_dir)
    create_directory(val_mask_dir)

    # Listar arquivos de imagens e máscaras
    image_files = os.listdir(source_image_dir)
    mask_files = os.listdir(source_mask_dir)

    # Garantir correspondência entre imagens e máscaras
    paired_files = [(img, mask) for img, mask in zip(image_files, mask_files) if img.split('.')[0] == mask.split('.')[0]]

    # Embaralhar e dividir
    random.shuffle(paired_files)
    split_index = int(len(paired_files) * train_val_split)
    train_files = paired_files[:split_index]
    val_files = paired_files[split_index:]

    # Mover arquivos para os diretórios de treino e validação
    for img, mask in train_files:
        shutil.move(os.path.join(source_image_dir, img), os.path.join(train_image_dir, img))
        shutil.move(os.path.join(source_mask_dir, mask), os.path.join(train_mask_dir, mask))

    for img, mask in val_files:
        shutil.move(os.path.join(source_image_dir, img), os.path.join(val_image_dir, img))
        shutil.move(os.path.join(source_mask_dir, mask), os.path.join(val_mask_dir, mask))

    print(f"Dados divididos com sucesso! {len(train_files)} para treino, {len(val_files)} para validação.")
