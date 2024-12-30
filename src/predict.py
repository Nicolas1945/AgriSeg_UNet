from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio

def predict_images(input_path, output_dir, model_path, img_size=256, threshold=0.3):
    """
    Realiza a predição em imagens e salva as máscaras geradas.
    Args:
        input_path (str): Caminho ou diretório contendo as imagens de entrada.
        output_dir (str): Diretório onde as máscaras preditas serão salvas.
        model_path (str): Caminho do modelo U-Net treinado.
        img_size (int): Tamanho das imagens de entrada (assume quadradas).
        threshold (float): Limiar para binarizar a máscara predita.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carregar o modelo treinado sem compilar automaticamente
    model = load_model(model_path, compile=False)

    # Verificar se o input é um diretório ou arquivo único
    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

    for filepath in image_paths:
        try:
            # Carregar e pré-processar a imagem
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Erro ao carregar a imagem: {filepath}")
                continue

            original_size = image.shape[:2]  # Guardar o tamanho original (altura, largura)
            image_resized = cv2.resize(image, (img_size, img_size))
            image_normalized = image_resized / 255.0
            image_input = np.expand_dims(image_normalized, axis=0)  # Adicionar dimensão do lote

            # Fazer a predição
            prediction = model.predict(image_input, verbose=0)
            predicted_mask = (prediction[0, :, :, 0] > threshold).astype(np.uint8)

            # Redimensionar a máscara para o tamanho original da imagem
            predicted_mask_resized = cv2.resize(predicted_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

            # Salvar a máscara predita
            filename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, f"predicted_{filename}")
            cv2.imwrite(output_path, predicted_mask_resized * 255)

            print(f"Predição salva em: {output_path}")
        except Exception as e:
            print(f"Erro ao processar {filepath}: {e}")

def binary_to_georeferenced_shapefile(input_dir, geo_image_path, output_shapefile, threshold=128):
    """
    Converte imagens binarizadas em shapefiles georreferenciados.

    Args:
        input_dir (str): Diretório contendo as imagens binarizadas.
        geo_image_path (str): Caminho para a imagem original georreferenciada.
        output_shapefile (str): Caminho para salvar o shapefile gerado.
        threshold (int): Limiar para considerar pixels como vegetação.
    """
    with rasterio.open(geo_image_path) as src:
        transform = src.transform
        crs = src.crs

    polygons = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
            filepath = os.path.join(input_dir, filename)
            binary_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if binary_image is None:
                continue

            _, binary_image = cv2.threshold(binary_image, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                coords = [(point[0][0], point[0][1]) for point in contour]
                if len(coords) > 2:
                    geo_coords = [
                        rasterio.transform.xy(transform, y, x, offset='center')
                        for x, y in coords
                    ]
                    polygons.append(Polygon(geo_coords))

    if polygons:
        gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)
        gdf.to_file(output_shapefile)
        print(f"Shapefile salvo em: {output_shapefile}")
