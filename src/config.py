# Configurações do projeto

# Tamanho das imagens e parâmetros gerais
IMG_SIZE = 256  # Tamanho para redimensionar imagens (quadrado)
PART_SIZE = 256  # Tamanho dos tiles
THRESHOLD_PREDICTION = 0.3  # Limiar para binarização na predição
BATCH_SIZE = 16  # Tamanho do lote para treinamento
EPOCHS = 100  # Número de épocas para treinamento

# Diretórios do projeto
DIRS = {
    # Diretórios de dados brutos e processados
    'raw': 'D:/Projetos/bemagro/gihub/data/raw',                       # Imagens originais
    'raw_masks': 'D:/Projetos/bemagro/gihub/data/raw/mask',            # Máscaras originais
    'processed_images': 'D:/Projetos/bemagro/gihub/data/processed/image',  # Tiles de imagens
    'processed_masks': 'D:/Projetos/bemagro/gihub/data/processed/mask',   # Tiles de máscaras

    # Diretórios para treinamento e validação
    'train_images': 'D:/Projetos/bemagro/gihub/data/train/image',  # Imagens de treino
    'train_masks': 'D:/Projetos/bemagro/gihub/data/train/mask',    # Máscaras de treino
    'val_images': 'D:/Projetos/bemagro/gihub/data/val/image',      # Imagens de validação
    'val_masks': 'D:/Projetos/bemagro/gihub/data/val/mask',        # Máscaras de validação

    # Diretórios para predição
    'pred_input': 'D:/Projetos/bemagro/gihub/data/predict/input',      # Entrada para predições
    'pred_output': 'D:/Projetos/bemagro/gihub/data/predict/output',    # Saída das predições

    # Diretórios georreferenciados
    'geo_image': 'D:/Projetos/bemagro/gihub/data/predict/geo/Orthomosaico_roi.tif',  # Imagem base para predições georreferenciadas
    'shapefile_output': 'D:/Projetos/bemagro/gihub/data/shapefile/vegetacao.shp',   # Caminho do shapefile gerado
}
