import os
import glob
from src.preprocessing import split_images, split_images_and_generate_masks, split_train_val
from src.train import train_unet
from src.predict import predict_images, binary_to_georeferenced_shapefile
from src.config import DIRS, IMG_SIZE, PART_SIZE, THRESHOLD_PREDICTION

def main():
    print("\nBem-vindo ao pipeline de processamento!")
    print("Escolha uma das opções abaixo:")
    print("1 - Recortar imagens em tiles")
    print("2 - Gerar amostras para treinamento (com máscaras binárias e divisão treino/validação)")
    print("3 - Iniciar treinamento do modelo")
    print("4 - Realizar predição")
    print("5 - Gerar shapefile")
    print("0 - Sair")

    while True:
        try:
            choice = int(input("\nDigite a opção desejada: "))
            
            if choice == 1:
                print("\n--- Etapa 1: Recortando imagens em tiles ---")
                split_images(
                    DIRS['raw'],               # Diretório com imagens originais
                    DIRS['processed_images'],  # Diretório de saída para imagens recortadas
                    PART_SIZE                  # Tamanho dos tiles
                )
                print("Recorte concluído!")

            elif choice == 2:
                print("\n--- Etapa 2: Gerando amostras para treinamento (com máscaras binárias) ---")
                split_images_and_generate_masks(
                    DIRS['raw'],               # Diretório com imagens originais
                    DIRS['processed_images'],  # Diretório de saída para imagens recortadas
                    DIRS['processed_masks'],   # Diretório de saída para máscaras recortadas
                    PART_SIZE,                 # Tamanho dos tiles
                    binarize_threshold=128     # Limiar para binarização
                )
                print("Amostras geradas com sucesso!")

                while True:
                    try:
                        train_val_split = float(input("\nDigite a proporção para treino (0.0 a 1.0, ex: 0.8 para 80% treino e 20% validação): "))
                        if 0.0 < train_val_split < 1.0:
                            break
                        else:
                            print("Por favor, insira um valor entre 0.0 e 1.0.")
                    except ValueError:
                        print("Por favor, insira um número válido.")

                print(f"\n--- Dividindo dados: {train_val_split * 100:.0f}% treino e {(1 - train_val_split) * 100:.0f}% validação ---")
                split_train_val(
                    DIRS['processed_images'],  # Diretório de imagens recortadas
                    DIRS['processed_masks'],   # Diretório de máscaras recortadas
                    DIRS['train_images'],      # Diretório de treino (imagens)
                    DIRS['train_masks'],       # Diretório de treino (máscaras)
                    DIRS['val_images'],        # Diretório de validação (imagens)
                    DIRS['val_masks'],         # Diretório de validação (máscaras)
                    train_val_split=train_val_split  # Proporção de treino/validação
                )
                print("Dados divididos entre treino e validação com sucesso!")

            elif choice == 3:
                print("\n--- Etapa 3: Treinando o modelo U-Net ---")

                # Solicitar o número de épocas ao usuário
                while True:
                    try:
                        epochs = int(input("\nDigite o número de épocas para o treinamento: "))
                        if epochs > 0:
                            break
                        else:
                            print("Por favor, insira um número positivo.")
                    except ValueError:
                        print("Por favor, insira um número válido.")

                # Iniciar o treinamento
                train_unet(
                    DIRS['train_images'],      # Diretório com imagens de treino
                    DIRS['train_masks'],       # Diretório com máscaras de treino
                    DIRS['val_images'],        # Diretório com imagens de validação
                    DIRS['val_masks'],         # Diretório com máscaras de validação
                    IMG_SIZE,                  # Tamanho das imagens para treinamento
                    epochs                     # Número de épocas definido pelo usuário
                )

            elif choice == 4:
                print("\n--- Etapa 4: Fazendo predições ---")
                
                # Procurar o modelo mais recente na pasta models
                models_dir = os.path.join("D:/Projetos/bemagro/gihub", "models")
                model_files = glob.glob(os.path.join(models_dir, "*.keras"))
                if not model_files:
                    print("Nenhum modelo encontrado na pasta 'models'. Treine o modelo antes de realizar predições.")
                    continue
                
                # Selecionar o modelo mais recente
                latest_model = max(model_files, key=os.path.getctime)
                print(f"Usando o modelo mais recente: {latest_model}")

                # Realizar predições
                predict_images(
                    DIRS['geo_image'],         # Caminho da imagem base para predição
                    DIRS['pred_output'],       # Diretório de saída para máscaras preditas
                    latest_model,              # Caminho para o modelo mais recente
                    IMG_SIZE,                  # Tamanho das imagens para predição
                    THRESHOLD_PREDICTION       # Limiar para binarizar as predições
                )
                print("Predições concluídas!")

            elif choice == 5:
                print("\n--- Etapa 5: Gerando shapefile georreferenciado ---")
                binary_to_georeferenced_shapefile(
                    DIRS['pred_output'],       # Diretório com máscaras preditas
                    DIRS['geo_image'],         # Imagem base georreferenciada
                    DIRS['shapefile_output']   # Caminho para o shapefile gerado
                )
                print("Shapefile gerado com sucesso!")

            elif choice == 0:
                print("Saindo do programa. Até mais!")
                break

            else:
                print("Opção inválida. Tente novamente.")

        except ValueError:
            print("Por favor, digite um número válido.")

if __name__ == "__main__":
    main()
