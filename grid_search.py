# grid_search.py
import os
import itertools
import json
import time
from keras import datasets
import tensorflow as tf

# Importa as funções modificadas dos seus scripts
import cnn_bruto
import cnn_hog

def run_grid_search(model_type, script_module, param_grid, train_data, test_data):
    """Executa o grid search para um determinado tipo de modelo e salva um relatório."""
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"--- Iniciando Grid Search para o modelo: {model_type} ---")
    print(f"Total de combinações a serem testadas: {len(combinations)}\n")

    results = []
    best_accuracy = 0
    best_params = None
    start_time = time.time()

    for i, params in enumerate(combinations):
        run_start_time = time.time()
        print(f"Testando combinação {i+1}/{len(combinations)}: {params}")
        
        accuracy = script_module.CNN_multi(train_images, train_labels, test_images, test_labels, save_files=False, **params)
        
        run_end_time = time.time()
        print(f"  -> Acurácia: {accuracy:.4f} (levou {run_end_time - run_start_time:.2f}s)\n")
        results.append({'params': params, 'accuracy': accuracy})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    total_time = time.time() - start_time
    print(f"--- Grid Search para {model_type} concluído em {total_time/60:.2f} minutos ---")
    print(f"Melhor acurácia encontrada: {best_accuracy:.4f}")
    print(f"Com os hiperparâmetros: {best_params}")

    report_filename = f'grid_search_report_{model_type.replace(" ", "_").lower()}.json'
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    report_data = {'best_result': {'params': best_params, 'accuracy': best_accuracy}, 'all_results': sorted_results}
    
    with open(report_filename, 'w') as f:
        json.dump(report_data, f, indent=4)
    print(f"Relatório completo salvo em: {report_filename}\n")
    
    return best_params

if __name__ == "__main__":
    print("Carregando e preparando dados MNIST uma única vez...")
    (train_images_raw, train_labels), (test_images_raw, test_labels) = datasets.mnist.load_data()
    
    # Dados para CNN Bruto
    train_images_bruto = train_images_raw / 255.0
    test_images_bruto = test_images_raw / 255.0
    train_images_bruto = train_images_bruto[..., tf.newaxis]
    test_images_bruto = test_images_bruto[..., tf.newaxis]

    # Dados para Rede Densa com HOG
    train_images_hog = train_images_raw / 255.0
    test_images_hog = test_images_raw / 255.0
    print("Dados prontos.\n")

    # --- Grade de Hiperparâmetros para CNN Bruto ---
    # CUIDADO: Adicionar mais valores resultará em um tempo de execução MUITO LONGO.
    param_grid_bruto = {
        'filters': [16, 32],
        'dense_units': [64, 128],
        'learning_rate': [0.001, 0.0005]
    }

    # --- Grade de Hiperparâmetros para Rede Densa com HOG ---
    param_grid_hog = {
        'dense1_units': [128, 256],
        'dense2_units': [64, 128],
        'learning_rate': [0.001, 0.0005]
    }

    # Executa o Grid Search e obtém os melhores parâmetros
    best_bruto_params = run_grid_search(
        'CNN Bruto', cnn_bruto, param_grid_bruto,
        (train_images_bruto, train_labels),
        (test_images_bruto, test_labels)
    )

    best_hog_params = run_grid_search(
        'CNN HOG', cnn_hog, param_grid_hog,
        (train_images_hog, train_labels),
        (test_images_hog, test_labels)
    )

    print("--- Treinamento final com os melhores parâmetros encontrados ---")

    # Roda uma última vez o modelo Bruto com os melhores parâmetros e salva os arquivos
    if best_bruto_params:
        print("\nTreinando o melhor modelo CNN Bruto e salvando os arquivos...")
        cnn_bruto.CNN_multi(
            train_images_bruto, train_labels, test_images_bruto, test_labels,
            save_files=True, 
            output_dir='outputs_bruto_best', 
            **best_bruto_params
        )

    # Roda uma última vez o modelo HOG com os melhores parâmetros e salva os arquivos
    if best_hog_params:
        print("\nTreinando o melhor modelo CNN HOG e salvando os arquivos...")
        cnn_hog.CNN_multi(
            train_images_hog, train_labels, test_images_hog, test_labels,
            save_files=True, 
            output_dir='outputs_hog_best', 
            **best_hog_params
        )

    print("\nProcesso de otimização e treinamento final concluído!")