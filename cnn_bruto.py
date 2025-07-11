import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras import datasets, layers, models, Input
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_outputs(dir, hiperp, init_w, train_hist, train_err, output_CNN, final_w, pred_class, real_class):
    """Salva todos os arquivos de saída em um diretório especificado."""
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(os.path.join(dir, 'hiperparametros.json'), 'w') as f:
        json.dump(hiperp, f, indent=4)
    
    with open(os.path.join(dir,'pesos_iniciais.txt'), 'w') as f:
        for i, weight_matrix in enumerate(init_w):
            f.write(f"\nPesos da Camada {i+1}:\n")
            if isinstance(weight_matrix, list):
                for m in weight_matrix:
                    np.savetxt(f, m.flatten(), fmt='%f')
            else:
                np.savetxt(f, weight_matrix.flatten(), fmt='%f')

    with open(os.path.join(dir, 'historico_treinamento.json'), 'w') as f:
        json.dump(train_hist, f, indent=4)
    
    np.savetxt(os.path.join(dir, 'erros_treinamento.csv'), train_err, delimiter=',')
    np.savetxt(os.path.join(dir, 'saidas_rede_neural.csv'), output_CNN, delimiter=',')

    with open(os.path.join(dir,'pesos_finais.txt'), 'w') as f:
        for i, weight_matrix in enumerate(final_w):
            f.write(f"\nPesos da Camada {i+1}:\n")
            if isinstance(weight_matrix, list):
                for m in weight_matrix:
                    np.savetxt(f, m.flatten(), fmt='%f')
            else:
                np.savetxt(f, weight_matrix.flatten(), fmt='%f')
    
    with open(os.path.join(dir,'classes_previstas.txt'), 'w') as f:
        for number, real in zip(pred_class, real_class):
            f.write(f"Previu: {number}\tReal: {real}\n")


def CNN_multi(train_images, train_labels, test_images, test_labels, 
              filters=8, dense_units=64, learning_rate=0.001, 
              save_files=True, output_dir='outputs_bruto'):
    """Cria, treina e avalia a CNN para multiclasse."""
    model = models.Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(filters, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    init_weights = model.get_weights()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # --- INÍCIO: INFORMAÇÕES DA ARQUITETURA ---
    if save_files:
        print("\n--- Arquitetura e Hiperparâmetros do Modelo (CNN Bruto Multiclasse) ---")
        print(f"   Parâmetros da Camada Conv2D: Filtros={filters}, Kernel=(3,3), Ativação=relu")
        print(f"   Parâmetros da Camada MaxPooling2D: Pool Size=(2,2)")
        print(f"   Parâmetros da Camada Densa: Unidades={dense_units}, Ativação=relu")
        print(f"   Taxa de Aprendizado (learning_rate): {learning_rate}")
        print("   Resumo do Modelo Keras:")
        model.summary()
        print("--------------------------------------------------------------------")


    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    max_epochs = 50
    verbosity_level = 1 if save_files else 0

    history = model.fit(train_images, train_labels, epochs=max_epochs, 
                        validation_data=(test_images, test_labels),
                        callbacks=[early_stopping],
                        verbose=verbosity_level)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    
    if save_files:
        print(f'\nExecução multiclasse - Acurácia no teste: {test_acc:.4f}')
        actual_epochs = len(history.history['loss'])
        
        hiperparametros = {
            'tipo_modelo': 'CNN Bruto Multiclasse',
            'arquitetura_CNN': {
                'camadas_conv': [
                    {'filters': filters, 'kernel_size': (3, 3), 'activation': 'relu'}
                ],
                'pooling': {'pool_size': (2, 2)},
                'camadas_dense': [
                    {'units': dense_units, 'activation': 'relu'},
                    {'units': 10, 'activation': 'softmax'}
                ]
            },
            'hiperparametros_treino': {
                'learning_rate': learning_rate,
                'loss': 'sparse_categorical_crossentropy',
                'metrics': ['accuracy'],
                'batch_size': 32,  # Se não for especificado no fit(), o padrão é 32
                'epochs': max_epochs,
                'epochs_reais_executadas': actual_epochs,
                'early_stopping_patience': 3
            },
            'acuracia_final': test_acc
        }

        post_train_weights = model.get_weights()
        historico = {key: [float(v) for v in val] for key, val in history.history.items()}
        erros = historico.get('loss', [])
        saidas = model.predict(test_images)
        predicoes = np.argmax(saidas, axis=1)

        print("\n--- Análise de Erros da Rede ---")
        error_indices = np.where(predicoes != test_labels)[0]
        print(f"   Total de erros na saída: {len(error_indices)} de {len(test_labels)} amostras de teste.")
        print(f"   Taxa de acerto: {100 * (1 - len(error_indices) / len(test_labels)):.2f}%")
        # Mostra os 5 primeiros exemplos de erros
        for i, error_idx in enumerate(error_indices[:5]):
            predicted_label = predicoes[error_idx]
            true_label = test_labels[error_idx]
            print(f"     - Exemplo {i+1}: Amostra de teste #{error_idx} | Previsto: {predicted_label}, Correto: {true_label}")
        print("--------------------------------\n")

        save_outputs(output_dir, hiperparametros, init_weights, historico, erros, saidas, post_train_weights, predicoes, test_labels)
        print(f"Arquivos de saída salvos em: '{output_dir}'")
        
        cm = confusion_matrix(test_labels, predicoes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de Confusão - CNN Bruto Multiclasse')
        plt.savefig(os.path.join(output_dir, 'matriz_confusao_multiclasse.png'))
        plt.show()

    return test_acc


def CNN_bin(train_images, train_labels, test_images, test_labels, 
            filters=8, dense_units=64, learning_rate=0.001, 
            save_files=True, output_dir='outputs_bruto_binario'):
    """Cria, treina e avalia a CNN para o caso binário."""
    train_labels_binary = (train_labels >= 5).astype(int)
    test_labels_binary = (test_labels >= 5).astype(int)

    model = models.Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(filters, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    init_weights = model.get_weights()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    if save_files:
        print("\n--- Arquitetura e Hiperparâmetros do Modelo (CNN Bruto Binário) ---")
        print(f"   Parâmetros da Camada Conv2D: Filtros={filters}, Kernel=(3,3), Ativação=relu")
        print(f"   Parâmetros da Camada MaxPooling2D: Pool Size=(2,2)")
        print(f"   Parâmetros da Camada Densa: Unidades={dense_units}, Ativação=relu")
        print(f"   Taxa de Aprendizado (learning_rate): {learning_rate}")
        print("   Resumo do Modelo Keras:")
        model.summary()
        print("------------------------------------------------------------------")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    max_epochs = 50
    verbosity_level = 1 if save_files else 0

    history = model.fit(train_images, train_labels_binary, epochs=max_epochs, 
                        validation_data=(test_images, test_labels_binary),
                        callbacks=[early_stopping], verbose=verbosity_level)

    test_loss, test_acc = model.evaluate(test_images, test_labels_binary, verbose=0)
    
    if save_files:
        print(f'\nExecução binária - Acurácia no teste: {test_acc:.4f}')
        actual_epochs = len(history.history['loss'])

        hiperparametros = {
            'tipo_modelo': 'CNN Bruto Binario',
            'arquitetura_CNN': {
                'camadas_conv': [
                    {'filters': filters, 'kernel_size': (3, 3), 'activation': 'relu'}
                ],
                'pooling': {'pool_size': (2, 2)},
                'camadas_dense': [
                    {'units': dense_units, 'activation': 'relu'},
                    {'units': 1, 'activation': 'sigmoid'}
                ]
            },
            'hiperparametros_treino': {
                'learning_rate': learning_rate,
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy'],
                'batch_size': 32,
                'epochs': max_epochs,
                'epochs_reais_executadas': actual_epochs,
                'early_stopping_patience': 3
            },
            'acuracia_final': test_acc
        }

        post_train_weights = model.get_weights()
        historico = {key: [float(v) for v in val] for key, val in history.history.items()}
        erros = historico.get('loss', [])
        saidas = model.predict(test_images)
        predicoes = (saidas > 0.5).astype(int).flatten()

        print("\n--- Análise de Erros da Rede ---")
        error_indices = np.where(predicoes != test_labels_binary)[0]
        print(f"   Total de erros na saída: {len(error_indices)} de {len(test_labels_binary)} amostras de teste.")
        print(f"   Taxa de acerto: {100 * (1 - len(error_indices) / len(test_labels_binary)):.2f}%")
        for i, error_idx in enumerate(error_indices[:5]):
            predicted_label = predicoes[error_idx]
            true_label = test_labels_binary[error_idx]
            print(f"     - Exemplo {i+1}: Amostra de teste #{error_idx} | Previsto: {predicted_label}, Correto: {true_label}")
        print("--------------------------------\n")

        save_outputs(output_dir, hiperparametros, init_weights, historico, erros, saidas, post_train_weights, predicoes, test_labels_binary)
        print(f"Arquivos de saída salvos em: '{output_dir}'")
        
        cm_bin = confusion_matrix(test_labels_binary, predicoes)
        labels = ['0-4', '5-9']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.title('Matriz de Confusão Binária (0-4 vs 5-9)')
        plt.savefig(os.path.join(output_dir, 'matriz_confusao_binaria.png'))
        plt.show()

    return test_acc


if __name__ == "__main__":
    print("Executando 'cnn_bruto.py'")
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]
    print("Dados preparados!")

    print("\n--- INICIANDO CNN MULTICLASSE ---")
    CNN_multi(train_images, train_labels, test_images, test_labels, save_files=True)

    print("\n--- INICIANDO CNN BINÁRIO ---")
    CNN_bin(train_images, train_labels, test_images, test_labels, save_files=True)