# cnn_hog.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras import datasets, layers, models, Input
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


def extract_hog_features(images):
    """Extrai as características HOG das imagens."""
    hog_features = [hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False) for image in images]
    return np.array(hog_features)


def CNN_multi(train_images, train_labels, test_images, test_labels, 
              dense1_units=128, dense2_units=64, learning_rate=0.001, 
              save_files=True, output_dir='outputs_hog'):
    """Cria, treina e avalia a Rede Densa com HOG para multiclasse."""
    if save_files: print("Extraindo características HOG...")
    train_hog_features = extract_hog_features(train_images)
    test_hog_features = extract_hog_features(test_images)

    scaler = StandardScaler()
    train_hog_features = scaler.fit_transform(train_hog_features)
    test_hog_features = scaler.transform(test_hog_features)
    
    X_train, X_val, y_train, y_val = train_test_split(train_hog_features, train_labels, test_size=0.2, random_state=42)

    model = models.Sequential([
        Input(shape=(train_hog_features.shape[1],)),
        layers.Dense(dense1_units, activation='relu'),
        layers.Dense(dense2_units, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    init_weights = model.get_weights()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    max_epochs = 50
    verbosity_level = 1 if save_files else 0

    history = model.fit(X_train, y_train, epochs=max_epochs, 
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping], verbose=verbosity_level)

    test_loss, test_acc = model.evaluate(test_hog_features, test_labels, verbose=0)
    
    if save_files:
        print(f'\nExecução multiclasse com HOG - Acurácia no teste: {test_acc:.4f}')
        actual_epochs = len(history.history['loss'])
        hiperparametros = {
            'tipo_modelo': 'Rede Densa HOG Multiclasse', 'dense1_units': dense1_units, 'dense2_units': dense2_units,
            'learning_rate': learning_rate, 'epochs_reais_executadas': actual_epochs, 'acuracia_final': test_acc
        }
        post_train_weights = model.get_weights()
        historico = {key: [float(v) for v in val] for key, val in history.history.items()}
        erros = historico.get('loss', [])
        saidas = model.predict(test_hog_features)
        predicoes = np.argmax(saidas, axis=1)

        save_outputs(output_dir, hiperparametros, init_weights, historico, erros, saidas, post_train_weights, predicoes, test_labels)
        print(f"Arquivos de saída salvos em: '{output_dir}'")
        
        cm = confusion_matrix(test_labels, predicoes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de Confusão - Rede Densa com HOG')
        plt.savefig(os.path.join(output_dir, 'matriz_confusao_hog_multiclasse.png'))
        plt.show()

    return test_acc


def CNN_bin(train_images, train_labels, test_images, test_labels, 
            dense1_units=128, dense2_units=64, learning_rate=0.001, 
            save_files=True, output_dir='outputs_hog_binario'):
    """Cria, treina e avalia a Rede Densa com HOG para o caso binário."""
    if save_files: print("Extraindo características HOG para o modelo binário...")
    train_hog_features = extract_hog_features(train_images)
    test_hog_features = extract_hog_features(test_images)

    scaler = StandardScaler()
    train_hog_features = scaler.fit_transform(train_hog_features)
    test_hog_features = scaler.transform(test_hog_features)
    
    train_labels_binary = (train_labels >= 5).astype(int)
    test_labels_binary = (test_labels >= 5).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(train_hog_features, train_labels_binary, test_size=0.2, random_state=42)

    model = models.Sequential([
        Input(shape=(train_hog_features.shape[1],)),
        layers.Dense(dense1_units, activation='relu'),
        layers.Dense(dense2_units, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    init_weights = model.get_weights()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    max_epochs = 50
    verbosity_level = 1 if save_files else 0

    history = model.fit(X_train, y_train, epochs=max_epochs, 
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping], verbose=verbosity_level)

    test_loss, test_acc = model.evaluate(test_hog_features, test_labels_binary, verbose=0)
    
    if save_files:
        print(f'\nExecução binária com HOG - Acurácia no teste: {test_acc:.4f}')
        actual_epochs = len(history.history['loss'])
        hiperparametros = {
            'tipo_modelo': 'Rede Densa HOG Binária', 'dense1_units': dense1_units, 'dense2_units': dense2_units,
            'learning_rate': learning_rate, 'epochs_reais_executadas': actual_epochs, 'acuracia_final': test_acc
        }
        post_train_weights = model.get_weights()
        historico = {key: [float(v) for v in val] for key, val in history.history.items()}
        erros = historico.get('loss', [])
        saidas = model.predict(test_hog_features)
        predicoes = (saidas > 0.5).astype(int).flatten()

        save_outputs(output_dir, hiperparametros, init_weights, historico, erros, saidas, post_train_weights, predicoes, test_labels_binary)
        print(f"Arquivos de saída salvos em: '{output_dir}'")
        
        cm_bin = confusion_matrix(test_labels_binary, predicoes)
        labels = ['0-4', '5-9']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.title('Matriz de Confusão Binária com HOG (0-4 vs 5-9)')
        plt.savefig(os.path.join(output_dir, 'matriz_confusao_hog_binaria.png'))
        plt.show()

    return test_acc


if __name__ == "__main__":
    print("Executando 'cnn_hog.py' de forma independente...")
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    print("Dados preparados!")

    print("\n--- INICIANDO REDE DENSA COM HOG (MULTICLASSE) ---")
    CNN_multi(train_images, train_labels, test_images, test_labels, save_files=True)

    print("\n--- INICIANDO REDE DENSA COM HOG (BINÁRIO) ---")
    CNN_bin(train_images, train_labels, test_images, test_labels, save_files=True)