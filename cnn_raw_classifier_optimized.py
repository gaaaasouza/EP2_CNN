import numpy as np
from typing import Tuple, Dict, Any
from keras import models, layers, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from cnn_raw_classifier import CNNRawClassifier

class CNNRawClassifierOptimized(CNNRawClassifier):
    """Versão otimizada do CNNRawClassifier para grid search"""
    
    def build_model_with_params(self, params: Dict[str, Any], input_shape: Tuple[int, ...], num_classes: int):
        """Constrói modelo com parâmetros específicos"""
        
        conv_filters = params.get('conv_filters', 32)
        dense_units = params.get('dense_units', 64)
        
        if num_classes == 1:
            activation = 'sigmoid'
        else:
            activation = 'softmax'
        
        self.model = models.Sequential([
            Input(shape=input_shape),
            layers.Conv2D(conv_filters, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(dense_units, activation='relu'),
            layers.Dense(num_classes, activation=activation)
        ])
        
        return self.model
    
    def compile_model_with_params(self, params: Dict[str, Any], loss: str):
        """Compila modelo com parâmetros específicos"""
        
        learning_rate = params.get('learning_rate', 0.001)
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    def train_and_evaluate_with_params(self, params: Dict[str, Any]) -> float:
        """Treina e avalia modelo com parâmetros específicos"""
        
        # Dividir dados de treino para validação
        X_train, X_val, y_train, y_val = train_test_split(
            self.data_manager.train_images, 
            self.data_manager.train_labels, 
            test_size=0.2, 
            random_state=42,
            stratify=self.data_manager.train_labels
        )
        
        # Construir modelo
        self.build_model_with_params(params, (28, 28, 1), 10)
        self.compile_model_with_params(params, 'sparse_categorical_crossentropy')
        
        # Treinar modelo
        epochs = params.get('epochs', 5)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=0  # Silencioso para não poluir output
        )
        
        # Retornar melhor accuracy de validação
        return max(history.history['val_accuracy'])