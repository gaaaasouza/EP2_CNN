import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import models
from typing import Tuple

class ModelEvaluator:
    """Classe responsável pela avaliação dos modelos"""
    
    @staticmethod
    def evaluate_and_plot(model: models.Sequential, test_data: np.ndarray, test_labels: np.ndarray,
                         title: str = "Matriz de Confusão") -> Tuple[float, np.ndarray]:
        """Avalia modelo e plota matriz de confusão"""
        
        # Fazer predições
        predictions = model.predict(test_data)
        
        # Determinar classes preditas baseado no tipo de modelo
        if predictions.shape[1] == 1:  # Classificação binária
            predicted_labels = (predictions > 0.5).astype(int).flatten()
        else:  # Classificação multiclasse
            predicted_labels = np.argmax(predictions, axis=1)
        
        # Calcular métricas
        accuracy = accuracy_score(test_labels, predicted_labels)
        conf_matrix = confusion_matrix(test_labels, predicted_labels)
        
        # Plotar matriz de confusão
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.show()
        
        return accuracy, predicted_labels