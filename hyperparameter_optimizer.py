import itertools
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

class HyperparameterOptimizer:
    """Classe para otimização de hiperparâmetros"""
    
    def __init__(self, data_manager, file_manager, evaluator):
        self.data_manager = data_manager
        self.file_manager = file_manager
        self.evaluator = evaluator
        self.results = []
    
    def define_parameter_grid(self, model_type: str) -> Dict[str, List]:
        """Define grade de parâmetros para busca"""
        
        if model_type == 'cnn_raw':
            return {
                'conv_filters': [16, 32, 64],
                'dense_units': [32, 64, 128],
                'epochs': [3, 5, 10],
                'learning_rate': [0.001, 0.01]
            }
        elif model_type == 'cnn_hog':
            return {
                'dense_units_1': [64, 128, 256],
                'dense_units_2': [32, 64, 128],
                'epochs': [3, 5, 10],
                'learning_rate': [0.001, 0.01]
            }
    
    def grid_search_cnn_raw(self, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Executa busca em grade para CNN Raw"""
        
        best_score = 0
        best_params = None
        all_results = []
        
        print(f"Iniciando Grid Search com {len(list(ParameterGrid(param_grid)))} combinações...")
        
        for i, params in enumerate(ParameterGrid(param_grid)):
            print(f"Testando combinação {i+1}: {params}")
            
            try:
                # Criar classificador modificado
                from cnn_raw_classifier_optimized import CNNRawClassifierOptimized
                classifier = CNNRawClassifierOptimized(
                    self.data_manager, self.file_manager, self.evaluator
                )
                
                # Treinar com parâmetros específicos
                score = classifier.train_and_evaluate_with_params(params)
                
                result = {
                    'params': params,
                    'score': score
                }
                
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                print(f"Score: {score:.4f}")
                
            except Exception as e:
                print(f"Erro na combinação {params}: {e}")
                continue
        
        self.results = all_results
        return {'best_params': best_params, 'best_score': best_score, 'all_results': all_results}
    
    def grid_search_cnn_hog(self, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Executa busca em grade para CNN HOG"""
        
        best_score = 0
        best_params = None
        all_results = []
        
        print(f"Iniciando Grid Search com {len(list(ParameterGrid(param_grid)))} combinações...")
        
        for i, params in enumerate(ParameterGrid(param_grid)):
            print(f"Testando combinação {i+1}: {params}")
            
            try:
                # Criar classificador modificado
                from cnn_hog_classifier_optimized import CNNHOGClassifierOptimized
                classifier = CNNHOGClassifierOptimized(
                    self.data_manager, self.file_manager, self.evaluator
                )
                
                # Treinar com parâmetros específicos
                score = classifier.train_and_evaluate_with_params(params)
                
                result = {
                    'params': params,
                    'score': score
                }
                
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                print(f"Score: {score:.4f}")
                
            except Exception as e:
                print(f"Erro na combinação {params}: {e}")
                continue
        
        self.results = all_results
        return {'best_params': best_params, 'best_score': best_score, 'all_results': all_results}
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Salva resultados da otimização"""
        
        # Converter numpy types para tipos serializáveis
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy_types(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados salvos em {filename}")
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """Plota resultados da otimização"""
        
        all_results = results['all_results']
        scores = [r['score'] for r in all_results]
        
        plt.figure(figsize=(12, 8))
        
        # Gráfico de barras dos scores
        plt.subplot(2, 1, 1)
        plt.bar(range(len(scores)), scores)
        plt.title('Scores por Combinação de Parâmetros')
        plt.xlabel('Combinação')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Destacar melhor resultado
        best_idx = scores.index(max(scores))
        plt.bar(best_idx, scores[best_idx], color='red', label='Melhor')
        plt.legend()
        
        # Histograma dos scores
        plt.subplot(2, 1, 2)
        plt.hist(scores, bins=10, alpha=0.7, edgecolor='black')
        plt.title('Distribuição dos Scores')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()