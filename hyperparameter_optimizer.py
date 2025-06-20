import itertools
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

from cnn_raw_classifier import CNNRawClassifier
from cnn_hog_classifier import CNNHOGClassifier

class HyperparameterOptimizer:
    """Classe para otimização de hiperparâmetros com Grid Search e Random Search."""
    
    def __init__(self, data_manager, file_manager, evaluator):
        self.data_manager = data_manager
        self.file_manager = file_manager
        self.evaluator = evaluator
        self.results = []
    
    def define_parameter_grid(self, model_type: str) -> Dict[str, List]:
        """Define grade de parâmetros para busca (usado por Grid Search e como base para Random Search)."""
        
        if model_type == 'cnn_raw':
            return {
                'conv_filters': [16, 32, 64],
                'dense_units': [64, 128, 256],
                'dropout_rate': [0.3, 0.4, 0.5],
                'epochs': [5, 10, 15],
                'learning_rate': [0.001, 0.0005, 0.01],
                'batch_size': [32, 64, 128]
            }
        elif model_type == 'cnn_hog':
            return {
                'dense_units_1': [128, 256, 512],
                'dense_units_2': [64, 128, 256],
                'dropout_rate': [0.3, 0.4, 0.5],
                'epochs': [5, 10, 15],
                'learning_rate': [0.001, 0.0005, 0.01],
                'batch_size': [32, 64, 128]
            }
        # Para referência:
        # Número de combinações para CNN Raw: 3 * 3 * 3 * 3 * 3 * 3 = 729 combinações (se todos forem listados)
        # O exemplo do CNN.py tinha 8 filtros e 64 densos, com dropout 0.5, epochs 50, batch 64/128
        # Ajustei os valores para serem razoáveis para Grid Search ainda.
    
    def grid_search_cnn_raw(self, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Executa busca em grade para CNN Raw."""
        
        best_score = 0
        best_params = None
        all_results = []
        
        total_combinations = len(list(ParameterGrid(param_grid)))
        print(f"Iniciando Grid Search com {total_combinations} combinações...")
        
        for i, params in enumerate(ParameterGrid(param_grid)):
            print(f"Testando combinação {i+1}/{total_combinations}: {params}")
            
            try:
                classifier = CNNRawClassifier(
                    self.data_manager, self.file_manager, self.evaluator
                )
                
                score = classifier.train_and_evaluate_for_optimization(params)
                
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
                all_results.append({'params': params, 'score': -1, 'error': str(e)})
                continue
        
        self.results = all_results
        return {'best_params': best_params, 'best_score': best_score, 'all_results': all_results}
    
    def grid_search_cnn_hog(self, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Executa busca em grade para CNN HOG."""
        
        best_score = 0
        best_params = None
        all_results = []
        
        total_combinations = len(list(ParameterGrid(param_grid)))
        print(f"Iniciando Grid Search com {total_combinations} combinações...")
        
        for i, params in enumerate(ParameterGrid(param_grid)):
            print(f"Testando combinação {i+1}/{total_combinations}: {params}")
            
            try:
                classifier = CNNHOGClassifier(
                    self.data_manager, self.file_manager, self.evaluator
                )
                
                score = classifier.train_and_evaluate_for_optimization(params)
                
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
                all_results.append({'params': params, 'score': -1, 'error': str(e)})
                continue
        
        self.results = all_results
        return {'best_params': best_params, 'best_score': best_score, 'all_results': all_results}

    def random_search_cnn_raw(self, param_distributions: Dict[str, List], n_iter: int) -> Dict[str, Any]:
        """Executa busca aleatória para CNN Raw."""
        
        best_score = 0
        best_params = None
        all_results = []
        
        print(f"Iniciando Random Search com {n_iter} iterações...")
        
        random_params_iterator = ParameterSampler(param_distributions, n_iter=n_iter, random_state=42)
        
        for i, params in enumerate(random_params_iterator):
            print(f"Testando iteração {i+1}/{n_iter}: {params}")
            
            try:
                classifier = CNNRawClassifier(
                    self.data_manager, self.file_manager, self.evaluator
                )
                score = classifier.train_and_evaluate_for_optimization(params)
                
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
                print(f"Erro na iteração {params}: {e}")
                all_results.append({'params': params, 'score': -1, 'error': str(e)})
                continue
        
        self.results = all_results
        return {'best_params': best_params, 'best_score': best_score, 'all_results': all_results}

    def random_search_cnn_hog(self, param_distributions: Dict[str, List], n_iter: int) -> Dict[str, Any]:
        """Executa busca aleatória para CNN HOG."""
        
        best_score = 0
        best_params = None
        all_results = []
        
        print(f"Iniciando Random Search com {n_iter} iterações...")
        
        random_params_iterator = ParameterSampler(param_distributions, n_iter=n_iter, random_state=42)
        
        for i, params in enumerate(random_params_iterator):
            print(f"Testando iteração {i+1}/{n_iter}: {params}")
            
            try:
                classifier = CNNHOGClassifier(
                    self.data_manager, self.file_manager, self.evaluator
                )
                score = classifier.train_and_evaluate_for_optimization(params)
                
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
                print(f"Erro na iteração {params}: {e}")
                all_results.append({'params': params, 'score': -1, 'error': str(e)})
                continue
        
        self.results = all_results
        return {'best_params': best_params, 'best_score': best_score, 'all_results': all_results}
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Salva resultados da otimização."""
        
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
        """Plota resultados da otimização."""
        
        all_results = results['all_results']
        scores = [r['score'] for r in all_results if r['score'] != -1] # Filtrar falhas
        
        if not scores:
            print("Não há scores válidos para plotar.")
            return

        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.bar(range(len(scores)), scores)
        plt.title('Scores por Combinação de Parâmetros')
        plt.xlabel('Combinação')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        best_idx = np.argmax(scores) # Usa numpy para encontrar o índice do melhor score
        plt.bar(best_idx, scores[best_idx], color='red', label='Melhor')
        plt.legend()
        
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