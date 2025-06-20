import os
import sys
from experiment_runner import ExperimentRunner
from data_manager import DataManager
from file_manager import FileManager
from model_evaluator import ModelEvaluator
from cnn_raw_classifier import CNNRawClassifier
from cnn_hog_classifier import CNNHOGClassifier
from hyperparameter_optimizer import HyperparameterOptimizer

class MenuInterface:
    """Interface de menu para seleção de algoritmos - compatível com Windows"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.file_manager = FileManager()
        self.evaluator = ModelEvaluator()
        self.data_prepared = False
    
    def clear_screen(self):
        """Limpa a tela do console"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def prepare_data_if_needed(self):
        """Prepara os dados apenas uma vez"""
        if not self.data_prepared:
            print("Carregando e preparando dados MNIST...")
            self.data_manager.load_mnist_data()
            self.data_manager.normalize_data()
            self.data_manager.add_channel_dimension()
            self.data_prepared = True
            print("Dados preparados com sucesso!")
    
    def show_main_menu(self):
        """Exibe o menu principal"""
        self.clear_screen()
        print("=" * 60)
        print("        SISTEMA DE CLASSIFICACAO MNIST")
        print("=" * 60)
        print()
        print("Selecione o algoritmo que deseja executar:")
        print()
        print("1. CNN com Dados Brutos - Multiclasse")
        print("2. CNN com Dados Brutos - Binaria (0 vs 1)")
        print("3. CNN com HOG - Multiclasse")
        print("4. CNN com HOG - Binaria (0 vs 1)")
        print("5. Executar Todos os Experimentos")
        print("6. Informacoes sobre os Algoritmos")
        print("7. Otimizar Hiperparametros CNN Raw")
        print("8. Otimizar Hiperparametros CNN HOG")
        print("0. Sair")
        print()
        print("=" * 60)
        
        while True:
            try:
                choice = input("Digite sua opcao (0-8): ").strip()
                if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                    return int(choice)
                else:
                    print("Opcao invalida! Digite um numero entre 0 e 8.")
            except (ValueError, KeyboardInterrupt):
                print("Entrada invalida! Digite um numero entre 0 e 8.")
    
    def run_cnn_raw_multiclass(self):
        """Executa CNN com dados brutos - multiclasse"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN com Dados Brutos - Multiclasse")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        cnn_raw = CNNRawClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_raw.run_multiclass()
        
        input("\nExperimento concluido! Pressione Enter para continuar...")
    
    def run_cnn_raw_binary(self):
        """Executa CNN com dados brutos - binária"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN com Dados Brutos - Binaria (0 vs 1)")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        cnn_raw = CNNRawClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_raw.run_binary()
        
        input("\nExperimento concluido! Pressione Enter para continuar...")
    
    def run_cnn_hog_multiclass(self):
        """Executa CNN com HOG - multiclasse"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN com HOG - Multiclasse")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        cnn_hog = CNNHOGClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_hog.run_multiclass()
        
        input("\nExperimento concluido! Pressione Enter para continuar...")
    
    def run_cnn_hog_binary(self):
        """Executa CNN com HOG - binária"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN com HOG - Binaria (0 vs 1)")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        cnn_hog = CNNHOGClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_hog.run_binary()
        
        input("\nExperimento concluido! Pressione Enter para continuar...")
    
    def run_all_experiments(self):
        """Executa todos os experimentos"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: Todos os Experimentos")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        # Executar experimentos CNN bruta
        cnn_raw = CNNRawClassifier(self.data_manager, self.file_manager, self.evaluator)
        print("\nExecutando CNN Bruta - Multiclasse...")
        cnn_raw.run_multiclass()
        print("\nExecutando CNN Bruta - Binaria...")
        cnn_raw.run_binary()
        
        # Executar experimentos CNN HOG
        cnn_hog = CNNHOGClassifier(self.data_manager, self.file_manager, self.evaluator)
        print("\nExecutando CNN HOG - Multiclasse...")
        cnn_hog.run_multiclass()
        print("\nExecutando CNN HOG - Binaria...")
        cnn_hog.run_binary()
        
        input("\nTodos os experimentos concluidos! Pressione Enter para continuar...")
    
    def show_info(self):
        """Mostra informações sobre os algoritmos"""
        self.clear_screen()
        print("=" * 60)
        print("        INFORMACOES SOBRE OS ALGORITMOS")
        print("=" * 60)
        print()
        print("CNN com Dados Brutos:")
        print("  - Usa imagens MNIST originais (28x28 pixels)")
        print("  - Arquitetura: Conv2D -> MaxPooling -> Flatten -> Dense")
        print("  - Multiclasse: Classifica digitos 0-9")
        print("  - Binaria: Classifica apenas digitos 0 vs 1")
        print()
        print("CNN com HOG (Histogram of Oriented Gradients):")
        print("  - Extrai caracteristicas HOG das imagens")
        print("  - Arquitetura: Dense -> Dense -> Dense")
        print("  - Usa descritores de caracteristicas em vez de pixels brutos")
        print("  - Multiclasse: Classifica digitos 0-9")
        print("  - Binaria: Classifica apenas digitos 0 vs 1")
        print()
        print("SAIDAS DOS EXPERIMENTOS:")
        print("  - outputs_bruto/: Resultados CNN bruta multiclasse")
        print("  - outputs_bruto_binario/: Resultados CNN bruta binaria")
        print("  - outputs_hog/: Resultados CNN HOG multiclasse")
        print("  - outputs_hog_binario/: Resultados CNN HOG binaria")
        print()
        print("=" * 60)
        
        input("\nPressione Enter para voltar ao menu...")
    
    def optimize_cnn_raw(self):
        """Otimiza hiperparâmetros CNN Raw"""
        self.clear_screen()
        print("=" * 60)
        print("OTIMIZANDO: Hiperparametros CNN Raw")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        optimizer = HyperparameterOptimizer(
            self.data_manager, self.file_manager, self.evaluator
        )
        
        # Definir grade de parâmetros
        param_grid = optimizer.define_parameter_grid('cnn_raw')
        print(f"Grade de parametros: {param_grid}")
        
        # Executar grid search
        results = optimizer.grid_search_cnn_raw(param_grid)
        
        # Salvar resultados
        optimizer.save_results(results, 'cnn_raw_optimization_results.json')
        
        # Plotar resultados
        optimizer.plot_results(results, 'cnn_raw_optimization_plot.png')
        
        print(f"\nMelhores parametros: {results['best_params']}")
        print(f"Melhor score: {results['best_score']:.4f}")
        
        input("\nOtimizacao concluida! Pressione Enter para continuar...")

    def optimize_cnn_hog(self):
        """Otimiza hiperparâmetros CNN HOG"""
        self.clear_screen()
        print("=" * 60)
        print("OTIMIZANDO: Hiperparametros CNN HOG")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        optimizer = HyperparameterOptimizer(
            self.data_manager, self.file_manager, self.evaluator
        )
        
        # Definir grade de parâmetros
        param_grid = optimizer.define_parameter_grid('cnn_hog')
        print(f"Grade de parametros: {param_grid}")
        
        # Executar grid search
        results = optimizer.grid_search_cnn_hog(param_grid)
        
        # Salvar resultados
        optimizer.save_results(results, 'cnn_hog_optimization_results.json')
        
        # Plotar resultados
        optimizer.plot_results(results, 'cnn_hog_optimization_plot.png')
        
        print(f"\nMelhores parametros: {results['best_params']}")
        print(f"Melhor score: {results['best_score']:.4f}")
        
        input("\nOtimizacao concluida! Pressione Enter para continuar...")
    
    def run(self):
        """Executa o loop principal do menu"""
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == 0:  # Sair
                    self.clear_screen()
                    print("Saindo do sistema...")
                    break
                elif choice == 1:
                    self.run_cnn_raw_multiclass()
                elif choice == 2:
                    self.run_cnn_raw_binary()
                elif choice == 3:
                    self.run_cnn_hog_multiclass()
                elif choice == 4:
                    self.run_cnn_hog_binary()
                elif choice == 5:
                    self.run_all_experiments()
                elif choice == 6:
                    self.show_info()
                elif choice == 7:
                    self.optimize_cnn_raw()
                elif choice == 8:
                    self.optimize_cnn_hog()
                    
            except KeyboardInterrupt:
                self.clear_screen()
                print("\nSaindo do sistema...")
                break
            except Exception as e:
                print(f"\nErro inesperado: {e}")
                input("Pressione Enter para continuar...")