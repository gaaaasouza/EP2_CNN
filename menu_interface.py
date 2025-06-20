import os
import sys
import json # Importar json
# from experiment_runner import ExperimentRunner # Não mais necessário se o menu gerenciar
from data_manager import DataManager
from file_manager import FileManager
from model_evaluator import ModelEvaluator
from cnn_raw_classifier import CNNRawClassifier # Usar a classe base
from cnn_hog_classifier import CNNHOGClassifier # Usar a classe base
from hyperparameter_optimizer import HyperparameterOptimizer

class MenuInterface:
    """Interface de menu para seleção de algoritmos - compatível com Windows"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.file_manager = FileManager()
        self.evaluator = ModelEvaluator()
        self.data_prepared = False
        self.best_cnn_raw_params = None # Para armazenar os melhores parâmetros
        self.best_cnn_hog_params = None # Para armazenar os melhores parâmetros
    
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
        print("1. CNN com Dados Brutos - Multiclasse (Padrao)")
        print("2. CNN com Dados Brutos - Binaria (0 vs 1) (Padrao)")
        print("3. CNN com HOG - Multiclasse (Padrao)")
        print("4. CNN com HOG - Binaria (0 vs 1) (Padrao)")
        print("5. Executar Todos os Experimentos (Padrao)")
        print("6. Informacoes sobre os Algoritmos")
        print("7. Otimizar Hiperparametros CNN Raw (Grid Search)") # Mudar para Grid Search
        print("8. Otimizar Hiperparametros CNN HOG (Grid Search)") # Mudar para Grid Search
        print("9. Otimizar Hiperparametros CNN Raw (Random Search)") # Nova opção
        print("10. Otimizar Hiperparametros CNN HOG (Random Search)") # Nova opção
        print("11. Rodar CNN Raw com MELHORES parametros otimizados") # Nova opção
        print("12. Rodar CNN HOG com MELHORES parametros otimizados") # Nova opção
        print("0. Sair")
        print()
        print("=" * 60)
        
        while True:
            try:
                choice = input("Digite sua opcao (0-12): ").strip()
                if choice in [str(i) for i in range(13)]: # Para abranger de 0 a 12
                    return int(choice)
                else:
                    print("Opcao invalida! Digite um numero entre 0 e 12.")
            except (ValueError, KeyboardInterrupt):
                print("Entrada invalida! Digite um numero.")
    
    def run_cnn_raw_multiclass(self):
        """Executa CNN com dados brutos - multiclasse com parametros padrao"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN com Dados Brutos - Multiclasse (Parametros Padrao)")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        cnn_raw = CNNRawClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_raw.run_multiclass()
        
        input("\nExperimento concluido! Pressione Enter para continuar...")
    
    def run_cnn_raw_binary(self):
        """Executa CNN com dados brutos - binária com parametros padrao"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN com Dados Brutos - Binaria (0 vs 1) (Parametros Padrao)")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        cnn_raw = CNNRawClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_raw.run_binary()
        
        input("\nExperimento concluido! Pressione Enter para continuar...")
    
    def run_cnn_hog_multiclass(self):
        """Executa CNN com HOG - multiclasse com parametros padrao"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN com HOG - Multiclasse (Parametros Padrao)")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        cnn_hog = CNNHOGClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_hog.run_multiclass()
        
        input("\nExperimento concluido! Pressione Enter para continuar...")
    
    def run_cnn_hog_binary(self):
        """Executa CNN com HOG - binária com parametros padrao"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN com HOG - Binaria (0 vs 1) (Parametros Padrao)")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        cnn_hog = CNNHOGClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_hog.run_binary()
        
        input("\nExperimento concluido! Pressione Enter para continuar...")
    
    def run_all_experiments(self):
        """Executa todos os experimentos com parametros padrao"""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: Todos os Experimentos (Parametros Padrao)")
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
        print("  - Arquitetura: Conv2D -> MaxPooling -> Flatten -> Dense -> Dropout -> Dense") # Atualizado
        print("  - Multiclasse: Classifica digitos 0-9")
        print("  - Binaria: Classifica apenas digitos 0 vs 1")
        print()
        print("CNN com HOG (Histogram of Oriented Gradients):")
        print("  - Extrai caracteristicas HOG das imagens")
        print("  - Arquitetura: Dense -> Dense -> Dropout -> Dense") # Atualizado
        print("  - Usa descritores de caracteristicas em vez de pixels brutos")
        print("  - Multiclasse: Classifica digitos 0-9")
        print("  - Binaria: Classifica apenas digitos 0 vs 1")
        print()
        print("SAIDAS DOS EXPERIMENTOS:")
        print("  - outputs_bruto/: Resultados CNN bruta multiclasse")
        print("  - outputs_bruto_binario/: Resultados CNN bruta binaria")
        print("  - outputs_hog/: Resultados CNN HOG multiclasse")
        print("  - outputs_hog_binario/: Resultados CNN HOG binaria")
        print("  - *.json e *.png: Resultados de otimizacao de hiperparametros") # Adicionado
        print()
        print("=" * 60)
        
        input("\nPressione Enter para voltar ao menu...")
    
    def _run_optimization(self, model_type: str, search_type: str):
        """Função auxiliar para executar otimização (Grid ou Random)"""
        self.clear_screen()
        print("=" * 60)
        print(f"OTIMIZANDO: Hiperparametros {model_type.upper()} ({search_type.upper()} Search)")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        optimizer = HyperparameterOptimizer(
            self.data_manager, self.file_manager, self.evaluator
        )
        
        param_grid = optimizer.define_parameter_grid(model_type)
        print(f"Grade de parametros: {param_grid}")
        
        results = None
        if search_type == 'grid':
            if model_type == 'cnn_raw':
                results = optimizer.grid_search_cnn_raw(param_grid)
            elif model_type == 'cnn_hog':
                results = optimizer.grid_search_cnn_hog(param_grid)
        elif search_type == 'random':
            n_iter = int(input("Quantas iterações para Random Search? (Ex: 20): "))
            if model_type == 'cnn_raw':
                results = optimizer.random_search_cnn_raw(param_grid, n_iter)
            elif model_type == 'cnn_hog':
                results = optimizer.random_search_cnn_hog(param_grid, n_iter)

        if results:
            filename_json = f"{model_type}_{search_type}_optimization_results.json"
            filename_plot = f"{model_type}_{search_type}_optimization_plot.png"
            
            optimizer.save_results(results, filename_json)
            optimizer.plot_results(results, filename_plot)
            
            if model_type == 'cnn_raw':
                self.best_cnn_raw_params = results['best_params']
            elif model_type == 'cnn_hog':
                self.best_cnn_hog_params = results['best_params']
            
            print(f"\nMelhores parametros: {results['best_params']}")
            print(f"Melhor score: {results['best_score']:.4f}")
        else:
            print("Nenhum resultado de otimização obtido.")
        
        input("\nOtimizacao concluida! Pressione Enter para continuar...")

    def optimize_cnn_raw_grid(self):
        self._run_optimization('cnn_raw', 'grid')

    def optimize_cnn_hog_grid(self):
        self._run_optimization('cnn_hog', 'grid')

    def optimize_cnn_raw_random(self):
        self._run_optimization('cnn_raw', 'random')

    def optimize_cnn_hog_random(self):
        self._run_optimization('cnn_hog', 'random')

    def run_cnn_raw_optimized(self):
        """Executa CNN Raw com os melhores parâmetros encontrados."""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN Raw com MELHORES PARAMETROS OTMIZADOS")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        # Tenta carregar os melhores parâmetros se não estiverem na memória
        if not self.best_cnn_raw_params:
            try:
                # Prioriza os resultados do Grid Search se existirem, senão tenta Random Search
                if os.path.exists('cnn_raw_grid_optimization_results.json'):
                    with open('cnn_raw_grid_optimization_results.json', 'r') as f:
                        results = json.load(f)
                        self.best_cnn_raw_params = results['best_params']
                elif os.path.exists('cnn_raw_random_optimization_results.json'):
                    with open('cnn_raw_random_optimization_results.json', 'r') as f:
                        results = json.load(f)
                        self.best_cnn_raw_params = results['best_params']
            except Exception as e:
                print(f"Erro ao carregar parâmetros otimizados para CNN Raw: {e}")

        if self.best_cnn_raw_params:
            print(f"Usando melhores parâmetros: {self.best_cnn_raw_params}")
            cnn_raw = CNNRawClassifier(self.data_manager, self.file_manager, self.evaluator)
            cnn_raw.run_multiclass(self.best_cnn_raw_params) # Assume multiclasse para rodar o final
        else:
            print("Nenhum parâmetro otimizado encontrado para CNN Raw. Execute a otimização primeiro.")
        
        input("\nExperimento concluido! Pressione Enter para continuar...")

    def run_cnn_hog_optimized(self):
        """Executa CNN HOG com os melhores parâmetros encontrados."""
        self.clear_screen()
        print("=" * 60)
        print("EXECUTANDO: CNN HOG com MELHORES PARAMETROS OTMIZADOS")
        print("=" * 60)
        
        self.prepare_data_if_needed()
        
        if not self.best_cnn_hog_params:
            try:
                if os.path.exists('cnn_hog_grid_optimization_results.json'):
                    with open('cnn_hog_grid_optimization_results.json', 'r') as f:
                        results = json.load(f)
                        self.best_cnn_hog_params = results['best_params']
                elif os.path.exists('cnn_hog_random_optimization_results.json'):
                    with open('cnn_hog_random_optimization_results.json', 'r') as f:
                        results = json.load(f)
                        self.best_cnn_hog_params = results['best_params']
            except Exception as e:
                print(f"Erro ao carregar parâmetros otimizados para CNN HOG: {e}")

        if self.best_cnn_hog_params:
            print(f"Usando melhores parâmetros: {self.best_cnn_hog_params}")
            cnn_hog = CNNHOGClassifier(self.data_manager, self.file_manager, self.evaluator)
            cnn_hog.run_multiclass(self.best_cnn_hog_params) # Assume multiclasse para rodar o final
        else:
            print("Nenhum parâmetro otimizado encontrado para CNN HOG. Execute a otimização primeiro.")
        
        input("\nExperimento concluido! Pressione Enter para continuar...")

    def run(self):
        """Executa o loop principal do menu"""
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == 0:
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
                    self.optimize_cnn_raw_grid()
                elif choice == 8:
                    self.optimize_cnn_hog_grid()
                elif choice == 9:
                    self.optimize_cnn_raw_random()
                elif choice == 10:
                    self.optimize_cnn_hog_random()
                elif choice == 11:
                    self.run_cnn_raw_optimized()
                elif choice == 12:
                    self.run_cnn_hog_optimized()
                    
            except KeyboardInterrupt:
                self.clear_screen()
                print("\nSaindo do sistema...")
                break
            except Exception as e:
                print(f"\nErro inesperado: {e}")
                input("Pressione Enter para continuar...")