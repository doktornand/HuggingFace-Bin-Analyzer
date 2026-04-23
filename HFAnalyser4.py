import torch
import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.spatial.distance import cosine
import networkx as nx

class HuggingFaceBinAnalyzer:
    """
    Analyseur progressif de fichiers .bin HuggingFace
    Permet une analyse en profondeur des modèles stockés au format PyTorch
    """
    
    def __init__(self, model_path: str):
        """
        Initialise l'analyseur avec le chemin vers le modèle
        
        Args:
            model_path: Chemin vers le dossier contenant les fichiers du modèle
        """
        self.model_path = Path(model_path)
        self.bin_files = list(self.model_path.glob("*.bin"))
        self.config_file = self.model_path / "config.json"
        self.tokenizer_config = self.model_path / "tokenizer_config.json"
        
        self.config = {}
        self.tensors_info = {}
        self.analysis_results = {}
        
        # Chargement de la configuration si disponible
        self._load_config()
    
    def _load_config(self):
        """Charge les fichiers de configuration disponibles"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                print(f"✓ Configuration chargée: {self.config.get('model_type', 'Unknown')}")
        except Exception as e:
            print(f"⚠ Erreur lors du chargement de la config: {e}")
    
    def analyze_structure(self) -> Dict[str, Any]:
        """
        Niveau 1: Analyse structurelle basique
        """
        print("🔍 Analyse structurelle en cours...")
        
        structure_info = {
            'files_found': len(self.bin_files),
            'total_size_mb': 0,
            'tensor_count': 0,
            'files_info': [],
            'model_type': self.config.get('model_type', 'Unknown'),
            'architecture': self.config.get('architectures', ['Unknown'])[0] if self.config.get('architectures') else 'Unknown'
        }
        
        for bin_file in self.bin_files:
            file_size = bin_file.stat().st_size / (1024 * 1024)  # MB
            structure_info['total_size_mb'] += file_size
            
            try:
                # Chargement avec map_location='cpu' pour éviter les problèmes GPU
                checkpoint = torch.load(bin_file, map_location='cpu', weights_only=True)
                
                file_info = {
                    'filename': bin_file.name,
                    'size_mb': round(file_size, 2),
                    'tensor_count': len(checkpoint),
                    'tensors': list(checkpoint.keys())[:10]  # Premiers 10 tenseurs
                }
                
                structure_info['files_info'].append(file_info)
                structure_info['tensor_count'] += len(checkpoint)
                
                # Stockage des tenseurs pour analyses ultérieures
                for name, tensor in checkpoint.items():
                    self.tensors_info[name] = {
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'size_mb': tensor.numel() * tensor.element_size() / (1024 * 1024),
                        'file': bin_file.name
                    }
                
            except Exception as e:
                print(f"⚠ Erreur lors de l'analyse de {bin_file.name}: {e}")
        
        structure_info['total_size_mb'] = round(structure_info['total_size_mb'], 2)
        self.analysis_results['structure'] = structure_info
        
        return structure_info
    
    def analyze_tensors(self, sample_size: int = 5) -> Dict[str, Any]:
        """
        Niveau 2: Analyse des tenseurs
        
        Args:
            sample_size: Nombre de tenseurs à analyser en détail
        """
        print("🔬 Analyse des tenseurs en cours...")
        
        if not self.tensors_info:
            print("⚠ Exécutez d'abord analyze_structure()")
            return {}
        
        tensor_analysis = {
            'layer_types': defaultdict(int),
            'dtype_distribution': defaultdict(int),
            'size_distribution': [],
            'largest_tensors': [],
            'detailed_analysis': {}
        }
        
        # Classification des couches
        for name, info in self.tensors_info.items():
            # Identification du type de couche
            if 'embed' in name.lower():
                tensor_analysis['layer_types']['embedding'] += 1
            elif 'attention' in name.lower() or 'attn' in name.lower():
                tensor_analysis['layer_types']['attention'] += 1
            elif 'mlp' in name.lower() or 'feed_forward' in name.lower():
                tensor_analysis['layer_types']['mlp'] += 1
            elif 'norm' in name.lower() or 'ln' in name.lower():
                tensor_analysis['layer_types']['normalization'] += 1
            elif 'lm_head' in name.lower() or 'classifier' in name.lower():
                tensor_analysis['layer_types']['output'] += 1
            else:
                tensor_analysis['layer_types']['other'] += 1
            
            # Distribution des types de données
            tensor_analysis['dtype_distribution'][info['dtype']] += 1
            
            # Distribution des tailles
            tensor_analysis['size_distribution'].append({
                'name': name,
                'size_mb': info['size_mb'],
                'shape': info['shape']
            })
        
        # Top des plus gros tenseurs
        tensor_analysis['largest_tensors'] = sorted(
            tensor_analysis['size_distribution'],
            key=lambda x: x['size_mb'],
            reverse=True
        )[:10]
        
        # Analyse détaillée d'un échantillon
        sample_tensors = list(self.tensors_info.keys())[:sample_size]
        
        for tensor_name in sample_tensors:
            try:
                # Chargement du tenseur spécifique
                tensor_file = None
                for bin_file in self.bin_files:
                    checkpoint = torch.load(bin_file, map_location='cpu', weights_only=True)
                    if tensor_name in checkpoint:
                        tensor = checkpoint[tensor_name]
                        tensor_file = bin_file
                        break
                
                if tensor is not None:
                    detailed_info = self._analyze_single_tensor(tensor, tensor_name)
                    tensor_analysis['detailed_analysis'][tensor_name] = detailed_info
                
            except Exception as e:
                print(f"⚠ Erreur lors de l'analyse détaillée de {tensor_name}: {e}")
        
        self.analysis_results['tensors'] = tensor_analysis
        return tensor_analysis
    
    def _analyze_single_tensor(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Analyse détaillée d'un tenseur individuel"""
        
        tensor_np = tensor.detach().numpy()
        
        analysis = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'total_params': tensor.numel(),
            'memory_mb': tensor.numel() * tensor.element_size() / (1024 * 1024),
            'statistics': {
                'mean': float(np.mean(tensor_np)),
                'std': float(np.std(tensor_np)),
                'min': float(np.min(tensor_np)),
                'max': float(np.max(tensor_np)),
                'zero_percentage': float(np.sum(tensor_np == 0) / tensor_np.size * 100)
            },
            'sparsity': {
                'is_sparse': float(np.sum(tensor_np == 0) / tensor_np.size) > 0.1,
                'sparsity_ratio': float(np.sum(tensor_np == 0) / tensor_np.size)
            }
        }
        
        # Détection de patterns de quantification
        unique_values = len(np.unique(tensor_np))
        total_values = tensor_np.size
        
        if unique_values < total_values * 0.1:  # Moins de 10% de valeurs uniques
            analysis['quantization'] = {
                'likely_quantized': True,
                'unique_values': unique_values,
                'compression_ratio': total_values / unique_values
            }
        else:
            analysis['quantization'] = {
                'likely_quantized': False,
                'unique_values': unique_values
            }
        
        return analysis
    
    def analyze_architecture(self) -> Dict[str, Any]:
        """
        Niveau 3: Analyse architecturale avancée
        Reconstruction de l'architecture et analyse comparative
        """
        print("🏗️ Analyse architecturale en cours...")
        
        if not self.tensors_info:
            print("⚠ Exécutez d'abord analyze_structure()")
            return {}
        
        arch_analysis = {
            'reconstructed_architecture': {},
            'layer_mapping': {},
            'attention_analysis': {},
            'parameter_distribution': {},
            'model_topology': {},
            'embedding_analysis': {},
            'architectural_patterns': []
        }
        
        # Reconstruction de l'architecture
        arch_analysis['reconstructed_architecture'] = self._reconstruct_architecture()
        
        # Analyse des couches d'attention
        arch_analysis['attention_analysis'] = self._analyze_attention_layers()
        
        # Distribution des paramètres par composant
        arch_analysis['parameter_distribution'] = self._analyze_parameter_distribution()
        
        # Analyse des embeddings
        arch_analysis['embedding_analysis'] = self._analyze_embeddings()
        
        # Détection de patterns architecturaux
        arch_analysis['architectural_patterns'] = self._detect_architectural_patterns()
        
        # Création de la topologie du modèle
        arch_analysis['model_topology'] = self._create_model_topology()
        
        self.analysis_results['architecture'] = arch_analysis
        return arch_analysis
    
    def _reconstruct_architecture(self) -> Dict[str, Any]:
        """Reconstruit l'architecture du modèle à partir des tenseurs"""
        
        architecture = {
            'model_type': self.config.get('model_type', 'unknown'),
            'num_layers': 0,
            'hidden_size': 0,
            'num_attention_heads': 0,
            'intermediate_size': 0,
            'vocab_size': 0,
            'layer_structure': [],
            'special_tokens': {}
        }
        
        # Analyse des patterns de noms pour identifier la structure
        layer_pattern = re.compile(r'layers?\.(\d+)\.|h\.(\d+)\.|transformer\.h\.(\d+)\.')
        
        layers_found = set()
        for tensor_name in self.tensors_info.keys():
            match = layer_pattern.search(tensor_name)
            if match:
                layer_num = int(match.group(1) or match.group(2) or match.group(3))
                layers_found.add(layer_num)
        
        architecture['num_layers'] = max(layers_found) + 1 if layers_found else 0
        
        # Déduction des dimensions depuis les tenseurs
        for name, info in self.tensors_info.items():
            shape = info['shape']
            
            # Hidden size depuis les embeddings ou linear layers
            if 'embed' in name.lower() and len(shape) >= 2:
                if architecture['vocab_size'] == 0:
                    architecture['vocab_size'] = max(shape)
                architecture['hidden_size'] = min(shape) if max(shape) > min(shape) else shape[-1]
            
            # Attention heads depuis les poids d'attention
            if 'attn' in name.lower() and 'weight' in name.lower():
                if len(shape) == 2 and shape[0] == shape[1]:
                    # Probablement une matrice d'attention
                    hidden_size = shape[0]
                    if architecture['hidden_size'] == 0:
                        architecture['hidden_size'] = hidden_size
                    
                    # Estimation du nombre de têtes (commun: 8, 12, 16, 32)
                    common_heads = [8, 12, 16, 32, 64]
                    for heads in common_heads:
                        if hidden_size % heads == 0:
                            architecture['num_attention_heads'] = heads
                            break
            
            # Intermediate size depuis les couches MLP
            if ('mlp' in name.lower() or 'ffn' in name.lower()) and 'weight' in name.lower():
                if len(shape) == 2:
                    architecture['intermediate_size'] = max(shape)
        
        # Analyse de la structure par couche
        for layer_num in sorted(layers_found):
            layer_tensors = [name for name in self.tensors_info.keys() 
                           if f'.{layer_num}.' in name or f'h.{layer_num}.' in name]
            
            layer_info = {
                'layer_id': layer_num,
                'tensors': layer_tensors,
                'components': []
            }
            
            # Classification des composants
            for tensor in layer_tensors:
                if 'attn' in tensor.lower():
                    layer_info['components'].append('attention')
                elif 'mlp' in tensor.lower() or 'ffn' in tensor.lower():
                    layer_info['components'].append('mlp')
                elif 'norm' in tensor.lower():
                    layer_info['components'].append('normalization')
            
            architecture['layer_structure'].append(layer_info)
        
        return architecture
    
    def _analyze_attention_layers(self) -> Dict[str, Any]:
        """Analyse spécialisée des couches d'attention"""
        
        attention_analysis = {
            'attention_patterns': {},
            'head_specialization': {},
            'attention_weights_stats': {},
            'multi_head_analysis': {}
        }
        
        # Recherche des tenseurs d'attention
        attention_tensors = {}
        for name, info in self.tensors_info.items():
            if 'attn' in name.lower() or 'attention' in name.lower():
                attention_tensors[name] = info
        
        # Analyse des patterns d'attention
        for name, info in attention_tensors.items():
            try:
                # Chargement du tenseur pour analyse détaillée
                tensor = self._load_specific_tensor(name)
                if tensor is not None:
                    attention_analysis['attention_patterns'][name] = {
                        'shape': info['shape'],
                        'attention_type': self._classify_attention_tensor(name, tensor),
                        'head_analysis': self._analyze_attention_heads(tensor, name)
                    }
            except Exception as e:
                print(f"⚠ Erreur analyse attention {name}: {e}")
        
        return attention_analysis
    
    def _analyze_parameter_distribution(self) -> Dict[str, Any]:
        """Analyse de la distribution des paramètres par composant"""
        
        distribution = {
            'by_component': defaultdict(float),
            'by_layer': defaultdict(float),
            'total_parameters': 0,
            'efficiency_metrics': {}
        }
        
        for name, info in self.tensors_info.items():
            param_count = np.prod(info['shape']) if info['shape'] else 0
            distribution['total_parameters'] += param_count
            
            # Classification par composant
            if 'embed' in name.lower():
                distribution['by_component']['embedding'] += param_count
            elif 'attn' in name.lower():
                distribution['by_component']['attention'] += param_count
            elif 'mlp' in name.lower() or 'ffn' in name.lower():
                distribution['by_component']['mlp'] += param_count
            elif 'norm' in name.lower():
                distribution['by_component']['normalization'] += param_count
            elif 'lm_head' in name.lower():
                distribution['by_component']['output'] += param_count
            else:
                distribution['by_component']['other'] += param_count
            
            # Classification par couche
            layer_match = re.search(r'layers?\.(\d+)\.|h\.(\d+)\.', name)
            if layer_match:
                layer_num = int(layer_match.group(1) or layer_match.group(2))
                distribution['by_layer'][f'layer_{layer_num}'] += param_count
        
        # Calcul des métriques d'efficacité
        total = distribution['total_parameters']
        if total > 0:
            distribution['efficiency_metrics'] = {
                'embedding_ratio': distribution['by_component']['embedding'] / total,
                'attention_ratio': distribution['by_component']['attention'] / total,
                'mlp_ratio': distribution['by_component']['mlp'] / total,
                'params_per_layer': total / max(1, len(distribution['by_layer']))
            }
        
        return distribution
    
    def _analyze_embeddings(self) -> Dict[str, Any]:
        """Analyse des couches d'embedding"""
        
        embedding_analysis = {
            'word_embeddings': {},
            'positional_embeddings': {},
            'token_type_embeddings': {},
            'embedding_quality': {}
        }
        
        for name, info in self.tensors_info.items():
            if 'embed' in name.lower():
                try:
                    tensor = self._load_specific_tensor(name)
                    if tensor is not None:
                        analysis = self._analyze_embedding_tensor(tensor, name)
                        
                        if 'word' in name.lower() or 'token' in name.lower():
                            embedding_analysis['word_embeddings'][name] = analysis
                        elif 'pos' in name.lower():
                            embedding_analysis['positional_embeddings'][name] = analysis
                        elif 'type' in name.lower():
                            embedding_analysis['token_type_embeddings'][name] = analysis
                        
                except Exception as e:
                    print(f"⚠ Erreur analyse embedding {name}: {e}")
        
        return embedding_analysis
    
    def _detect_architectural_patterns(self) -> List[Dict[str, Any]]:
        """Détecte les patterns architecturaux connus"""
        
        patterns = []
        
        # Pattern Transformer standard
        if self._has_standard_transformer_pattern():
            patterns.append({
                'type': 'Standard Transformer',
                'confidence': 0.9,
                'description': 'Architecture Transformer classique avec attention multi-têtes'
            })
        
        # Pattern GPT
        if self._has_gpt_pattern():
            patterns.append({
                'type': 'GPT-like',
                'confidence': 0.85,
                'description': 'Architecture de type GPT (décodeur uniquement)'
            })
        
        # Pattern BERT
        if self._has_bert_pattern():
            patterns.append({
                'type': 'BERT-like',
                'confidence': 0.8,
                'description': 'Architecture de type BERT (encodeur bidirectionnel)'
            })
        
        # Pattern avec quantification
        if self._has_quantization_pattern():
            patterns.append({
                'type': 'Quantized Model',
                'confidence': 0.7,
                'description': 'Modèle avec quantification détectée'
            })
        
        return patterns
    
    def _create_model_topology(self) -> Dict[str, Any]:
        """Crée une représentation topologique du modèle"""
        
        topology = {
            'graph': {},
            'flow_analysis': {},
            'bottlenecks': [],
            'connection_patterns': {}
        }
        
        # Construction du graphe de connexions (simplifié)
        G = nx.DiGraph()
        
        # Ajout des nœuds basés sur les couches
        layer_pattern = re.compile(r'layers?\.(\d+)\.|h\.(\d+)\.')
        layers = set()
        
        for name in self.tensors_info.keys():
            match = layer_pattern.search(name)
            if match:
                layer_num = int(match.group(1) or match.group(2))
                layers.add(layer_num)
                G.add_node(f"layer_{layer_num}")
        
        # Connexions séquentielles entre couches
        sorted_layers = sorted(layers)
        for i in range(len(sorted_layers) - 1):
            G.add_edge(f"layer_{sorted_layers[i]}", f"layer_{sorted_layers[i+1]}")
        
        topology['graph'] = {
            'nodes': list(G.nodes()),
            'edges': list(G.edges()),
            'density': nx.density(G)
        }
        
        return topology
    
    def analyze_advanced_patterns(self) -> Dict[str, Any]:
        """
        Niveau 4: Analyse avancée des patterns et anomalies
        """
        print("🔬 Analyse avancée des patterns en cours...")
        
        if not self.tensors_info:
            print("⚠ Exécutez d'abord analyze_structure()")
            return {}
        
        advanced_analysis = {
            'weight_distributions': {},
            'anomaly_detection': {},
            'performance_estimation': {},
            'optimization_suggestions': [],
            'pattern_correlations': {},
            'weight_visualizations': {}
        }
        
        # Analyse des distributions de poids
        advanced_analysis['weight_distributions'] = self._analyze_weight_distributions()
        
        # Détection d'anomalies
        advanced_analysis['anomaly_detection'] = self._detect_anomalies()
        
        # Estimation des performances
        advanced_analysis['performance_estimation'] = self._estimate_performance()
        
        # Suggestions d'optimisation
        advanced_analysis['optimization_suggestions'] = self._generate_optimization_suggestions()
        
        # Analyse des corrélations entre patterns
        advanced_analysis['pattern_correlations'] = self._analyze_pattern_correlations()
        
        self.analysis_results['advanced'] = advanced_analysis
        return advanced_analysis
    
    def _analyze_weight_distributions(self) -> Dict[str, Any]:
        """Analyse statistique avancée des distributions de poids"""
        
        distributions = {
            'global_stats': {},
            'layer_comparisons': {},
            'distribution_types': {},
            'outlier_analysis': {}
        }
        
        all_weights = []
        layer_weights = defaultdict(list)
        
        # Échantillonnage des poids pour analyse globale
        sample_tensors = list(self.tensors_info.keys())[:10]  # Limiter pour performance
        
        for tensor_name in sample_tensors:
            try:
                tensor = self._load_specific_tensor(tensor_name)
                if tensor is not None:
                    weights = tensor.detach().numpy().flatten()
                    all_weights.extend(weights[:1000])  # Échantillon pour performance
                    
                    # Classification par couche
                    layer_match = re.search(r'layers?\.(\d+)\.|h\.(\d+)\.', tensor_name)
                    if layer_match:
                        layer_num = int(layer_match.group(1) or layer_match.group(2))
                        layer_weights[layer_num].extend(weights[:500])
                    
                    # Analyse de distribution par tenseur
                    distributions['distribution_types'][tensor_name] = self._classify_distribution(weights)
                    
            except Exception as e:
                print(f"⚠ Erreur analyse distribution {tensor_name}: {e}")
        
        # Statistiques globales
        if all_weights:
            all_weights = np.array(all_weights)
            distributions['global_stats'] = {
                'mean': float(np.mean(all_weights)),
                'std': float(np.std(all_weights)),
                'skewness': float(stats.skew(all_weights)),
                'kurtosis': float(stats.kurtosis(all_weights)),
                'entropy': self._calculate_entropy(all_weights)
            }
            
            # Détection d'outliers
            z_scores = np.abs(stats.zscore(all_weights))
            outliers = all_weights[z_scores > 3]
            distributions['outlier_analysis'] = {
                'outlier_percentage': len(outliers) / len(all_weights) * 100,
                'outlier_threshold': 3.0,
                'extreme_values': {
                    'min': float(np.min(all_weights)),
                    'max': float(np.max(all_weights))
                }
            }
        
        return distributions
    
    def _detect_anomalies(self) -> Dict[str, Any]:
        """Détection d'anomalies dans les poids du modèle"""
        
        anomalies = {
            'dead_neurons': [],
            'saturated_weights': [],
            'irregular_patterns': [],
            'suspicious_tensors': []
        }
        
        # Analyse d'un échantillon de tenseurs
        sample_tensors = list(self.tensors_info.keys())[:15]
        
        for tensor_name in sample_tensors:
            try:
                tensor = self._load_specific_tensor(tensor_name)
                if tensor is not None:
                    weights = tensor.detach().numpy()
                    
                    # Détection de neurones "morts" (tous à zéro)
                    if len(weights.shape) >= 2:
                        zero_neurons = np.sum(np.all(weights == 0, axis=1))
                        if zero_neurons > 0:
                            anomalies['dead_neurons'].append({
                                'tensor': tensor_name,
                                'dead_count': int(zero_neurons),
                                'total_neurons': weights.shape[0]
                            })
                    
                    # Détection de poids saturés
                    extreme_threshold = 3 * np.std(weights)
                    saturated = np.sum(np.abs(weights) > extreme_threshold)
                    if saturated > len(weights.flatten()) * 0.01:  # Plus de 1%
                        anomalies['saturated_weights'].append({
                            'tensor': tensor_name,
                            'saturated_count': int(saturated),
                            'threshold': float(extreme_threshold)
                        })
                    
                    # Patterns irréguliers (variance très faible ou très élevée)
                    variance = np.var(weights)
                    if variance < 1e-8 or variance > 100:
                        anomalies['irregular_patterns'].append({
                            'tensor': tensor_name,
                            'variance': float(variance),
                            'pattern_type': 'low_variance' if variance < 1e-8 else 'high_variance'
                        })
                    
            except Exception as e:
                anomalies['suspicious_tensors'].append({
                    'tensor': tensor_name,
                    'error': str(e)
                })
        
        return anomalies
    
    def _estimate_performance(self) -> Dict[str, Any]:
        """Estimation des performances théoriques du modèle"""
        
        performance = {
            'theoretical_flops': 0,
            'memory_requirements': {},
            'inference_estimation': {},
            'bottleneck_analysis': {}
        }
        
        # Calcul des FLOPS théoriques
        total_params = sum(np.prod(info['shape']) for info in self.tensors_info.values() if info['shape'])
        
        # Estimation approximative (dépend de l'architecture)
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']['reconstructed_architecture']
            seq_length = 512  # Longueur de séquence typique
            
            # FLOPS pour attention: O(n²d + nd²) où n=seq_length, d=hidden_size
            if arch['hidden_size'] > 0:
                attention_flops = arch['num_layers'] * (
                    seq_length * seq_length * arch['hidden_size'] +
                    seq_length * arch['hidden_size'] * arch['hidden_size']
                )
                
                # FLOPS pour MLP: O(nd_ff) où d_ff=intermediate_size
                mlp_flops = arch['num_layers'] * seq_length * arch['intermediate_size'] * arch['hidden_size']
                
                performance['theoretical_flops'] = attention_flops + mlp_flops
        
        # Estimation mémoire
        total_size_mb = sum(info['size_mb'] for info in self.tensors_info.values())
        performance['memory_requirements'] = {
            'model_size_mb': total_size_mb,
            'inference_memory_mb': total_size_mb * 1.5,  # Approximation avec activations
            'training_memory_mb': total_size_mb * 4  # Approximation avec gradients
        }
        
        return performance
    
    def _generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Génère des suggestions d'optimisation basées sur l'analyse"""
        
        suggestions = []
        
        # Vérification des anomalies détectées
        if 'advanced' in self.analysis_results:
            anomalies = self.analysis_results['advanced'].get('anomaly_detection', {})
            
            if anomalies.get('dead_neurons'):
                suggestions.append({
                    'type': 'pruning',
                    'priority': 'high',
                    'description': 'Neurones morts détectés - considérer le pruning',
                    'potential_reduction': '5-15%'
                })
            
            if anomalies.get('saturated_weights'):
                suggestions.append({
                    'type': 'regularization',
                    'priority': 'medium',
                    'description': 'Poids saturés détectés - augmenter la régularisation',
                    'potential_improvement': 'Stabilité accrue'
                })
        
        # Suggestions basées sur la distribution des paramètres
        if 'architecture' in self.analysis_results:
            param_dist = self.analysis_results['architecture'].get('parameter_distribution', {})
            
            if param_dist.get('efficiency_metrics', {}).get('embedding_ratio', 0) > 0.3:
                suggestions.append({
                    'type': 'vocabulary_optimization',
                    'priority': 'medium',
                    'description': 'Embeddings représentent >30% des paramètres - optimiser le vocabulaire',
                    'potential_reduction': '10-20%'
                })
        
        # Suggestions de quantification
        quantized_tensors = 0
        total_tensors = len(self.tensors_info)
        
        if 'tensors' in self.analysis_results:
            for details in self.analysis_results['tensors'].get('detailed_analysis', {}).values():
                if details.get('quantization', {}).get('likely_quantized', False):
                    quantized_tensors += 1
        
        if quantized_tensors / total_tensors < 0.5:
            suggestions.append({
                'type': 'quantization',
                'priority': 'high',
                'description': 'Modèle peu quantifié - considérer INT8 ou FP16',
                'potential_reduction': '25-50%'
            })
        
        return suggestions
    
    def _analyze_pattern_correlations(self) -> Dict[str, Any]:
        """Analyse les corrélations entre différents patterns de poids"""
        
        correlations = {
            'layer_correlations': {},
            'component_correlations': {},
            'similarity_matrix': {}
        }
        
        # Cette analyse nécessiterait plus de ressources computationnelles
        # Implémentation simplifiée pour la démonstration
        
        return correlations
    
    # Méthodes utilitaires pour les analyses avancées
    
    def _load_specific_tensor(self, tensor_name: str) -> Optional[torch.Tensor]:
        """Charge un tenseur spécifique depuis les fichiers .bin"""
        for bin_file in self.bin_files:
            try:
                checkpoint = torch.load(bin_file, map_location='cpu', weights_only=True)
                if tensor_name in checkpoint:
                    return checkpoint[tensor_name]
            except Exception:
                continue
        return None
    
    def _classify_attention_tensor(self, name: str, tensor: torch.Tensor) -> str:
        """Classifie le type de tenseur d'attention"""
        if 'query' in name.lower() or 'q_proj' in name.lower():
            return 'query'
        elif 'key' in name.lower() or 'k_proj' in name.lower():
            return 'key'
        elif 'value' in name.lower() or 'v_proj' in name.lower():
            return 'value'
        elif 'out' in name.lower() or 'o_proj' in name.lower():
            return 'output'
        else:
            return 'unknown'
    
    def _analyze_attention_heads(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Analyse les têtes d'attention individuelles"""
        shape = tensor.shape
        if len(shape) >= 2:
            return {
                'num_potential_heads': shape[0] // 64 if shape[0] >= 64 else 1,  # Approximation
                'head_dimension': 64 if shape[0] >= 64 else shape[0],
                'total_params': tensor.numel()
            }
        return {}
    
    def _analyze_embedding_tensor(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Analyse un tenseur d'embedding"""
        shape = tensor.shape
        weights = tensor.detach().numpy()
        
        analysis = {
            'vocab_size': shape[0] if len(shape) >= 2 else 0,
            'embedding_dim': shape[1] if len(shape) >= 2 else shape[0],
            'statistics': {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'sparsity': float(np.sum(weights == 0) / weights.size)
            }
        }
        
        # Analyse de la qualité des embeddings (échantillon)
        if len(shape) >= 2 and shape[0] > 100:
            sample_embeddings = weights[:100]  # Échantillon pour performance
            
            # Calcul des distances moyennes (mesure de diversité)
            distances = []
            for i in range(min(50, len(sample_embeddings))):
                for j in range(i+1, min(50, len(sample_embeddings))):
                    dist = cosine(sample_embeddings[i], sample_embeddings[j])
                    if not np.isnan(dist):
                        distances.append(dist)
            
            if distances:
                analysis['quality_metrics'] = {
                    'avg_cosine_distance': float(np.mean(distances)),
                    'embedding_diversity': float(np.std(distances))
                }
        
        return analysis
    
    def _classify_distribution(self, weights: np.ndarray) -> Dict[str, Any]:
        """Classifie le type de distribution des poids"""
        
        # Test de normalité
        _, p_normal = stats.normaltest(weights[:1000])  # Échantillon pour performance
        
        # Calcul des moments
        skewness = stats.skew(weights)
        kurtosis = stats.kurtosis(weights)
        
        classification = {
            'is_normal': p_normal > 0.05,
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'distribution_type': 'normal'
        }
        
        # Classification simple basée sur les moments
        if abs(skewness) > 1:
            classification['distribution_type'] = 'skewed'
        elif kurtosis > 3:
            classification['distribution_type'] = 'heavy_tailed'
        elif kurtosis < -1:
            classification['distribution_type'] = 'light_tailed'
        
        return classification
    
    def _calculate_entropy(self, weights: np.ndarray) -> float:
        """Calcule l'entropie des poids (mesure de complexité)"""
        # Discrétisation pour le calcul d'entropie
        hist, _ = np.histogram(weights, bins=50)
        hist = hist[hist > 0]  # Éliminer les bins vides
        prob = hist / np.sum(hist)
        return float(-np.sum(prob * np.log2(prob)))
    
    def _has_standard_transformer_pattern(self) -> bool:
        """Détecte le pattern Transformer standard"""
        has_attention = any('attn' in name.lower() for name in self.tensors_info.keys())
        has_mlp = any('mlp' in name.lower() or 'ffn' in name.lower() for name in self.tensors_info.keys())
        has_norm = any('norm' in name.lower() for name in self.tensors_info.keys())
        return has_attention and has_mlp and has_norm
    
    def _has_gpt_pattern(self) -> bool:
        """Détecte le pattern GPT"""
        model_type = self.config.get('model_type', '').lower()
        return 'gpt' in model_type or any('lm_head' in name.lower() for name in self.tensors_info.keys())
    
    def _has_bert_pattern(self) -> bool:
        """Détecte le pattern BERT"""
        model_type = self.config.get('model_type', '').lower()
        return 'bert' in model_type or any('pooler' in name.lower() for name in self.tensors_info.keys())
    
    def _has_quantization_pattern(self) -> bool:
        """Détecte la présence de quantification"""
        if 'tensors' in self.analysis_results:
            for details in self.analysis_results['tensors'].get('detailed_analysis', {}).values():
                if details.get('quantization', {}).get('likely_quantized', False):
                    return True
        
        """
        Génère un rapport complet de l'analyse
        """
        report = []
        report.append("=" * 60)
        report.append("RAPPORT D'ANALYSE - MODÈLE HUGGINGFACE")
        report.append("=" * 60)
        report.append("")
        
        # Section structure
        if 'structure' in self.analysis_results:
            struct = self.analysis_results['structure']
            report.append("📁 STRUCTURE DU MODÈLE")
            report.append("-" * 30)
            report.append(f"Type de modèle: {struct['model_type']}")
            report.append(f"Architecture: {struct['architecture']}")
            report.append(f"Nombre de fichiers .bin: {struct['files_found']}")
            report.append(f"Taille totale: {struct['total_size_mb']} MB")
            report.append(f"Nombre total de tenseurs: {struct['tensor_count']}")
            report.append("")
            
            for file_info in struct['files_info']:
                report.append(f"  📄 {file_info['filename']}: {file_info['size_mb']} MB, {file_info['tensor_count']} tenseurs")
            report.append("")
        
        # Section tenseurs
        if 'tensors' in self.analysis_results:
            tensors = self.analysis_results['tensors']
            report.append("🔬 ANALYSE DES TENSEURS")
            report.append("-" * 30)
            
            report.append("Types de couches:")
            for layer_type, count in tensors['layer_types'].items():
                report.append(f"  • {layer_type}: {count} tenseurs")
            report.append("")
            
            report.append("Distribution des types de données:")
            for dtype, count in tensors['dtype_distribution'].items():
                report.append(f"  • {dtype}: {count} tenseurs")
            report.append("")
            
            report.append("Top 5 des plus gros tenseurs:")
            for i, tensor_info in enumerate(tensors['largest_tensors'][:5], 1):
                report.append(f"  {i}. {tensor_info['name']}: {tensor_info['size_mb']:.2f} MB {tensor_info['shape']}")
            report.append("")
            
            # Analyses détaillées
            if tensors['detailed_analysis']:
                report.append("🔍 ANALYSES DÉTAILLÉES")
                report.append("-" * 30)
                
                for name, details in tensors['detailed_analysis'].items():
                    report.append(f"Tenseur: {name}")
                    report.append(f"  Shape: {details['shape']}")
                    report.append(f"  Paramètres: {details['total_params']:,}")
                    report.append(f"  Mémoire: {details['memory_mb']:.2f} MB")
                    
                    stats = details['statistics']
                    report.append(f"  Statistiques:")
                    report.append(f"    Moyenne: {stats['mean']:.6f}")
                    report.append(f"    Écart-type: {stats['std']:.6f}")
                    report.append(f"    Min/Max: {stats['min']:.6f} / {stats['max']:.6f}")
                    report.append(f"    Zéros: {stats['zero_percentage']:.2f}%")
                    
                    if details['quantization']['likely_quantized']:
                        report.append(f"  ⚡ Quantification détectée (ratio: {details['quantization']['compression_ratio']:.2f})")
                    
                    if details['sparsity']['is_sparse']:
                        report.append(f"  🕳️ Tenseur sparse ({details['sparsity']['sparsity_ratio']*100:.2f}% de zéros)")
                    
                    report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Rapport sauvegardé: {save_path}")
        
        return report_text
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> str:
        """
        Génère un rapport complet incluant tous les niveaux d'analyse
        """
        report = []
        report.append("=" * 80)
        report.append("RAPPORT COMPLET D'ANALYSE - MODÈLE HUGGINGFACE")
        report.append("=" * 80)
        report.append("")
        
        # Section structure (Niveau 1)
        if 'structure' in self.analysis_results:
            struct = self.analysis_results['structure']
            report.append("📁 1. ANALYSE STRUCTURELLE")
            report.append("-" * 40)
            report.append(f"Type de modèle: {struct['model_type']}")
            report.append(f"Architecture: {struct['architecture']}")
            report.append(f"Nombre de fichiers .bin: {struct['files_found']}")
            report.append(f"Taille totale: {struct['total_size_mb']} MB")
            report.append(f"Nombre total de tenseurs: {struct['tensor_count']}")
            report.append("")
            
            for file_info in struct['files_info']:
                report.append(f"  📄 {file_info['filename']}: {file_info['size_mb']} MB, {file_info['tensor_count']} tenseurs")
            report.append("")
        
        # Section tenseurs (Niveau 2)
        if 'tensors' in self.analysis_results:
            tensors = self.analysis_results['tensors']
            report.append("🔬 2. ANALYSE DES TENSEURS")
            report.append("-" * 40)
            
            report.append("Types de couches:")
            for layer_type, count in tensors['layer_types'].items():
                report.append(f"  • {layer_type}: {count} tenseurs")
            report.append("")
            
            report.append("Distribution des types de données:")
            for dtype, count in tensors['dtype_distribution'].items():
                report.append(f"  • {dtype}: {count} tenseurs")
            report.append("")
            
            report.append("Top 5 des plus gros tenseurs:")
            for i, tensor_info in enumerate(tensors['largest_tensors'][:5], 1):
                report.append(f"  {i}. {tensor_info['name']}: {tensor_info['size_mb']:.2f} MB {tensor_info['shape']}")
            report.append("")
        
        # Section architecture (Niveau 3)
        if 'architecture' in self.analysis_results:
            arch_analysis = self.analysis_results['architecture']
            report.append("🏗️ 3. ANALYSE ARCHITECTURALE")
            report.append("-" * 40)
            
            # Architecture reconstruite
            arch = arch_analysis['reconstructed_architecture']
            report.append("Architecture reconstruite:")
            report.append(f"  • Nombre de couches: {arch.get('num_layers', 'N/A')}")
            report.append(f"  • Taille cachée: {arch.get('hidden_size', 'N/A')}")
            report.append(f"  • Têtes d'attention: {arch.get('num_attention_heads', 'N/A')}")
            report.append(f"  • Taille intermédiaire: {arch.get('intermediate_size', 'N/A')}")
            report.append(f"  • Taille vocabulaire: {arch.get('vocab_size', 'N/A')}")
            report.append("")
            
            # Distribution des paramètres
            param_dist = arch_analysis['parameter_distribution']
            report.append("Distribution des paramètres:")
            total_params = param_dist.get('total_parameters', 0)
            report.append(f"  • Total: {total_params:,} paramètres")
            
            for component, count in param_dist['by_component'].items():
                percentage = (count / total_params * 100) if total_params > 0 else 0
                report.append(f"  • {component}: {count:,} ({percentage:.1f}%)")
            report.append("")
            
            # Patterns architecturaux détectés
            patterns = arch_analysis.get('architectural_patterns', [])
            if patterns:
                report.append("Patterns architecturaux détectés:")
                for pattern in patterns:
                    report.append(f"  • {pattern['type']} (confiance: {pattern['confidence']:.1%})")
                    report.append(f"    {pattern['description']}")
                report.append("")
        
        # Section analyse avancée (Niveau 4)
        if 'advanced' in self.analysis_results:
            advanced = self.analysis_results['advanced']
            report.append("🔬 4. ANALYSE AVANCÉE")
            report.append("-" * 40)
            
            # Distributions de poids
            weight_dist = advanced.get('weight_distributions', {})
            global_stats = weight_dist.get('global_stats', {})
            if global_stats:
                report.append("Statistiques globales des poids:")
                report.append(f"  • Moyenne: {global_stats.get('mean', 0):.6f}")
                report.append(f"  • Écart-type: {global_stats.get('std', 0):.6f}")
                report.append(f"  • Asymétrie: {global_stats.get('skewness', 0):.3f}")
                report.append(f"  • Aplatissement: {global_stats.get('kurtosis', 0):.3f}")
                report.append(f"  • Entropie: {global_stats.get('entropy', 0):.3f}")
                report.append("")
            
            # Détection d'anomalies
            anomalies = advanced.get('anomaly_detection', {})
            anomaly_found = False
            
            if anomalies.get('dead_neurons'):
                anomaly_found = True
                report.append(f"⚠️ Neurones morts détectés: {len(anomalies['dead_neurons'])} cas")
                for anomaly in anomalies['dead_neurons'][:3]:  # Top 3
                    report.append(f"  • {anomaly['tensor']}: {anomaly['dead_count']}/{anomaly['total_neurons']} neurones")
            
            if anomalies.get('saturated_weights'):
                anomaly_found = True
                report.append(f"⚠️ Poids saturés détectés: {len(anomalies['saturated_weights'])} cas")
            
            if anomalies.get('irregular_patterns'):
                anomaly_found = True
                report.append(f"⚠️ Patterns irréguliers: {len(anomalies['irregular_patterns'])} cas")
            
            if not anomaly_found:
                report.append("✅ Aucune anomalie majeure détectée")
            
            report.append("")
            
            # Estimation des performances
            performance = advanced.get('performance_estimation', {})
            memory_req = performance.get('memory_requirements', {})
            if memory_req:
                report.append("Estimation des performances:")
                model_size = memory_req.get('model_size_mb', 0)
                report.append(f"  • Taille modèle: {model_size:.1f} MB ({model_size/1024:.1f} GB)")
                
                inference_mem = memory_req.get('inference_memory_mb', 0)
                report.append(f"  • Mémoire inférence: {inference_mem:.1f} MB ({inference_mem/1024:.1f} GB)")
                
                training_mem = memory_req.get('training_memory_mb', 0)
                report.append(f"  • Mémoire entraînement: {training_mem:.1f} MB ({training_mem/1024:.1f} GB)")
                
                flops = performance.get('theoretical_flops', 0)
                if flops > 0:
                    report.append(f"  • FLOPS théoriques: {flops:.2e}")
                report.append("")
            
            # Suggestions d'optimisation
            suggestions = advanced.get('optimization_suggestions', [])
            if suggestions:
                report.append("💡 SUGGESTIONS D'OPTIMISATION")
                report.append("-" * 40)
                
                high_priority = [s for s in suggestions if s.get('priority') == 'high']
                medium_priority = [s for s in suggestions if s.get('priority') == 'medium']
                
                if high_priority:
                    report.append("🔴 Priorité élevée:")
                    for sugg in high_priority:
                        report.append(f"  • {sugg['description']}")
                        if 'potential_reduction' in sugg:
                            report.append(f"    Réduction potentielle: {sugg['potential_reduction']}")
                        report.append("")
                
                if medium_priority:
                    report.append("🟡 Priorité moyenne:")
                    for sugg in medium_priority:
                        report.append(f"  • {sugg['description']}")
                        if 'potential_reduction' in sugg:
                            report.append(f"    Réduction potentielle: {sugg['potential_reduction']}")
                        report.append("")
        
        # Section résumé
        report.append("📋 RÉSUMÉ EXÉCUTIF")
        report.append("-" * 40)
        
        if 'structure' in self.analysis_results:
            total_size = self.analysis_results['structure']['total_size_mb']
            total_tensors = self.analysis_results['structure']['tensor_count']
            report.append(f"• Modèle de {total_size} MB avec {total_tensors:,} tenseurs")
        
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']['reconstructed_architecture']
            layers = arch.get('num_layers', 0)
            hidden_size = arch.get('hidden_size', 0)
            if layers > 0 and hidden_size > 0:
                report.append(f"• Architecture: {layers} couches, dimension cachée {hidden_size}")
        
        if 'advanced' in self.analysis_results:
            suggestions = self.analysis_results['advanced']['optimization_suggestions']
            if suggestions:
                high_priority_count = len([s for s in suggestions if s.get('priority') == 'high'])
                report.append(f"• {len(suggestions)} suggestions d'optimisation dont {high_priority_count} prioritaires")
            
            anomalies = self.analysis_results['advanced']['anomaly_detection']
            total_anomalies = sum(len(anomalies.get(key, [])) for key in ['dead_neurons', 'saturated_weights', 'irregular_patterns'])
            if total_anomalies > 0:
                report.append(f"• ⚠️ {total_anomalies} anomalies détectées nécessitant attention")
            else:
                report.append("• ✅ Modèle en bon état, aucune anomalie critique")
        
        report.append("")
        report.append("=" * 80)
        
        comprehensive_report = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(comprehensive_report)
            print(f"📄 Rapport complet sauvegardé: {save_path}")
        
        return comprehensive_report
    
    def visualize_analysis(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Génère des visualisations de l'analyse
        """
        if 'tensors' not in self.analysis_results:
            print("⚠ Exécutez d'abord analyze_tensors()")
            return
        
        tensors = self.analysis_results['tensors']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Analyse du Modèle HuggingFace', fontsize=16, fontweight='bold')
        
        # Distribution des types de couches
        layer_types = list(tensors['layer_types'].keys())
        layer_counts = list(tensors['layer_types'].values())
        
        axes[0, 0].pie(layer_counts, labels=layer_types, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Distribution des Types de Couches')
        
        # Distribution des tailles de tenseurs
        sizes = [t['size_mb'] for t in tensors['size_distribution']]
        axes[0, 1].hist(sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Taille (MB)')
        axes[0, 1].set_ylabel('Nombre de tenseurs')
        axes[0, 1].set_title('Distribution des Tailles de Tenseurs')
        axes[0, 1].set_yscale('log')
        
        # Top 10 des plus gros tenseurs
        top_tensors = tensors['largest_tensors'][:10]
        names = [t['name'].split('.')[-1] for t in top_tensors]  # Noms simplifiés
        sizes = [t['size_mb'] for t in top_tensors]
        
        axes[1, 0].barh(range(len(names)), sizes, color='lightcoral')
        axes[1, 0].set_yticks(range(len(names)))
        axes[1, 0].set_yticklabels(names, fontsize=8)
        axes[1, 0].set_xlabel('Taille (MB)')
        axes[1, 0].set_title('Top 10 des Plus Gros Tenseurs')
        
        # Distribution des types de données
        dtypes = list(tensors['dtype_distribution'].keys())
        dtype_counts = list(tensors['dtype_distribution'].values())
        
        axes[1, 1].bar(dtypes, dtype_counts, color='lightgreen', alpha=0.7)
        axes[1, 1].set_xlabel('Type de données')
        axes[1, 1].set_ylabel('Nombre de tenseurs')
        axes[1, 1].set_title('Distribution des Types de Données')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Exemple d'utilisation
def analyze_model(model_path: str):
    """
    Fonction utilitaire pour analyser un modèle complet
    """
    analyzer = HuggingFaceBinAnalyzer(model_path)
    
    # Analyse structurelle
    structure = analyzer.analyze_structure()
    print("\n" + "="*50)
    print("RÉSULTATS DE L'ANALYSE STRUCTURELLE")
    print("="*50)
    print(f"Modèle: {structure['model_type']} ({structure['architecture']})")
    print(f"Taille totale: {structure['total_size_mb']} MB")
    print(f"Nombre de tenseurs: {structure['tensor_count']}")
    
    # Analyse des tenseurs
    tensor_analysis = analyzer.analyze_tensors(sample_size=3)
    
    # Génération du rapport
    report = analyzer.generate_report()
    print("\n" + report)
    
    # Visualisation (commentée pour éviter les erreurs si matplotlib n'est pas disponible)
    # analyzer.visualize_analysis()
    
    return analyzer

# Fonction d'analyse complète avec tous les niveaux
def comprehensive_analysis(model_path: str, sample_size: int = 5, visualize: bool = True):
    """
    Analyse complète du modèle avec tous les niveaux d'analyse
    
    Args:
        model_path: Chemin vers le modèle HuggingFace
        sample_size: Nombre de tenseurs à analyser en détail
        visualize: Générer les visualisations
    """
    print("🚀 Démarrage de l'analyse complète du modèle...")
    print("=" * 60)
    
    analyzer = HuggingFaceBinAnalyzer(model_path)
    
    # Niveau 1: Analyse structurelle
    print("\n🔍 Niveau 1: Analyse structurelle...")
    structure = analyzer.analyze_structure()
    print(f"✅ Structure analysée: {structure['tensor_count']} tenseurs, {structure['total_size_mb']} MB")
    
    # Niveau 2: Analyse des tenseurs
    print("\n🔬 Niveau 2: Analyse des tenseurs...")
    tensors = analyzer.analyze_tensors(sample_size=sample_size)
    print(f"✅ Tenseurs analysés: {len(tensors['layer_types'])} types de couches détectés")
    
    # Niveau 3: Analyse architecturale
    print("\n🏗️ Niveau 3: Analyse architecturale...")
    try:
        architecture = analyzer.analyze_architecture()
        arch_info = architecture['reconstructed_architecture']
        print(f"✅ Architecture reconstruite: {arch_info.get('num_layers', 0)} couches")
        print(f"   Hidden size: {arch_info.get('hidden_size', 'N/A')}")
        print(f"   Attention heads: {arch_info.get('num_attention_heads', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Erreur dans l'analyse architecturale: {e}")
    
    # Niveau 4: Analyse avancée
    print("\n🔬 Niveau 4: Analyse avancée des patterns...")
    try:
        advanced = analyzer.analyze_advanced_patterns()
        anomalies = advanced['anomaly_detection']
        total_anomalies = sum(len(anomalies.get(key, [])) for key in ['dead_neurons', 'saturated_weights', 'irregular_patterns'])
        print(f"✅ Analyse avancée terminée: {total_anomalies} anomalies détectées")
        
        suggestions = advanced['optimization_suggestions']
        if suggestions:
            high_priority = len([s for s in suggestions if s.get('priority') == 'high'])
            print(f"💡 {len(suggestions)} suggestions d'optimisation ({high_priority} haute priorité)")
    except Exception as e:
        print(f"⚠️ Erreur dans l'analyse avancée: {e}")
    
    # Génération du rapport complet
    print("\n📄 Génération du rapport complet...")
    report = analyzer.generate_comprehensive_report()
    
    # Visualisations
    if visualize:
        print("\n📊 Génération des visualisations...")
        try:
            # Visualisations de base
            analyzer.visualize_analysis()
            
            # Visualisations avancées si disponibles
            if 'architecture' in analyzer.analysis_results or 'advanced' in analyzer.analysis_results:
                analyzer.visualize_advanced_analysis()
        except Exception as e:
            print(f"⚠️ Erreur dans la visualisation: {e}")
    
    print("\n✅ Analyse complète terminée!")
    print("=" * 60)
    
    return analyzer, report

# Fonction utilitaire pour comparer deux modèles
def compare_models(model_path1: str, model_path2: str):
    """
    Compare deux modèles HuggingFace
    """
    print("🔄 Comparaison de modèles en cours...")
    
    analyzer1 = HuggingFaceBinAnalyzer(model_path1)
    analyzer2 = HuggingFaceBinAnalyzer(model_path2)
    
    # Analyses de base
    struct1 = analyzer1.analyze_structure()
    struct2 = analyzer2.analyze_structure()
    
    comparison = {
        'model_1': model_path1,
        'model_2': model_path2,
        'size_comparison': {
            'model_1_mb': struct1['total_size_mb'],
            'model_2_mb': struct2['total_size_mb'],
            'size_ratio': struct2['total_size_mb'] / struct1['total_size_mb'] if struct1['total_size_mb'] > 0 else 0
        },
        'tensor_comparison': {
            'model_1_tensors': struct1['tensor_count'],
            'model_2_tensors': struct2['tensor_count'],
            'tensor_ratio': struct2['tensor_count'] / struct1['tensor_count'] if struct1['tensor_count'] > 0 else 0
        },
        'architecture_comparison': {}
    }
    
    # Comparaison architecturale si possible
    try:
        arch1 = analyzer1.analyze_architecture()
        arch2 = analyzer2.analyze_architecture()
        
        arch_info1 = arch1['reconstructed_architecture']
        arch_info2 = arch2['reconstructed_architecture']
        
        comparison['architecture_comparison'] = {
            'layers': (arch_info1.get('num_layers', 0), arch_info2.get('num_layers', 0)),
            'hidden_size': (arch_info1.get('hidden_size', 0), arch_info2.get('hidden_size', 0)),
            'attention_heads': (arch_info1.get('num_attention_heads', 0), arch_info2.get('num_attention_heads', 0))
        }
    except Exception as e:
        print(f"⚠️ Erreur dans la comparaison architecturale: {e}")
    
    # Rapport de comparaison
    print("\n📊 RAPPORT DE COMPARAISON")
    print("-" * 40)
    print(f"Modèle 1: {model_path1}")
    print(f"Modèle 2: {model_path2}")
    print("")
    print(f"Taille: {struct1['total_size_mb']:.1f} MB vs {struct2['total_size_mb']:.1f} MB")
    print(f"Ratio de taille: {comparison['size_comparison']['size_ratio']:.2f}x")
    print("")
    print(f"Tenseurs: {struct1['tensor_count']} vs {struct2['tensor_count']}")
    print(f"Ratio de tenseurs: {comparison['tensor_comparison']['tensor_ratio']:.2f}x")
    
    if comparison['architecture_comparison']:
        arch_comp = comparison['architecture_comparison']
        print("")
        print("Architecture:")
        print(f"  Couches: {arch_comp['layers'][0]} vs {arch_comp['layers'][1]}")
        print(f"  Hidden size: {arch_comp['hidden_size'][0]} vs {arch_comp['hidden_size'][1]}")
        print(f"  Attention heads: {arch_comp['attention_heads'][0]} vs {arch_comp['attention_heads'][1]}")
    
    return comparison

# Fonction pour analyser un batch de modèles
def batch_analysis(model_paths: List[str], output_dir: str = "./analysis_results"):
    """
    Analyse un lot de modèles et génère des rapports comparatifs
    """
    print(f"📦 Analyse en lot de {len(model_paths)} modèles...")
    
    # Création du dossier de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {}
    summaries = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n🔄 Analyse du modèle {i}/{len(model_paths)}: {model_path}")
        
        try:
            analyzer = HuggingFaceBinAnalyzer(model_path)
            
            # Analyse complète
            structure = analyzer.analyze_structure()
            tensors = analyzer.analyze_tensors(sample_size=3)
            
            # Stockage des résultats
            model_name = Path(model_path).name
            results[model_name] = {
                'path': model_path,
                'structure': structure,
                'tensors': tensors,
                'analyzer': analyzer
            }
            
            # Génération du rapport individuel
            report_path = output_path / f"{model_name}_report.txt"
            report = analyzer.generate_comprehensive_report(str(report_path))
            
            # Résumé pour comparaison
            summaries.append({
                'name': model_name,
                'size_mb': structure['total_size_mb'],
                'tensor_count': structure['tensor_count'],
                'model_type': structure['model_type']
            })
            
            print(f"✅ Analyse terminée pour {model_name}")
            
        except Exception as e:
            print(f"❌ Erreur pour {model_path}: {e}")
    
    # Génération du rapport comparatif
    if len(summaries) > 1:
        comparison_report = []
        comparison_report.append("📊 RAPPORT COMPARATIF - ANALYSE EN LOT")
        comparison_report.append("=" * 60)
        comparison_report.append("")
        
        # Tableau de comparaison
        comparison_report.append("Résumé des modèles:")
        comparison_report.append("-" * 40)
        
        for summary in sorted(summaries, key=lambda x: x['size_mb'], reverse=True):
            comparison_report.append(f"• {summary['name']}")
            comparison_report.append(f"  Taille: {summary['size_mb']:.1f} MB")
            comparison_report.append(f"  Tenseurs: {summary['tensor_count']:,}")
            comparison_report.append(f"  Type: {summary['model_type']}")
            comparison_report.append("")
        
        # Statistiques du lot
        sizes = [s['size_mb'] for s in summaries]
        tensor_counts = [s['tensor_count'] for s in summaries]
        
        comparison_report.append("Statistiques du lot:")
        comparison_report.append("-" * 40)
        comparison_report.append(f"• Nombre de modèles: {len(summaries)}")
        comparison_report.append(f"• Taille moyenne: {np.mean(sizes):.1f} MB")
        comparison_report.append(f"• Taille médiane: {np.median(sizes):.1f} MB")
        comparison_report.append(f"• Plus gros modèle: {max(sizes):.1f} MB")
        comparison_report.append(f"• Plus petit modèle: {min(sizes):.1f} MB")
        comparison_report.append(f"• Tenseurs moyens: {np.mean(tensor_counts):,.0f}")
        
        # Sauvegarde du rapport comparatif
        comparison_path = output_path / "batch_comparison_report.txt"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(comparison_report))
        
        print(f"\n📄 Rapport comparatif sauvegardé: {comparison_path}")
    
    print(f"\n✅ Analyse en lot terminée! Résultats dans: {output_path}")
    return results, summaries

# Exemples d'utilisation avancée
def demo_advanced_usage():
    """
    Démonstration des fonctionnalités avancées
    """
    print("🎯 DÉMONSTRATION DES FONCTIONNALITÉS AVANCÉES")
    print("=" * 60)
    
    # Exemple 1: Analyse complète d'un modèle
    print("\n1. Analyse complète:")
    print("analyzer, report = comprehensive_analysis('/path/to/model')")
    
    # Exemple 2: Comparaison de modèles
    print("\n2. Comparaison de modèles:")
    print("comparison = compare_models('/path/to/model1', '/path/to/model2')")
    
    # Exemple 3: Analyse en lot
    print("\n3. Analyse en lot:")
    print("models = ['/path/to/model1', '/path/to/model2', '/path/to/model3']")
    print("results, summaries = batch_analysis(models, './results')")
    
    # Exemple 4: Analyse ciblée
    print("\n4. Analyse ciblée:")
    print("analyzer = HuggingFaceBinAnalyzer('/path/to/model')")
    print("structure = analyzer.analyze_structure()")
    print("architecture = analyzer.analyze_architecture()")
    print("advanced = analyzer.analyze_advanced_patterns()")
    print("analyzer.visualize_advanced_analysis()")
    
    print("\n💡 CONSEILS D'UTILISATION:")
    print("- Commencez par comprehensive_analysis() pour une vue d'ensemble")
    print("- Utilisez compare_models() pour comparer des variantes")
    print("- batch_analysis() est idéal pour analyser plusieurs modèles")
    print("- Les visualisations aident à identifier rapidement les patterns")
    print("- Consultez les suggestions d'optimisation pour améliorer les performances")

# Pour tester l'analyseur (exemples):
# analyzer, report = comprehensive_analysis("/path/to/your/huggingface/model")
# comparison = compare_models("/path/to/model1", "/path/to/model2")
# results, summaries = batch_analysis(["/path/to/model1", "/path/to/model2"])

if __name__ == "__main__":
    demo_advanced_usage()