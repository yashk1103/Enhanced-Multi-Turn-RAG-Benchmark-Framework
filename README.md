# Multi-Turn RAG Evaluation Suite

A comprehensive collection of benchmarking frameworks for evaluating Retrieval-Augmented Generation (RAG) systems on multi-turn conversational scenarios.

## 📁 Repository Structure

```
├── Custom_model_custom_dataset/
│   ├── performance.py
│   ├── readme.md
│   └── sample_dataset.*
├── domain_specific_using_external_corporas/
│   ├── enhanced_rag_benchmark.py
│   └── readme.md
├── single_turn_to_multi_turn/
│   ├── multiturn_evaluation.py
│   └── readme.md
└── using_existing_multiturn_datasets_custom_dataset/
    ├── performance.py
    └── readme.md
```

## 🚀 Framework Overview

### 1. **Custom Model & Dataset Benchmark**
- **File**: `Custom_model_custom_dataset/performance.py`
- **Focus**: Multi-model evaluation with custom datasets (JSON/CSV/JSONL)
- **Features**: 13 default models, flexible data formats, comprehensive visualizations
- **Use Case**: Testing custom models on proprietary conversational datasets

### 2. **Domain-Specific External Corpus**
- **File**: `domain_specific_using_external_corporas/enhanced_rag_benchmark.py`  
- **Focus**: Domain-aware evaluation with external document collections
- **Features**: External corpus integration, domain matching, specialized metrics
- **Use Case**: Enterprise/domain-specific RAG system evaluation

### 3. **Single-to-Multi Turn Conversion**
- **File**: `single_turn_to_multi_turn/multiturn_evaluation.py`
- **Focus**: Synthetic multi-turn conversation generation from single-turn datasets
- **Features**: BEIR dataset support, automatic conversation synthesis, context preservation metrics
- **Use Case**: Converting existing single-turn benchmarks to multi-turn scenarios

### 4. **Existing Multi-Turn Datasets**
- **File**: `using_existing_multiturn_datasets_custom_dataset/performance.py`
- **Focus**: Evaluation using standard multi-turn datasets (QuAC, TopiOCQA)
- **Features**: Standard dataset integration, custom model support, detailed metrics
- **Use Case**: Benchmarking against established multi-turn conversation datasets

## 🛠️ Quick Start

Choose the framework that best fits your use case:

```bash
# Custom datasets with multiple models
python Custom_model_custom_dataset/performance.py --datasets "your_data.json"

# Domain-specific evaluation with external corpus
python domain_specific_using_external_corporas/enhanced_rag_benchmark.py --corpus "domain_docs.json"

# Convert single-turn to multi-turn
python single_turn_to_multi_turn/multiturn_evaluation.py --dataset "scifact"

# Standard multi-turn datasets
python using_existing_multiturn_datasets_custom_dataset/performance.py
```

## 📊 Common Features

- **Multiple Embedding Models**: Support for 10+ state-of-the-art models
- **Comprehensive Metrics**: NDCG@10, Recall@10, F1@10, MAP, MRR, context coherence
- **Adaptive Visualizations**: Charts and reports that adapt to your evaluation scenario
- **Custom Model Support**: Load local or Hugging Face models
- **Flexible Output**: JSON results, CSV tables, PDF reports

## 📋 Requirements

```bash
pip install sentence-transformers datasets pandas numpy scikit-learn matplotlib seaborn tabulate beir
```

## 📖 Documentation

Each framework includes detailed documentation in its respective `readme.md` file with:
- Specific installation instructions
- Dataset format specifications  
- Advanced configuration options
- Troubleshooting guides
- Performance optimization tips

---

*Choose the framework that matches your evaluation needs and refer to the specific README for detailed usage instructions.*