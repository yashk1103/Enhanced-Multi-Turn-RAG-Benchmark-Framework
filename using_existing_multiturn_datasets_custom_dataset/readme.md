# Enhanced Multi-Turn RAG Benchmark

A comprehensive benchmarking framework for evaluating embedding models on multi-turn conversational retrieval tasks. Specifically designed for **QuAC** (Question Answering in Context) and **TopiOCQA** (Topic-aware Conversational Question Answering) datasets, with extensive support for custom datasets and complete visualization suite.

## Table of Contents

1. [Overview](#overview)
2. [Core Features](#core-features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Primary Datasets](#primary-datasets)
6. [Custom Dataset Support](#custom-dataset-support)
7. [Multi-Model Benchmarking](#multi-model-benchmarking)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Visualization System](#visualization-system)
10. [Command Line Usage](#command-line-usage)
11. [Output and Results](#output-and-results)
12. [Configuration Options](#configuration-options)
13. [Advanced Usage](#advanced-usage)
14. [Performance Optimization](#performance-optimization)
15. [Troubleshooting](#troubleshooting)

## Overview

This framework addresses the critical challenge of evaluating embedding models on **multi-turn conversational scenarios** where context builds across conversation turns. Unlike single-turn retrieval benchmarks, this system evaluates how well models maintain context coherence, preserve conversation history, and adapt retrieval performance as conversations progress.

### Primary Focus

**Main Datasets**: QuAC and TopiOCQA - industry-standard multi-turn conversational datasets
**Custom Dataset Support**: Comprehensive support for JSON, CSV, and JSONL formats
**Multi-Model Evaluation**: Simultaneous benchmarking of multiple embedding models
**Complete Visualization**: Adaptive charts, heatmaps, and comprehensive PDF reports

### Key Applications

- **Conversational AI Systems**: Evaluate chatbot retrieval components
- **Multi-Turn Search**: Assess context-aware search systems
- **RAG System Optimization**: Choose optimal embedding models for conversational RAG
- **Research & Development**: Compare model performance on conversation-aware tasks

## Core Features

### Multi-Turn Conversation Analysis

**Context Preservation Metrics**:
- Measures performance maintenance across conversation turns
- Evaluates context coherence and progression
- Tracks answer relevance evolution

**Turn-by-Turn Evaluation**:
- Individual turn performance analysis
- Cumulative context impact assessment
- Conversation success rate measurement

### Comprehensive Model Support

**Default Model Library** (13 models):
- **Fast Models**: all-MiniLM-L6-v2, all-MiniLM-L12-v2
- **Balanced Models**: all-mpnet-base-v2, paraphrase-MiniLM-L6-v2, paraphrase-mpnet-base-v2
- **High Performance**: all-roberta-large-v1, gtr-t5-base, gtr-t5-large
- **Specialized**: msmarco-distilbert-base-v4, multi-qa-mpnet-base-dot-v1
- **Recent Models**: BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5, BAAI/bge-large-en-v1.5

**Custom Model Support**:
- Hugging Face model identifiers
- Local model paths
- Fine-tuned models
- Mixed model configurations

### Advanced Visualization System

**Adaptive Visualizations**:
- Single model profiles
- Multi-model comparisons
- Dataset-specific analysis
- Performance vs. time trade-offs

**Complete Reporting**:
- Interactive charts and heatmaps
- PDF report generation
- Exportable data tables
- Performance summaries

## Installation

### Requirements

```bash
pip install sentence-transformers
pip install datasets
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install tabulate
```

### Quick Installation

```bash
pip install sentence-transformers datasets pandas numpy scikit-learn matplotlib seaborn tabulate
```

### Verify Installation

```python
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
print("Installation successful!")
```

## Quick Start

### Default Benchmark (QuAC + TopiOCQA)

```bash
python enhanced_benchmark.py
```

This runs the complete benchmark with:
- All 13 default models
- QuAC and TopiOCQA datasets
- 25 conversations per evaluation
- Full visualization suite
- PDF report generation

### Quick Model Comparison

```bash
python enhanced_benchmark.py --models "all-MiniLM-L6-v2" "all-mpnet-base-v2" "BAAI/bge-base-en-v1.5"
```

### Single Dataset Focus

```bash
python enhanced_benchmark.py --datasets "quac"
python enhanced_benchmark.py --datasets "topicqa"
```

### Fast Evaluation (Testing)

```bash
python enhanced_benchmark.py \
  --models "all-MiniLM-L6-v2" \
  --max_conversations 10 \
  --no-graphs
```

## Primary Datasets

### QuAC (Question Answering in Context)

**Description**: Multi-turn question answering dataset where questions build on previous context.

**Characteristics**:
- **Source**: Wikipedia articles with conversational questions
- **Structure**: Question-answer pairs with shared context
- **Turns**: 2-10 turns per conversation
- **Context**: Rich background passages
- **Evaluation Focus**: Context-dependent question answering

**Dataset Structure**:
```python
{
  'context': 'Wikipedia article text...',
  'questions': ['Question 1', 'Question 2', ...],
  'answers': [
    {'text': 'Answer 1', 'answer_start': 123},
    {'text': 'Answer 2', 'answer_start': 456}
  ],
  'wikipedia_page_title': 'Article Title'
}
```

**Why QuAC**:
- Industry-standard benchmark for conversational QA
- Natural conversation flow with context dependency
- Realistic multi-turn scenarios
- Comprehensive evaluation coverage

### TopiOCQA (Topic-aware Conversational Question Answering)

**Description**: Topic-focused conversational dataset with explicit topic tracking.

**Characteristics**:
- **Source**: Topic-guided conversations
- **Structure**: Conversation turns with topic labels
- **Turns**: Variable length conversations
- **Topics**: Explicit topic categorization
- **Evaluation Focus**: Topic-aware context preservation

**Dataset Structure**:
```python
{
  'Conversation_no': 1,
  'Turn_no': 1,
  'Question': 'What is machine learning?',
  'Answer': 'Machine learning is...',
  'Topic': 'Artificial Intelligence',
  'Context': ['Previous context...'],
  'Gold_passage': {'text': 'Reference passage...'}
}
```

**Why TopiOCQA**:
- Topic-aware conversation evaluation
- Explicit context tracking
- Research-grade conversation quality
- Multi-domain coverage

### Dataset Loading and Processing

**Automatic Dataset Loading**:
```python
# QuAC Dataset
dataset = load_dataset("quac", trust_remote_code=True)
conversations = process_quac_data(dataset['validation'])

# TopiOCQA Dataset  
dataset = load_dataset("McGill-NLP/TopiOCQA", trust_remote_code=True)
conversations = process_topicqa_data(dataset['train'])
```

**Conversation Extraction**:
- Automatic turn grouping by conversation ID
- Context extraction and corpus building
- Relevance assessment preparation
- Multi-turn sequence validation

## Custom Dataset Support

### Supported Formats

The system provides comprehensive support for custom datasets in multiple formats:

#### JSON Format

**Complete Conversation Structure**:
```json
{
  "conversations": [
    {
      "conversation_id": "conv_1",
      "turns": [
        {
          "turn_id": 0,
          "question": "What is deep learning?",
          "answer": "Deep learning is a subset of machine learning...",
          "context": "Background context about AI..."
        },
        {
          "turn_id": 1,
          "question": "How does it differ from traditional ML?",
          "answer": "Deep learning uses neural networks...",
          "context": "Extended context building on previous..."
        }
      ],
      "topic": "Machine Learning"
    }
  ],
  "corpus": {
    "conv_1": "Comprehensive background text for retrieval evaluation..."
  }
}
```

**QA Pairs Structure**:
```json
{
  "data": [
    {
      "question": "What is neural network?",
      "answer": "A neural network is...",
      "context": "Context about neural networks...",
      "topic": "Deep Learning"
    },
    {
      "question": "How do they learn?",
      "answer": "Neural networks learn through...",
      "context": "Learning context...",
      "topic": "Deep Learning"
    }
  ]
}
```

#### CSV Format

**Multi-turn Conversation Format**:
```csv
conversation_id,turn_id,question,answer,context,topic
conv_1,0,What is AI?,Artificial Intelligence is...,AI background,AI Basics
conv_1,1,How does it work?,AI works by...,Previous context + new info,AI Basics
conv_2,0,Explain NLP,Natural Language Processing...,NLP context,NLP
conv_2,1,What are the applications?,NLP applications include...,Extended NLP context,NLP
```

**Simple QA Format**:
```csv
question,answer,context,topic
What is computer vision?,Computer vision enables machines...,CV background,Computer Vision
How does image recognition work?,Image recognition uses...,Technical CV details,Computer Vision
```

#### JSONL Format

**One JSON object per line**:
```jsonl
{"question": "What is reinforcement learning?", "answer": "RL is about agents learning...", "context": "RL background", "topic": "RL"}
{"question": "How do agents explore?", "answer": "Agents explore using...", "context": "Exploration context", "topic": "RL"}
{"conversation_id": "conv_1", "turns": [{"turn_id": 0, "question": "...", "answer": "..."}], "topic": "Advanced RL"}
```

### Custom Dataset Usage

```bash
# Single custom dataset
python enhanced_benchmark.py --datasets custom_conversations.json

# Multiple custom datasets
python enhanced_benchmark.py --datasets data1.json data2.csv data3.jsonl

# Mixed standard and custom
python enhanced_benchmark.py --datasets quac custom_data.json topicqa
```

### Field Recognition

**Flexible Field Mapping**:
- **Questions**: `question`, `Q`, `query`
- **Answers**: `answer`, `A`, `response`
- **Context**: `context`, `passage`, `background`
- **Topics**: `topic`, `category`, `subject`

**Automatic Structure Detection**:
- Multi-turn conversation format detection
- Single QA pair conversion to multi-turn
- Corpus extraction from various fields
- ID and text field automatic mapping

## Multi-Model Benchmarking

### Model Selection Strategies

**Complete Evaluation** (Default):
```bash
python enhanced_benchmark.py
# Tests all 13 default models on QuAC and TopiOCQA
```

**Speed vs. Quality Comparison**:
```bash
python enhanced_benchmark.py --models \
  "all-MiniLM-L6-v2" \
  "all-mpnet-base-v2" \
  "BAAI/bge-large-en-v1.5"
```

**Domain-Specific Models**:
```bash
python enhanced_benchmark.py --models \
  "sentence-transformers/multi-qa-mpnet-base-cos-v1" \
  "sentence-transformers/msmarco-distilbert-base-v4" \
  "BAAI/bge-base-en-v1.5"
```

**Custom Model Integration**:
```bash
python enhanced_benchmark.py --models \
  "./local/fine-tuned-model" \
  "organization/custom-model" \
  "all-mpnet-base-v2"
```

### Model Categories

**Fast Models** (< 50MB, < 30s per dataset):
- `all-MiniLM-L6-v2`: Lightweight, good performance
- `all-MiniLM-L12-v2`: Slightly larger, better quality

**Balanced Models** (50-200MB, 30-90s per dataset):
- `all-mpnet-base-v2`: Best balance of speed and quality
- `sentence-transformers/paraphrase-mpnet-base-v2`: Paraphrase-optimized

**High-Performance Models** (200MB+, 90s+ per dataset):
- `BAAI/bge-large-en-v1.5`: State-of-the-art performance
- `sentence-transformers/all-roberta-large-v1`: RoBERTa-based, high quality

**Specialized Models**:
- `sentence-transformers/multi-qa-mpnet-base-cos-v1`: QA-optimized
- `sentence-transformers/msmarco-distilbert-base-v4`: Search-optimized

## Evaluation Metrics

### Multi-Turn Specific Metrics

**Context Preservation**:
- **Definition**: Measures whether retrieval performance maintains or improves as conversation context builds
- **Calculation**: Compares final turn NDCG@10 with first turn NDCG@10
- **Score**: 1.0 if `final_turn >= first_turn * 0.8`, else 0.0
- **Interpretation**: Higher scores indicate better context utilization

**Context Coherence**:
- **Definition**: Consistency of context progression across conversation turns
- **Calculation**: Based on context length evolution and semantic consistency
- **Range**: 0.0 to 1.0 (higher is better)
- **Significance**: Measures natural conversation flow

**Turn Progression**:
- **Definition**: Quality improvement in responses across conversation turns
- **Calculation**: Compares answer relevance across consecutive turns
- **Interpretation**: Indicates whether the system learns from conversation history

### Standard Retrieval Metrics

**NDCG@10 (Normalized Discounted Cumulative Gain)**:
- **Purpose**: Measures ranking quality considering document position
- **Calculation**: DCG normalized by ideal DCG
- **Range**: 0.0 to 1.0 (higher is better)
- **Significance**: Primary ranking quality metric

**Recall@10**:
- **Purpose**: Proportion of relevant documents found in top 10 results
- **Calculation**: (Retrieved Relevant) / (Total Relevant)
- **Range**: 0.0 to 1.0 (higher is better)
- **Significance**: Coverage of relevant information

**Precision@10**:
- **Purpose**: Proportion of top 10 results that are relevant
- **Calculation**: (Retrieved Relevant) / 10
- **Range**: 0.0 to 1.0 (higher is better)
- **Significance**: Result quality assessment

**F1@10**:
- **Purpose**: Harmonic mean of Precision@10 and Recall@10
- **Calculation**: 2 * (Precision * Recall) / (Precision + Recall)
- **Range**: 0.0 to 1.0 (higher is better)
- **Significance**: Balanced precision-recall measure

**MAP (Mean Average Precision)**:
- **Purpose**: Average precision across all relevant documents
- **Calculation**: Mean of precision values at relevant document positions
- **Range**: 0.0 to 1.0 (higher is better)
- **Significance**: Overall ranking quality

**MRR (Mean Reciprocal Rank)**:
- **Purpose**: Average reciprocal rank of first relevant document
- **Calculation**: Mean of 1/(rank of first relevant document)
- **Range**: 0.0 to 1.0 (higher is better)
- **Significance**: Early precision measurement

### Conversational Metrics

**Answer Relevance**:
- **Definition**: Semantic similarity between questions and their answers
- **Calculation**: Cosine similarity of question-answer embeddings
- **Range**: -1.0 to 1.0 (typically 0.0 to 1.0)
- **Purpose**: Measures response appropriateness

**Retrieval Quality**:
- **Definition**: Maximum similarity achieved in retrieval results
- **Calculation**: Highest cosine similarity score
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Peak retrieval performance indicator

### Relevance Scoring System

**Realistic Similarity Thresholds**:
- **High Relevance** (≥ 0.50): Score 1.0 - Highly relevant documents
- **Medium Relevance** (≥ 0.35): Score 0.7 - Moderately relevant documents
- **Low Relevance** (≥ 0.20): Score 0.3 - Somewhat relevant documents
- **Not Relevant** (< 0.20): Score 0.0 - Irrelevant documents

## Visualization System

### Adaptive Visualization Engine

The system automatically generates different visualizations based on your evaluation scenario:

**Single Model, Single Dataset**:
- Detailed performance profile
- Complete metrics breakdown
- Turn-by-turn analysis

**Single Model, Multiple Datasets**:
- Dataset comparison charts
- Cross-dataset consistency analysis
- Performance variation assessment

**Multiple Models, Single Dataset**:
- Model ranking visualizations
- Performance distribution analysis
- Speed vs. quality trade-offs

**Multiple Models, Multiple Datasets**:
- Comprehensive heatmaps
- Multi-dimensional comparisons
- Top performer identification

### Visualization Types

**Metrics Overview**:
- Bar charts for key performance metrics
- Color-coded performance levels
- Precise value labels

**Individual Metric Charts**:
- Separate visualizations for each metric
- Model ranking displays
- Performance gap analysis

**Multi-Metric Comparison**:
- Grouped bar charts for side-by-side comparison
- Top 8 models focus for clarity
- Comprehensive metric coverage

**Performance vs. Time Analysis**:
- Scatter plots showing efficiency trade-offs
- Speed benchmarking results
- Optimal model identification

**Dataset-Specific Comparisons**:
- Individual dataset performance analysis
- Domain-specific model recommendations
- Cross-dataset performance consistency

**Top Performers Summary**:
- Best model for each metric
- Performance ceiling identification
- Specialized model recommendations

**Detailed Breakdown**:
- Complete metrics heatmaps (multiple models)
- Comprehensive performance profiles (single model)
- All evaluation dimensions coverage

### PDF Report Generation

**Comprehensive Reports**:
- Executive summary with key findings
- All visualizations in high resolution
- Detailed performance tables
- Model recommendations

**Adaptive Content**:
- Report structure adapts to evaluation scope
- Relevant charts and analysis only
- Professional formatting for presentations

## Command Line Usage

### Basic Commands

```bash
# Complete default benchmark
python enhanced_benchmark.py

# Custom output directory
python enhanced_benchmark.py -o ./my_results

# Adjust memory usage
python enhanced_benchmark.py --batch_size 8

# Limit conversations for testing
python enhanced_benchmark.py --max_conversations 10
```

### Model Specification

```bash
# Single model evaluation
python enhanced_benchmark.py --models "all-mpnet-base-v2"

# Compare specific models
python enhanced_benchmark.py --models \
  "all-MiniLM-L6-v2" \
  "all-mpnet-base-v2" \
  "BAAI/bge-base-en-v1.5"

# Include custom models
python enhanced_benchmark.py --models \
  "./local/fine-tuned-model" \
  "organization/custom-model" \
  "all-mpnet-base-v2"

# Fast models only
python enhanced_benchmark.py --models \
  "all-MiniLM-L6-v2" \
  "all-MiniLM-L12-v2"
```

### Dataset Selection

```bash
# Standard datasets only
python enhanced_benchmark.py --datasets "quac" "topicqa"

# Single dataset focus
python enhanced_benchmark.py --datasets "quac"
python enhanced_benchmark.py --datasets "topicqa"

# Custom datasets only
python enhanced_benchmark.py --datasets \
  "custom_conversations.json" \
  "evaluation_data.csv"

# Mixed standard and custom
python enhanced_benchmark.py --datasets \
  "quac" \
  "custom_data.json" \
  "topicqa"
```

### Output Control

```bash
# Disable visualizations (faster execution)
python enhanced_benchmark.py --no-graphs

# Custom output location
python enhanced_benchmark.py -o /path/to/results

# Complete custom configuration
python enhanced_benchmark.py \
  --models "all-mpnet-base-v2" "BAAI/bge-base-en-v1.5" \
  --datasets "quac" "custom_data.json" \
  --max_conversations 50 \
  --batch_size 32 \
  -o ./evaluation_results
```

## Output and Results

### File Structure

```
benchmark_results/
├── full_benchmark_results.json              # Complete raw results
├── comprehensive_benchmark_results.csv      # Tabular summary
├── comprehensive_rankings.json              # Model rankings by metric
├── quac_comprehensive_results.csv           # QuAC-specific results
├── topicqa_comprehensive_results.csv        # TopiOCQA-specific results
└── graphs/                                  # Visualization directory
    ├── metrics_overview.png
    ├── individual_metrics_comparison.png    # (if multiple models)
    ├── multi_metric_comparison.png          # (if multiple models)
    ├── top_performers.png                   # (if multiple models)
    ├── performance_time_analysis.png
    ├── detailed_breakdown.png
    ├── quac_comparison.png                  # (if applicable)
    ├── topicqa_comparison.png               # (if applicable)
    └── benchmark_report.pdf                 # Complete report
```

### Results Format

**full_benchmark_results.json**:
```json
{
  "benchmark_info": {
    "models_tested": 13,
    "datasets_tested": 2,
    "successful_evaluations": 26,
    "total_time_seconds": 1547.3,
    "timestamp": "2024-01-15T14:30:45",
    "max_conversations_per_eval": 25
  },
  "individual_results": [...]
}
```

**Individual Result Structure**:
```json
{
  "model": "all-mpnet-base-v2",
  "dataset": "quac",
  "conversations_evaluated": 25,
  "metrics": {
    "ndcg_10": 0.2341,
    "recall_10": 0.1875,
    "precision_10": 0.1654,
    "f1_10": 0.1757,
    "map": 0.1432,
    "mrr": 0.2876,
    "context_coherence": 0.7832,
    "answer_relevance": 0.4567,
    "context_preservation": 0.7500,
    "avg_turns_per_conv": 4.2
  },
  "timing": {
    "model_load_time": 3.2,
    "encoding_time": 45.8,
    "evaluation_time": 67.3,
    "total_time": 116.3
  }
}
```

### Console Output

```
COMPREHENSIVE MULTI-TURN RAG BENCHMARK RESULTS
===============================================
Model                           Dataset  NDCG@10  Recall@10  Precision@10  F1@10   MAP     MRR     Context_Coh  Ans_Rel  Time
all-MiniLM-L6-v2               QUAC     0.234    0.187      0.156         0.169   0.134   0.298   0.782         0.456    45.2s
all-mpnet-base-v2              QUAC     0.289    0.234      0.189         0.208   0.167   0.345   0.801         0.523    67.8s
BAAI/bge-base-en-v1.5          QUAC     0.312    0.251      0.203         0.225   0.182   0.367   0.798         0.545    72.1s

TOP PERFORMERS BY METRIC (Average across datasets)
===================================================
Metric       Best Model                               Score
NDCG@10      BAAI/bge-large-en-v1.5                  0.3456
Recall@10    sentence-transformers/all-roberta-large-v1  0.2890
Precision@10 BAAI/bge-base-en-v1.5                   0.2234
F1@10        BAAI/bge-base-en-v1.5                   0.2456
MAP          sentence-transformers/gtr-t5-large      0.1987
MRR          BAAI/bge-large-en-v1.5                  0.3789
```

## Configuration Options

### Performance Tuning

**Batch Size Configuration**:
```bash
# Memory-constrained systems
python enhanced_benchmark.py --batch_size 4

# Standard systems
python enhanced_benchmark.py --batch_size 16

# High-memory systems
python enhanced_benchmark.py --batch_size 32

# GPU-accelerated systems
python enhanced_benchmark.py --batch_size 64
```

**Conversation Limits**:
```bash
# Quick testing
python enhanced_benchmark.py --max_conversations 5

# Standard evaluation
python enhanced_benchmark.py --max_conversations 25

# Comprehensive evaluation
python enhanced_benchmark.py --max_conversations 100
```

### Model Loading Optimization

**Automatic Configuration**:
- Device detection (CPU/GPU)
- Memory optimization
- Batch processing efficiency
- Progress tracking

**Manual Optimization**:
```python
# Custom batch sizes per model
benchmark = EnhancedMultiTurnBenchmark(batch_size=16)

# Model-specific configurations
for model in large_models:
    benchmark.batch_size = 8  # Reduce for large models
```

## Advanced Usage

### Custom Evaluation Pipeline

```python
from enhanced_benchmark import EnhancedMultiTurnBenchmark

# Initialize with custom settings
benchmark = EnhancedMultiTurnBenchmark(batch_size=16)

# Set specific models
benchmark.embedding_models = [
    "all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "./local/custom-model"
]

# Set specific datasets
benchmark.datasets = ["quac", "custom_data.json"]

# Run evaluation
results = benchmark.run_full_benchmark(
    output_dir="./custom_results",
    max_conversations=50,
    create_graphs=True
)

# Access results
for result in results['individual_results']:
    print(f"{result['model']} on {result['dataset']}: "
          f"NDCG@10 = {result['metrics']['ndcg_10']:.3f}")
```

### Single Model-Dataset Evaluation

```python
# Evaluate specific combination
result = benchmark.evaluate_model_on_dataset(
    model_name="all-mpnet-base-v2",
    dataset_name="quac",
    max_conversations=30
)

# Access detailed metrics
metrics = result['metrics']
print(f"Context Preservation: {metrics['context_preservation']:.3f}")
print(f"Average Turns: {metrics['avg_turns_per_conv']:.1f}")
```

### Custom Visualization

```python
# Create visualizations for specific results
benchmark.create_visualizations(
    all_results=filtered_results,
    output_dir="./custom_viz"
)

# Generate PDF report only
benchmark._create_pdf_report(
    df=results_dataframe,
    output_dir="./reports",
    num_models=3,
    num_datasets=2
)
```

### Batch Evaluation Across Configurations

```python
# Test different configurations
model_groups = {
    "fast": ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
    "balanced": ["all-mpnet-base-v2", "paraphrase-mpnet-base-v2"],
    "high_performance": ["BAAI/bge-large-en-v1.5", "all-roberta-large-v1"]
}

for group_name, models in model_groups.items():
    benchmark.embedding_models = models
    results = benchmark.run_full_benchmark(
        output_dir=f"./results_{group_name}",
        max_conversations=25
    )
```

## Performance Optimization

### Hardware Recommendations

**Minimum Requirements**:
- 8GB RAM
- 4 CPU cores
- 20GB disk space

**Recommended Configuration**:
- 16GB+ RAM
- 8+ CPU cores or GPU
- SSD storage
- 50GB+ disk space

**Optimal Setup**:
- 32GB+ RAM
- GPU with 8GB+ VRAM
- NVMe SSD storage
- 100GB+ disk space

### Optimization Strategies

**Memory Optimization**:
```bash
# Reduce batch size for memory-constrained systems
python enhanced_benchmark.py --batch_size 4

# Limit conversations for testing
python enhanced_benchmark.py --max_conversations 10

# Single dataset evaluation
python enhanced_benchmark.py --datasets "quac"
```

**Speed Optimization**:
```bash
# Fast models only
python enhanced_benchmark.py --models "all-MiniLM-L6-v2"

# Disable visualizations for faster execution
python enhanced_benchmark.py --no-graphs

# Increase batch size (if memory allows)
python enhanced_benchmark.py --batch_size 32
```

**Comprehensive Evaluation Strategy**:
1. **Screening Phase**: Use fast models and limited conversations
2. **Selection Phase**: Test top models with standard settings
3. **Final Evaluation**: Comprehensive testing of selected models

### Batch Size Guidelines

| System RAM | CPU Batch Size | GPU Batch Size |
|------------|----------------|----------------|
| 8GB        | 4-8            | 16-32          |
| 16GB       | 8-16           | 32-64          |
| 32GB       | 16-32          | 64-128         |
| 64GB+      | 32-64          | 128-256        |

## Troubleshooting

### Common Issues

**Memory Errors**:
```bash
# Reduce batch size
python enhanced_benchmark.py --batch_size 4

# Limit evaluation scope
python enhanced_benchmark.py --max_conversations 5 --models "all-MiniLM-L6-v2"
```

**Dataset Loading Failures**:
- **QuAC/TopiOCQA**: Check internet connection and dataset availability
- **Custom datasets**: Verify file paths and format validity
- **Permission issues**: Ensure read access to dataset files

**Model Loading Issues**:
- **Hugging Face models**: Verify model names and internet connection
- **Local models**: Check file paths and model compatibility
- **Memory issues**: Use smaller models or reduce batch size

**Slow Performance**:
- **CPU systems**: Reduce batch size and use faster models
- **Large datasets**: Limit conversations or use sampling
- **Multiple models**: Process sequentially rather than simultaneously

### Debug Information

**Enable Detailed Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Dataset Validation**:
```python
# Check dataset structure
data = benchmark.load_dataset_safe("custom_data.json", max_conversations=5)
if data:
    print(f"Loaded {len(data['conversations'])} conversations")
    print(f"Sample conversation: {data['conversations'][0]['conversation_id']}")
```

**Model Testing**:
```python
# Test model loading
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer("model-name")
    embeddings = model.encode(["test sentence"])
    print(f"Model loaded successfully. Embedding shape: {embeddings.shape}")
except Exception as e:
    print(f"Model loading failed: {e}")
```

### Performance Diagnostics

**Memory Usage Monitoring**:
```python
import psutil
import os

# Monitor memory usage during evaluation
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

**Timing Analysis**:
- Model loading time: Indicates download/loading efficiency
- Encoding time: Corpus processing performance
- Evaluation time: Conversation processing speed
- Total time: Overall benchmark performance

This comprehensive framework provides everything needed for thorough evaluation of embedding models on multi-turn conversational tasks, with primary focus on industry-standard QuAC and TopiOCQA datasets while offering extensive customization capabilities for domain-specific evaluation needs.