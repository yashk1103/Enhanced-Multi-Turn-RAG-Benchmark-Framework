# Enhanced Multi-Model Multi-Turn RAG Benchmark

A comprehensive benchmarking framework for evaluating multiple embedding models on multi-turn conversational retrieval tasks. This system supports extensive customization with custom models, datasets, and provides detailed performance analysis with visualizations.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Custom Models](#custom-models)
6. [Custom Datasets](#custom-datasets)
7. [Metrics and Evaluation](#metrics-and-evaluation)
8. [Command Line Usage](#command-line-usage)
9. [Output and Results](#output-and-results)
10. [Configuration Options](#configuration-options)
11. [Visualization and Reporting](#visualization-and-reporting)
12. [Dataset Format Specifications](#dataset-format-specifications)
13. [Troubleshooting](#troubleshooting)
14. [Performance Optimization](#performance-optimization)
15. [Advanced Usage](#advanced-usage)

## Overview

This benchmarking system evaluates embedding models on multi-turn conversational retrieval tasks, measuring their ability to understand context progression, maintain coherence across turns, and retrieve relevant information. The framework supports both standard datasets (QuAC, TopiOCQA) and fully customizable datasets in multiple formats.

### Key Capabilities

- **Multi-Model Evaluation**: Test multiple embedding models simultaneously
- **Custom Model Support**: Load and evaluate custom or fine-tuned models
- **Flexible Dataset Support**: JSON, CSV, JSONL formats with automatic structure detection
- **Comprehensive Metrics**: 10+ evaluation metrics including NDCG, MAP, MRR, and custom coherence measures
- **Advanced Visualizations**: Adaptive charts, heatmaps, and PDF reports
- **Batch Processing**: Optimized for large-scale evaluations
- **Robust Error Handling**: Graceful failure handling with detailed logging

## Features

### Supported Models

**Default Models Included:**
- Fast Models: all-MiniLM-L6-v2, all-MiniLM-L12-v2
- Balanced Models: all-mpnet-base-v2, paraphrase-MiniLM-L6-v2, paraphrase-mpnet-base-v2
- High Performance: all-roberta-large-v1, gtr-t5-base, gtr-t5-large
- Specialized: msmarco-distilbert-base-v4, multi-qa-mpnet-base-dot-v1
- Recent: BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5, BAAI/bge-large-en-v1.5

**Custom Model Support:**
- Local model paths
- Hugging Face model identifiers
- Fine-tuned models
- Custom sentence transformers

### Evaluation Metrics

**Retrieval Metrics:**
- NDCG@10: Normalized Discounted Cumulative Gain
- Recall@10: Proportion of relevant documents retrieved
- Precision@10: Proportion of retrieved documents that are relevant
- F1@10: Harmonic mean of precision and recall
- MAP: Mean Average Precision
- MRR: Mean Reciprocal Rank

**Conversational Metrics:**
- Context Coherence: Consistency across conversation turns
- Turn Progression: Quality improvement across turns
- Answer Relevance: Semantic similarity between questions and answers
- Retrieval Quality: Maximum similarity scores achieved

### Dataset Support

**Built-in Datasets:**
- QuAC (Question Answering in Context)
- TopiOCQA (Topic-aware Conversational Question Answering)

**Custom Dataset Formats:**
- JSON: Nested conversation structures
- CSV: Tabular conversation data
- JSONL: Line-delimited JSON objects

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

### Quick Install

```bash
pip install sentence-transformers datasets pandas numpy scikit-learn matplotlib seaborn tabulate
```

## Quick Start

### Basic Usage

```bash
python performance.py
```

This runs the benchmark with default settings:
- All default models
- QuAC and TopiOCQA datasets
- 25 conversations per evaluation
- Full visualization suite

### Custom Models Only

```bash
python performance.py --models "sentence-transformers/all-mpnet-base-v2" "local/path/to/model"
```

### Custom Dataset Only

```bash
python performance.py --datasets "sample_dataset.json" "sample_dataset.csv"
```

### Minimal Evaluation

```bash
python performance.py --models "all-MiniLM-L6-v2" --datasets "sample_dataset.json" --max_conversations 10 --no-graphs
```

## Custom Models

### Loading Custom Models

The system accepts various model specifications:

**Hugging Face Models:**
```bash
--models "sentence-transformers/all-mpnet-base-v2"
--models "BAAI/bge-large-en-v1.5"
--models "intfloat/e5-large-v2"
```

**Local Models:**
```bash
--models "/path/to/local/model"
--models "./models/fine-tuned-model"
--models "~/models/custom-embeddings"
```

**Mixed Model Types:**
```bash
--models "sentence-transformers/all-mpnet-base-v2" "./local/model" "BAAI/bge-base-en-v1.5"
```

### Model Requirements

Custom models must be compatible with the sentence-transformers library:

1. **Directory Structure:**
   ```
   custom-model/
   ├── config.json
   ├── pytorch_model.bin
   ├── tokenizer.json
   ├── tokenizer_config.json
   └── vocab.txt
   ```

2. **Loading Test:**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('path/to/model')
   ```

## Custom Datasets

### Supported Formats

The system automatically detects and processes multiple dataset formats:

#### JSON Format

**Option 1: Direct Conversation Structure**
```json
{
  "conversations": [
    {
      "conversation_id": "conv_1",
      "turns": [
        {
          "turn_id": 0,
          "question": "What is machine learning?",
          "answer": "Machine learning is a subset of AI...",
          "context": "Background information about AI..."
        },
        {
          "turn_id": 1,
          "question": "How does it work?",
          "answer": "It works by training algorithms...",
          "context": "Previous context plus new information..."
        }
      ],
      "topic": "Machine Learning Basics"
    }
  ],
  "corpus": {
    "conv_1": "Comprehensive background text for retrieval evaluation..."
  }
}
```

**Option 2: QA Pairs Structure**
```json
{
  "data": [
    {
      "question": "What is deep learning?",
      "answer": "Deep learning uses neural networks...",
      "context": "Context information...",
      "topic": "AI"
    },
    {
      "question": "What are the applications?",
      "answer": "Applications include computer vision...",
      "context": "Extended context...",
      "topic": "AI"
    }
  ]
}
```

#### CSV Format

**Option 1: Multi-turn Conversation Format**
```csv
conversation_id,turn_id,question,answer,context,topic
conv_1,0,What is NLP?,Natural Language Processing...,Background on NLP,NLP Basics
conv_1,1,How does it differ from ML?,NLP focuses on language...,Previous context + new info,NLP Basics
conv_2,0,Explain transformers,Transformers are neural networks...,Context about transformers,Deep Learning
conv_2,1,What is attention?,Attention mechanisms allow...,Extended context,Deep Learning
```

**Option 2: Simple QA Format**
```csv
question,answer,context,topic
What is computer vision?,Computer vision enables...,CV background,Computer Vision
How does image recognition work?,Image recognition uses...,Technical details,Computer Vision
```

**Option 3: With Separate Corpus**
```csv
conversation_id,turn_id,question,answer,context,topic,corpus_id,corpus_text
conv_1,0,Question 1,Answer 1,Context 1,Topic A,corp_1,Full corpus text for retrieval...
conv_1,1,Question 2,Answer 2,Context 2,Topic A,corp_1,
```

#### JSONL Format

**One JSON object per line:**
```jsonl
{"question": "What is reinforcement learning?", "answer": "RL is about learning through interaction...", "context": "Background on RL", "topic": "RL"}
{"question": "How do agents learn?", "answer": "Agents learn through rewards and punishments...", "context": "Extended context", "topic": "RL"}
{"conversation_id": "conv_3", "turns": [{"turn_id": 0, "question": "...", "answer": "..."}], "topic": "Advanced RL"}
```

### Field Mapping

The system recognizes multiple field names for flexibility:

**Question Fields:** `question`, `Q`, `query`
**Answer Fields:** `answer`, `A`, `response`
**Context Fields:** `context`, `passage`, `background`
**Topic Fields:** `topic`, `category`, `subject`

### Dataset Validation

Datasets are automatically validated for:
- Minimum 2 turns per conversation
- Non-empty questions and answers
- Proper conversation structure
- Valid encoding (UTF-8)

## Metrics and Evaluation

### Retrieval Metrics Calculation

**NDCG@10 (Normalized Discounted Cumulative Gain)**
- Measures ranking quality of retrieved documents
- Accounts for position bias in rankings
- Range: 0.0 to 1.0 (higher is better)

**Precision@10**
- Proportion of top-10 retrieved documents that are relevant
- Formula: (Relevant Retrieved) / 10
- Range: 0.0 to 1.0

**Recall@10**
- Proportion of all relevant documents found in top-10
- Formula: (Relevant Retrieved) / (Total Relevant)
- Range: 0.0 to 1.0

**F1@10**
- Harmonic mean of Precision@10 and Recall@10
- Formula: 2 * (Precision * Recall) / (Precision + Recall)
- Balances precision and recall

**MAP (Mean Average Precision)**
- Average of precision values at each relevant document position
- Considers the ranking of all relevant documents
- Range: 0.0 to 1.0

**MRR (Mean Reciprocal Rank)**
- Average of reciprocal ranks of first relevant document
- Formula: 1/rank of first relevant document
- Range: 0.0 to 1.0

### Conversational Metrics

**Context Coherence**
- Measures consistency of context across conversation turns
- Higher values indicate better context maintenance
- Calculated based on context length progression

**Turn Progression**
- Evaluates improvement in answer quality across turns
- Measures whether later turns build effectively on earlier ones
- Range: 0.0 to 1.0

**Answer Relevance**
- Semantic similarity between questions and their answers
- Uses cosine similarity of embeddings
- Range: -1.0 to 1.0 (typically 0.0 to 1.0)

### Relevance Thresholds

The system uses realistic similarity thresholds:
- **High Relevance**: ≥ 0.50 (score: 1.0)
- **Medium Relevance**: ≥ 0.35 (score: 0.7)
- **Low Relevance**: ≥ 0.20 (score: 0.3)
- **Not Relevant**: < 0.20 (score: 0.0)

## Command Line Usage

### Basic Commands

```bash
# Default benchmark (all models, standard datasets)
python performance.py

# Specify output directory
python performance.py -o ./results

# Custom batch size for memory optimization
python performance.py --batch_size 8

# Limit conversations for faster testing
python performance.py --max_conversations 10
```

### Model Selection

```bash
# Single model
python performance.py --models "all-MiniLM-L6-v2"

# Multiple specific models
python performance.py --models "all-MiniLM-L6-v2" "all-mpnet-base-v2" "BAAI/bge-base-en-v1.5"

# Local and remote models
python performance.py --models "./local/model" "sentence-transformers/all-mpnet-base-v2"
```

### Dataset Selection

```bash
# Standard datasets only
python performance.py --datasets "quac" "topicqa"

# Custom datasets only
python performance.py --datasets "data.json" "conversations.csv"

# Mixed datasets
python performance.py --datasets "quac" "custom_data.json" "sample.csv"

# Single custom dataset
python performance.py --datasets "my_conversations.jsonl"
```

### Output Control

```bash
# Disable visualizations (faster execution)
python performance.py --no-graphs

# Custom output directory
python performance.py -o /path/to/results

# Minimal fast evaluation
python performance.py --models "all-MiniLM-L6-v2" --datasets "sample.json" --max_conversations 5 --no-graphs
```

### Complete Example

```bash
python performance.py \
  --models "sentence-transformers/all-mpnet-base-v2" "BAAI/bge-base-en-v1.5" "./local/model" \
  --datasets "custom_conversations.json" "evaluation_data.csv" \
  --max_conversations 50 \
  --batch_size 16 \
  -o ./benchmark_results_2024
```

## Output and Results

### Directory Structure

```
benchmark_results/
├── full_benchmark_results.json          # Complete raw results
├── comprehensive_benchmark_results.csv  # Tabular summary
├── comprehensive_rankings.json          # Model rankings by metric
├── quac_comprehensive_results.csv       # Dataset-specific results
├── topicqa_comprehensive_results.csv    # Dataset-specific results
└── graphs/                              # Visualization directory
    ├── metrics_overview.png
    ├── individual_metrics_comparison.png
    ├── multi_metric_comparison.png
    ├── top_performers.png
    ├── performance_time_analysis.png
    ├── detailed_breakdown.png
    ├── quac_comparison.png
    ├── topicqa_comparison.png
    └── benchmark_report.pdf
```

### Results Files

**full_benchmark_results.json**
- Complete evaluation results with all metrics
- Timing information for each evaluation
- Model and dataset metadata
- Individual conversation results

**comprehensive_benchmark_results.csv**
- Tabular format for easy analysis
- All metrics for each model-dataset combination
- Suitable for further statistical analysis

**comprehensive_rankings.json**
- Model rankings for each metric
- Average scores across datasets
- Top 10 performers per metric

### Performance Metrics Output

**Console Output Format:**
```
COMPREHENSIVE MULTI-TURN RAG BENCHMARK RESULTS
===============================================
Model                           Dataset  NDCG@10  Recall@10  Precision@10  F1@10   MAP     MRR     Context_Coh  Ans_Rel  Time
all-MiniLM-L6-v2               QUAC     0.245    0.187      0.156         0.169   0.134   0.298   0.782         0.456    45.2s
all-mpnet-base-v2              QUAC     0.289    0.234      0.189         0.208   0.167   0.345   0.801         0.523    67.8s
BAAI/bge-base-en-v1.5          QUAC     0.312    0.251      0.203         0.225   0.182   0.367   0.798         0.545    72.1s
```

**Top Performers Summary:**
```
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

### Batch Size Optimization

**Memory-Constrained Systems:**
```bash
python performance.py --batch_size 4
```

**High-Memory Systems:**
```bash
python performance.py --batch_size 32
```

**GPU Optimization:**
```bash
python performance.py --batch_size 64  # For systems with GPU
```

### Conversation Limits

**Quick Testing:**
```bash
python performance.py --max_conversations 5
```

**Comprehensive Evaluation:**
```bash
python performance.py --max_conversations 100
```

**Full Dataset Evaluation:**
```bash
python performance.py --max_conversations 500
```

### Model Loading Configuration

Models are loaded with optimized settings:
- Automatic device detection (CPU/GPU)
- Normalized embeddings for cosine similarity
- Batch processing for efficiency
- Progress tracking for long operations

## Visualization and Reporting

### Adaptive Visualization System

The system generates different visualizations based on the evaluation scenario:

**Single Model, Single Dataset:**
- Detailed metrics profile
- Performance breakdown
- Time analysis

**Single Model, Multiple Datasets:**
- Dataset comparison charts
- Performance by dataset
- Consistency analysis

**Multiple Models, Single Dataset:**
- Model comparison charts
- Ranking visualizations
- Performance distribution

**Multiple Models, Multiple Datasets:**
- Comprehensive heatmaps
- Top performers analysis
- Multi-dimensional comparisons

### Visualization Types

**Metrics Overview**
- Bar charts for key metrics
- Color-coded performance levels
- Value labels for precision

**Individual Metric Charts**
- Separate charts for each metric
- Model ranking visualization
- Performance gaps analysis

**Multi-Metric Comparison**
- Grouped bar charts
- Side-by-side comparisons
- Top 8 models focus

**Performance vs Time Analysis**
- Scatter plots showing trade-offs
- Efficiency quadrant analysis
- Speed benchmarking

**Detailed Breakdown**
- Comprehensive heatmaps (multiple models)
- Complete metrics profiles (single model)
- All evaluation dimensions

**PDF Report Generation**
- Title page with summary statistics
- All charts in high resolution
- Print-ready format
- Comprehensive documentation

### Customizing Visualizations

**Disable Visualizations:**
```bash
python performance.py --no-graphs
```

**Custom Output Directory:**
```bash
python performance.py -o ./custom_results
```

The visualization system automatically adapts to your evaluation scenario, ensuring relevant and informative charts are generated regardless of the number of models or datasets being evaluated.

## Dataset Format Specifications

### JSON Schema

**Complete Conversation Format:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "conversations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "conversation_id": {"type": "string"},
          "turns": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "turn_id": {"type": "integer"},
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "context": {"type": "string"}
              },
              "required": ["turn_id", "question", "answer"]
            },
            "minItems": 2
          },
          "topic": {"type": "string"}
        },
        "required": ["conversation_id", "turns"]
      }
    },
    "corpus": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    }
  },
  "required": ["conversations"]
}
```

### CSV Column Specifications

**Required Columns (Multi-turn):**
- `conversation_id`: Unique identifier for each conversation
- `turn_id`: Sequential turn number within conversation (0, 1, 2, ...)
- `question`: The question text
- `answer`: The answer text

**Optional Columns:**
- `context`: Background context for the turn
- `topic`: Conversation topic or category
- `corpus_id`: Identifier for corpus documents
- `corpus_text`: Full text for retrieval evaluation

**Alternative Column Names:**
- Question: `question`, `Q`, `query`
- Answer: `answer`, `A`, `response`
- Context: `context`, `passage`, `background`

### JSONL Format Specifications

Each line must contain a valid JSON object:

**Single Turn Format:**
```json
{"question": "...", "answer": "...", "context": "...", "topic": "..."}
```

**Conversation Format:**
```json
{"conversation_id": "conv_1", "turns": [...], "topic": "..."}
```

**Mixed Format Support:**
Lines can contain different formats, and the system will automatically group single turns into conversations.

### Data Quality Requirements

**Minimum Requirements:**
- At least 2 turns per conversation
- Non-empty questions and answers
- UTF-8 encoding
- Valid JSON/CSV format

**Recommended:**
- 3-10 turns per conversation for meaningful multi-turn analysis
- Context information for retrieval evaluation
- Topic labels for categorization
- Corpus documents for comprehensive retrieval testing

**Optimal:**
- 5-15 turns per conversation
- Rich context with progression across turns
- Diverse topics and question types
- Comprehensive corpus with varied document lengths

## Troubleshooting

### Common Issues

**Memory Errors:**
```bash
# Reduce batch size
python performance.py --batch_size 4

# Limit conversations
python performance.py --max_conversations 10
```

**Model Loading Failures:**
- Verify model path exists
- Check internet connection for remote models
- Ensure model is compatible with sentence-transformers
- Try different model names or paths

**Dataset Loading Issues:**
- Verify file exists and is readable
- Check file encoding (must be UTF-8)
- Validate JSON/CSV format
- Ensure minimum 2 turns per conversation

**Slow Performance:**
- Use smaller batch sizes on CPU
- Reduce max_conversations for testing
- Disable visualizations with --no-graphs
- Use faster models for initial testing

### Error Messages

**"Failed to load dataset":**
- Check file path and permissions
- Verify file format and structure
- Look for encoding issues

**"Error encoding query":**
- Model loading failed
- Memory insufficient
- Try smaller batch size

**"No valid conversations found":**
- Dataset doesn't meet minimum requirements
- All conversations have < 2 turns
- Data format not recognized

### Debug Mode

Enable detailed logging by modifying the script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

**CPU Optimization:**
- Use batch_size 8-16
- Limit to 2-3 models simultaneously
- Use smaller, faster models for initial testing

**GPU Optimization:**
- Increase batch_size to 32-64
- Load models on GPU if available
- Use mixed precision if supported

**Memory Management:**
- Monitor memory usage during evaluation
- Use gc.collect() between model evaluations
- Reduce max_conversations if needed

## Performance Optimization

### Hardware Recommendations

**Minimum Requirements:**
- 8GB RAM
- 4 CPU cores
- 10GB disk space

**Recommended:**
- 16GB+ RAM
- 8+ CPU cores or GPU
- 50GB+ disk space for results

**Optimal:**
- 32GB+ RAM
- GPU with 8GB+ VRAM
- SSD storage for faster I/O

### Optimization Strategies

**For Large-Scale Evaluations:**
1. Use GPU acceleration
2. Increase batch sizes
3. Process datasets in parallel
4. Cache model embeddings

**For Memory-Constrained Systems:**
1. Reduce batch sizes
2. Limit concurrent models
3. Use smaller models first
4. Process subsets of data

**For Time-Constrained Evaluations:**
1. Disable visualizations initially
2. Use faster models for screening
3. Limit conversations per evaluation
4. Run specific model-dataset combinations

### Batch Size Guidelines

| System RAM | Recommended Batch Size |
|------------|------------------------|
| 8GB        | 4-8                    |
| 16GB       | 8-16                   |
| 32GB       | 16-32                  |
| 64GB+      | 32-64                  |

### Model Loading Optimization

**Pre-download Models:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model-name')
model.save('./local/model-name')
```

**Use Local Models:**
```bash
python performance.py --models "./local/model-name"
```

## Advanced Usage

### Custom Evaluation Pipeline

```python
from performance import EnhancedMultiTurnBenchmark

# Initialize benchmark
benchmark = EnhancedMultiTurnBenchmark(batch_size=16)

# Set custom models
benchmark.embedding_models = [
    "path/to/custom/model1",
    "path/to/custom/model2"
]

# Set custom datasets
benchmark.datasets = [
    "custom_dataset1.json",
    "custom_dataset2.csv"
]

# Run evaluation
results = benchmark.run_full_benchmark(
    output_dir="./custom_results",
    max_conversations=100,
    create_graphs=True
)
```

### Single Model-Dataset Evaluation

```python
# Evaluate specific combination
result = benchmark.evaluate_model_on_dataset(
    model_name="custom/model",
    dataset_name="data.json",
    max_conversations=50
)
```

### Custom Metrics Calculation

```python
# Access individual metrics
ndcg = benchmark.calculate_ndcg_at_k(relevance_scores, k=10)
precision = benchmark.calculate_precision_at_k(retrieved_relevant, k=10)
f1 = benchmark.calculate_f1_score(precision, recall)
```

### Extending the Framework

**Add Custom Metrics:**
```python
def custom_metric(self, conversation_results):
    # Implement custom evaluation logic
    return metric_value

# Add to metrics calculation
EnhancedMultiTurnBenchmark.custom_metric = custom_metric
```

**Custom Dataset Loaders:**
```python
def _load_custom_format(self, dataset_path, max_conversations):
    # Implement custom loading logic
    return {'conversations': conversations, 'corpus': corpus}

# Add to dataset loading
EnhancedMultiTurnBenchmark._load_custom_format = _load_custom_format
```

### Integration with ML Pipelines

**Export Results for Analysis:**
```python
import pandas as pd

# Load results
with open('results/full_benchmark_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results['individual_results'])

# Perform custom analysis
top_models = df.groupby('model')['metrics'].apply(lambda x: x['ndcg_10']).sort_values(ascending=False)
```

**Automated Model Selection:**
```python
def select_best_model(results, metric='ndcg_10', dataset=None):
    filtered_results = results['individual_results']
    if dataset:
        filtered_results = [r for r in filtered_results if r['dataset'] == dataset]
    
    best_result = max(filtered_results, key=lambda x: x['metrics'][metric])
    return best_result['model']

best_model = select_best_model(results, 'f1_10', 'custom_dataset')
```

This comprehensive framework provides everything needed for thorough evaluation of embedding models on multi-turn conversational tasks, with extensive customization options and detailed analysis capabilities.