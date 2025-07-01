# Ultimate Multi-Model Multi-Turn RAG Benchmark

A comprehensive, domain-aware benchmarking framework for evaluating embedding models on multi-turn conversational retrieval tasks. Features **custom model support**, **flexible dataset formats**, and **external corpus integration** for domain-specific evaluation at scale.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Datasets](#core-datasets)
6. [External Corpus Integration](#external-corpus-integration)
7. [Domain-Specific Evaluation](#domain-specific-evaluation)
8. [Custom Dataset Support](#custom-dataset-support)
9. [Multi-Model Benchmarking](#multi-model-benchmarking)
10. [Command Line Usage](#command-line-usage)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Visualization System](#visualization-system)
13. [Output and Results](#output-and-results)
14. [Domain Compatibility Guidelines](#domain-compatibility-guidelines)
15. [Advanced Usage](#advanced-usage)
16. [Performance Optimization](#performance-optimization)
17. [Troubleshooting](#troubleshooting)

download corpora from the https://drive.google.com/drive/folders/1OOwsyjigUn5J-bnVBP4Y3mvmPEV6trmq?usp=sharing

## Overview

This **Ultimate RAG Benchmark** addresses the critical need for **domain-aware multi-turn conversational evaluation**. Unlike standard benchmarks that use generic corpora, this system enables evaluation with **domain-specific external corpora**, providing realistic assessment of how embedding models perform in specialized domains like finance, healthcare, legal, technical documentation, and more.

### Primary Innovation: External Corpus Integration

**Core Capability**: Load domain-specific corpora to evaluate model performance in specialized contexts
**Use Case**: Test how well models retrieve relevant information from domain-specific document collections
**Impact**: Get realistic performance metrics for production RAG systems in specific industries

### Key Applications

- **Domain-Specific RAG Systems**: Finance, Legal, Healthcare, Technical Support
- **Enterprise Document Search**: Company knowledge bases, technical documentation
- **Specialized Chatbots**: Industry-specific conversational AI systems
- **Research & Development**: Model selection for domain-specific applications
- **Production Deployment**: Real-world performance assessment

## Key Features

### 🎯 Domain-Aware Evaluation
- **External Corpus Support**: Load domain-specific document collections
- **Flexible Corpus Formats**: JSON, JSONL, CSV with automatic field detection
- **Scalable Processing**: Handle large corpora with configurable document limits
- **Domain Matching**: Combine appropriate datasets with relevant corpora

### 🤖 Comprehensive Model Support
- **13 Default Models**: From fast (MiniLM) to high-performance (BGE-large)
- **Custom Models**: Local paths, Hugging Face identifiers, fine-tuned models
- **Batch Processing**: Optimized for efficient large-scale evaluation
- **Memory Management**: Adaptive batch sizing for different hardware

### 📊 Advanced Visualization & Analysis
- **Corpus Impact Analysis**: Visualize how external corpora affect performance
- **Domain-Specific Charts**: Performance breakdowns by corpus type
- **Adaptive Reporting**: Charts adapt to single/multiple models/datasets/corpora
- **PDF Reports**: Comprehensive professional reports with all visualizations

### 🔧 Flexible Data Pipeline
- **Multiple Dataset Formats**: JSON, CSV, JSONL with intelligent structure detection
- **Conversation Assembly**: Automatic multi-turn conversation creation from Q&A pairs
- **Field Mapping**: Flexible field recognition for various data schemas
- **Validation**: Automatic data quality checks and format validation

## Installation

### Requirements

```bash
pip install sentence-transformers datasets pandas numpy scikit-learn matplotlib seaborn tabulate
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
print("Ultimate RAG Benchmark ready!")
```

## Quick Start

### Default Benchmark (QuAC + TopiOCQA)

```bash
python enhanced_rag_benchmark.py
```

### With External Corpus

```bash
python enhanced_rag_benchmark.py --corpus ./domain_corpus.json
```

### Domain-Specific Evaluation

```bash
python enhanced_rag_benchmark.py \
  --datasets "quac" \
  --corpus "./finance_corpus.json" \
  --models "BAAI/bge-base-en-v1.5"
```

### Complete Custom Setup

```bash
python enhanced_rag_benchmark.py \
  --models "all-mpnet-base-v2" "BAAI/bge-large-en-v1.5" \
  --datasets "custom_conversations.json" \
  --corpus "./domain_docs.jsonl" \
  --max_corpus_docs 5000 \
  --max_conversations 50 \
  -o ./domain_evaluation_results
```

## Core Datasets

### QuAC (Question Answering in Context)

**Description**: Multi-turn question answering based on Wikipedia articles
**Characteristics**:
- **Domain**: General knowledge (Wikipedia)
- **Structure**: 2-10 turns per conversation
- **Context**: Rich background passages
- **Size**: ~13K conversations from validation set

**Compatible Corpora**: General knowledge, Wikipedia-based, educational content

### TopiOCQA (Topic-aware Conversational Question Answering) 

**Description**: Topic-focused conversational dataset with explicit topic tracking
**Characteristics**:
- **Domain**: Multi-domain with topic labels
- **Structure**: Variable-length conversations
- **Context**: Topic-aware context progression
- **Size**: Research-grade conversation quality

**Compatible Corpora**: Multi-domain, topic-categorized document collections

## External Corpus Integration

### Supported Corpus Formats

The system provides **universal corpus loading** with intelligent format detection:

#### JSON Format

**Direct Mapping**:
```json
{
  "doc_001": "Document content about financial regulations...",
  "doc_002": "Healthcare policy guidelines...",
  "doc_003": "Technical specifications for cloud architecture..."
}
```

**Structured Documents**:
```json
[
  {
    "_id": "fin_001",
    "title": "Financial Compliance Guide",
    "text": "Comprehensive guide to financial regulations...",
    "domain": "finance",
    "metadata": {"year": 2024, "source": "regulatory_body"}
  },
  {
    "document_id": "tech_001", 
    "content": "Cloud architecture best practices...",
    "category": "technical"
  }
]
```

#### JSONL Format

**One document per line**:
```jsonl
{"_id": "doc1", "text": "Financial risk assessment methodologies..."}
{"document_id": "doc2", "content": "Healthcare data privacy regulations..."}
{"id": "doc3", "passage": "Legal precedents in intellectual property..."}
```

#### CSV Format

**Simple Structure**:
```csv
document_id,text,domain,metadata
doc_001,"Financial compliance regulations...",finance,"regulatory"
doc_002,"Healthcare privacy guidelines...",healthcare,"policy"
doc_003,"Technical API documentation...",technology,"reference"
```

**Extended Structure**:
```csv
_id,title,content,description,category,source
fin_001,"Risk Management","Detailed risk assessment...","Guide for analysts",finance,"internal"
tech_001,"API Docs","REST API specifications...","Developer reference",technology,"public"
```

### Automatic Field Detection

**Text Fields** (priority order):
- `text`, `content`, `passage`, `document`, `body`
- `description`, `answer`, `response`, `message`, `data`

**ID Fields** (priority order):
- `_id`, `document_id`, `id`, `doc_id`, `idx`, `index`

**Metadata Fields**:
- `title`, `category`, `domain`, `source`, `metadata`

### Corpus Loading Examples

```bash
# JSON corpus with 10K document limit
python enhanced_rag_benchmark.py --corpus corpus.json --max_corpus_docs 10000

# JSONL corpus (no limit)
python enhanced_rag_benchmark.py --corpus documents.jsonl

# CSV corpus with specific models
python enhanced_rag_benchmark.py \
  --corpus financial_docs.csv \
  --models "BAAI/bge-base-en-v1.5" \
  --datasets "quac"
```

## Domain-Specific Evaluation

### Available Domain Corpora

| **Corpus** | **Domain** | **Documents** | **Passages** | **Best For** |
|------------|------------|---------------|--------------|--------------|
| **ClapNQ** | Wikipedia/General | 4,293 | 183,408 | General QA, Educational content |
| **Cloud** | Technical Documentation | 57,638 | 61,022 | Technical support, DevOps, Cloud services |
| **FiQA** | Finance | 7,661 | 49,607 | Financial QA, Investment, Banking |
| **Govt** | Government/Policy | 8,578 | 72,422 | Legal, Regulatory, Public policy |

### Domain Matching Strategy

**✅ Recommended Combinations**:

```bash
# General Knowledge Evaluation
python enhanced_rag_benchmark.py --datasets "quac" --corpus "clapnq_corpus.json"

# Technical Documentation Testing  
python enhanced_rag_benchmark.py --datasets "topicqa" --corpus "cloud_docs.jsonl"

# Financial Domain Assessment
python enhanced_rag_benchmark.py --datasets "custom_finance_qa.json" --corpus "fiqa_corpus.json"

# Government/Legal Evaluation
python enhanced_rag_benchmark.py --datasets "legal_conversations.csv" --corpus "govt_corpus.json"
```

**⚠️ Domain Compatibility Warning**:

**AVOID mismatched domain combinations** - they will produce poor results due to semantic mismatch:

```bash
# ❌ BAD: General knowledge questions with finance corpus
--datasets "quac" --corpus "fiqa_corpus.json"  # Poor performance expected

# ❌ BAD: Technical questions with government corpus  
--datasets "tech_support_qa.json" --corpus "govt_corpus.json"  # Low relevance

# ❌ BAD: Finance questions with medical corpus
--datasets "finance_qa.csv" --corpus "medical_corpus.jsonl"  # No semantic overlap
```

**Why Domain Matching Matters**:
- **Semantic Alignment**: Models perform best when query and corpus domains align
- **Vocabulary Overlap**: Domain-specific terminology must be present in both
- **Context Relevance**: Retrieval quality depends on topical relevance
- **Realistic Assessment**: Production systems operate within specific domains

### Creating Domain-Specific Evaluations

**Step 1: Identify Your Domain**
```bash
# Example domains: healthcare, legal, finance, technical, academic, retail
DOMAIN="healthcare"
```

**Step 2: Prepare Domain Corpus**
```bash
# Collect domain-specific documents
# Format: JSON/JSONL/CSV with text content
# Size: 1000+ documents recommended for meaningful evaluation
```

**Step 3: Create or Adapt Dataset**
```bash
# Option A: Use existing dataset from similar domain
--datasets "topicqa"  # Multi-domain, works with most corpora

# Option B: Create domain-specific conversations
# Format conversations as multi-turn Q&A in your domain
```

**Step 4: Run Domain Evaluation**
```bash
python enhanced_rag_benchmark.py \
  --datasets "domain_conversations.json" \
  --corpus "domain_corpus.jsonl" \
  --models "BAAI/bge-base-en-v1.5" "all-mpnet-base-v2" \
  --max_conversations 50 \
  -o "./domain_evaluation_results"
```

## Custom Dataset Support

### Supported Dataset Formats

#### Multi-Turn Conversation Format

**Complete Structure**:
```json
{
  "conversations": [
    {
      "conversation_id": "conv_001",
      "turns": [
        {
          "turn_id": 0,
          "question": "What are the new financial regulations?",
          "answer": "The new regulations focus on...",
          "context": "Background about recent policy changes"
        },
        {
          "turn_id": 1, 
          "question": "How do these affect small businesses?",
          "answer": "Small businesses will need to...",
          "context": "Previous context plus new information"
        }
      ],
      "topic": "Financial Regulation"
    }
  ]
}
```

#### CSV Multi-Turn Format

```csv
conversation_id,turn_id,question,answer,context,topic
conv_1,0,"What is cloud architecture?","Cloud architecture refers to...","Technical background",Technology
conv_1,1,"What are the main components?","The main components include...","Extended technical context",Technology
conv_2,0,"How does risk management work?","Risk management involves...","Financial context",Finance
conv_2,1,"What are the key metrics?","Key metrics include...","Risk assessment context",Finance
```

#### JSONL Conversation Format

```jsonl
{"conversation_id": "c1", "turns": [{"turn_id": 0, "question": "...", "answer": "..."}, {"turn_id": 1, "question": "...", "answer": "..."}], "topic": "domain"}
{"question": "Single question", "answer": "Single answer", "context": "Context", "topic": "domain"}
{"Q": "Alternative format", "A": "Alternative answer", "context": "Context"}
```

### Field Recognition System

**Flexible Field Mapping**:
- **Questions**: `question`, `Q`, `query`
- **Answers**: `answer`, `A`, `response`  
- **Context**: `context`, `passage`, `background`
- **Topics**: `topic`, `category`, `subject`, `domain`
- **IDs**: `conversation_id`, `turn_id`, `id`

### Custom Dataset Usage

```bash
# Single custom dataset
python enhanced_rag_benchmark.py --datasets "my_conversations.json"

# Multiple custom datasets  
python enhanced_rag_benchmark.py --datasets "dataset1.csv" "dataset2.jsonl" "dataset3.json"

# Mixed standard and custom
python enhanced_rag_benchmark.py --datasets "quac" "my_domain_qa.json" "topicqa"
```

## Multi-Model Benchmarking

### Default Model Library (13 Models)

**Fast Models** (< 50MB, < 30s per evaluation):
- `all-MiniLM-L6-v2`: Lightweight, good performance
- `all-MiniLM-L12-v2`: Slightly larger, better quality

**Balanced Models** (50-200MB, 30-90s per evaluation):
- `all-mpnet-base-v2`: Best balance of speed and quality
- `sentence-transformers/paraphrase-MiniLM-L6-v2`: Paraphrase-optimized
- `sentence-transformers/paraphrase-mpnet-base-v2`: Enhanced paraphrase handling

**High-Performance Models** (200MB+, 90s+ per evaluation):
- `sentence-transformers/all-roberta-large-v1`: RoBERTa-based, high quality
- `sentence-transformers/gtr-t5-base`: T5-based encoder
- `sentence-transformers/gtr-t5-large`: Large T5 encoder
- `BAAI/bge-small-en-v1.5`: Recent high-performance model
- `BAAI/bge-base-en-v1.5`: Balanced recent model
- `BAAI/bge-large-en-v1.5`: State-of-the-art performance

**Specialized Models**:
- `sentence-transformers/msmarco-distilbert-base-v4`: Search-optimized
- `sentence-transformers/multi-qa-mpnet-base-dot-v1`: QA-optimized

### Custom Model Integration

```bash
# Local models
python enhanced_rag_benchmark.py --models "./path/to/custom-model"

# Hugging Face models
python enhanced_rag_benchmark.py --models "organization/model-name"

# Mixed model types
python enhanced_rag_benchmark.py --models \
  "all-mpnet-base-v2" \
  "./local/fine-tuned-model" \
  "intfloat/e5-large-v2"
```

### Model Selection Strategies

**Domain-Specific Optimization**:
```bash
# Finance domain - test specialized models
python enhanced_rag_benchmark.py \
  --models "BAAI/bge-base-en-v1.5" "sentence-transformers/multi-qa-mpnet-base-cos-v1" \
  --corpus "financial_corpus.json"

# Technical domain - test larger models
python enhanced_rag_benchmark.py \
  --models "BAAI/bge-large-en-v1.5" "sentence-transformers/gtr-t5-large" \
  --corpus "technical_docs.jsonl"
```

## Command Line Usage

### Basic Commands

```bash
# Default benchmark (QuAC + TopiOCQA, all models)
python enhanced_rag_benchmark.py

# With external corpus
python enhanced_rag_benchmark.py --corpus "./domain_corpus.json"

# Custom output directory
python enhanced_rag_benchmark.py -o "./evaluation_results"

# Memory optimization
python enhanced_rag_benchmark.py --batch_size 8
```

### Model Selection

```bash
# Single model evaluation
python enhanced_rag_benchmark.py --models "all-mpnet-base-v2"

# Model comparison
python enhanced_rag_benchmark.py --models \
  "all-MiniLM-L6-v2" \
  "all-mpnet-base-v2" \
  "BAAI/bge-base-en-v1.5"

# Include custom models
python enhanced_rag_benchmark.py --models \
  "./fine-tuned-model" \
  "sentence-transformers/all-mpnet-base-v2"
```

### Dataset and Corpus Configuration

```bash
# Standard datasets with external corpus
python enhanced_rag_benchmark.py \
  --datasets "quac" "topicqa" \
  --corpus "./comprehensive_corpus.json"

# Custom dataset with domain corpus
python enhanced_rag_benchmark.py \
  --datasets "domain_conversations.csv" \
  --corpus "./domain_docs.jsonl" \
  --max_corpus_docs 10000

# Multiple custom datasets
python enhanced_rag_benchmark.py \
  --datasets "dataset1.json" "dataset2.csv" "dataset3.jsonl"
```

### Performance and Scale Control

```bash
# Quick evaluation
python enhanced_rag_benchmark.py \
  --models "all-MiniLM-L6-v2" \
  --max_conversations 10 \
  --max_corpus_docs 1000 \
  --no-graphs

# Comprehensive evaluation
python enhanced_rag_benchmark.py \
  --max_conversations 100 \
  --max_corpus_docs 50000 \
  --batch_size 32

# Memory-constrained systems
python enhanced_rag_benchmark.py \
  --batch_size 4 \
  --max_conversations 20 \
  --max_corpus_docs 5000
```

### Complete Example: Domain-Specific Evaluation

```bash
python enhanced_rag_benchmark.py \
  --models "BAAI/bge-base-en-v1.5" "all-mpnet-base-v2" "sentence-transformers/multi-qa-mpnet-base-cos-v1" \
  --datasets "financial_qa_conversations.json" \
  --corpus "comprehensive_finance_corpus.jsonl" \
  --max_conversations 75 \
  --max_corpus_docs 15000 \
  --batch_size 16 \
  -o "./finance_domain_evaluation"
```

## Evaluation Metrics

### Multi-Turn Specific Metrics

**Context Preservation**:
- **Definition**: Performance maintenance across conversation turns
- **Calculation**: Final turn NDCG ≥ First turn NDCG × 0.8
- **Range**: 0.0 to 1.0 (higher = better context utilization)

**Context Coherence**:
- **Definition**: Consistency of context progression
- **Purpose**: Measures natural conversation flow
- **Range**: 0.0 to 1.0 (higher = better coherence)

**Turn Progression**:
- **Definition**: Quality improvement across turns
- **Purpose**: Indicates learning from conversation history
- **Range**: 0.0 to 1.0 (higher = better progression)

### Retrieval Quality Metrics

**NDCG@10** (Primary Ranking Metric):
- **Purpose**: Ranking quality with position discounting
- **Range**: 0.0 to 1.0 (higher = better ranking)
- **Significance**: Most important metric for retrieval quality

**Recall@10**:
- **Purpose**: Coverage of relevant documents
- **Range**: 0.0 to 1.0 (higher = better coverage)
- **Significance**: Measures completeness of retrieval

**Precision@10**:
- **Purpose**: Accuracy of top 10 results
- **Range**: 0.0 to 1.0 (higher = better accuracy)
- **Significance**: Measures result quality

**F1@10**:
- **Purpose**: Balance of precision and recall
- **Range**: 0.0 to 1.0 (higher = better balance)
- **Significance**: Overall retrieval effectiveness

**MAP (Mean Average Precision)**:
- **Purpose**: Average precision across all relevant documents
- **Range**: 0.0 to 1.0 (higher = better overall precision)
- **Significance**: Comprehensive ranking quality

**MRR (Mean Reciprocal Rank)**:
- **Purpose**: Average reciprocal rank of first relevant document
- **Range**: 0.0 to 1.0 (higher = better early precision)
- **Significance**: First-result quality

### Domain-Specific Metrics

**Answer Relevance**:
- **Purpose**: Semantic similarity between questions and answers
- **Range**: 0.0 to 1.0 (higher = more relevant responses)
- **Significance**: Response appropriateness

**Retrieval Quality**:
- **Purpose**: Maximum similarity achieved in corpus
- **Range**: 0.0 to 1.0 (higher = better corpus match)
- **Significance**: Peak retrieval performance

### Realistic Relevance Thresholds

**Optimized for External Corpora**:
- **High Relevance** (≥ 0.50): Score 1.0 - Highly relevant documents
- **Medium Relevance** (≥ 0.35): Score 0.7 - Moderately relevant documents  
- **Low Relevance** (≥ 0.20): Score 0.3 - Somewhat relevant documents
- **Not Relevant** (< 0.20): Score 0.0 - Irrelevant documents

## Visualization System

### Adaptive Visualization Engine

The system automatically generates relevant visualizations based on your evaluation configuration:

**Single Model + Single Dataset + Single Corpus**:
- Detailed performance profile
- Corpus impact analysis
- Turn-by-turn breakdown

**Multiple Models + Single Dataset + Single Corpus**:
- Model comparison charts
- Performance ranking
- Speed vs. quality analysis

**Single Model + Multiple Datasets + Single Corpus**:
- Dataset-specific performance
- Cross-dataset consistency
- Domain adaptation assessment

**Multiple Models + Multiple Datasets + External Corpus**:
- Comprehensive heatmaps
- Multi-dimensional analysis
- Corpus impact visualization

### Generated Visualizations

**Performance Overview**:
- Key metrics comparison across all configurations
- Color-coded performance levels
- Statistical significance indicators

**Corpus Impact Analysis** (NEW):
- Performance difference with/without external corpus
- Corpus size vs. performance correlation
- Domain matching effectiveness

**Individual Metric Charts**:
- Detailed breakdown for each evaluation metric
- Model rankings for specific metrics
- Performance gap analysis

**Multi-Metric Comparison**:
- Comprehensive side-by-side model comparison
- Balanced scorecard approach
- Top performer identification

**Dataset-Specific Analysis**:
- Performance by dataset type
- Domain-specific model recommendations
- Cross-dataset generalization assessment

**Performance vs. Time Analysis**:
- Efficiency trade-off visualization
- Processing time benchmarks
- Resource utilization optimization

**PDF Report Generation**:
- Executive summary with key findings
- All visualizations in publication quality
- Model and corpus recommendations
- Performance interpretation guide

## Output and Results

### File Structure

```
benchmark_results/
├── full_benchmark_results.json              # Complete raw results with corpus info
├── comprehensive_benchmark_results.csv      # Tabular summary with corpus details
├── comprehensive_rankings.json              # Model rankings by metric
├── quac_comprehensive_results.csv           # QuAC-specific results
├── topicqa_comprehensive_results.csv        # TopiOCQA-specific results
└── graphs/                                  # Visualization directory
    ├── metrics_overview.png                 # Key performance metrics
    ├── corpus_analysis.png                  # Corpus impact analysis (NEW)
    ├── individual_metrics_comparison.png    # (if multiple models)
    ├── multi_metric_comparison.png          # (if multiple models)
    ├── top_performers.png                   # (if multiple models)
    ├── performance_time_analysis.png        # Speed vs quality
    ├── detailed_breakdown.png               # Complete metrics heatmap
    └── benchmark_report.pdf                 # Comprehensive report
```

### Results Format with Corpus Information

**Enhanced Result Structure**:
```json
{
  "model": "BAAI/bge-base-en-v1.5",
  "dataset": "quac", 
  "external_corpus": "./finance_corpus.json",
  "corpus_size": 7661,
  "conversations_evaluated": 50,
  "metrics": {
    "ndcg_10": 0.3456,
    "recall_10": 0.2890,
    "precision_10": 0.2234,
    "f1_10": 0.2456,
    "map": 0.1987,
    "mrr": 0.3789,
    "context_coherence": 0.8123,
    "answer_relevance": 0.5678,
    "context_preservation": 0.7890
  },
  "timing": {
    "model_load_time": 4.2,
    "encoding_time": 78.5,
    "evaluation_time": 124.3,
    "total_time": 207.0
  }
}
```

### Console Output with Corpus Information

```
ULTIMATE RAG BENCHMARK RESULTS
================================================================================
Model                     Dataset  Corpus           Corp_Size  NDCG@10  F1@10   MAP     MRR     Time
BAAI/bge-base-en-v1.5    QUAC     finance_corp...  7661       0.345    0.246   0.199   0.379   207.0s
all-mpnet-base-v2        QUAC     finance_corp...  7661       0.312    0.225   0.182   0.356   156.8s
all-MiniLM-L6-v2         QUAC     finance_corp...  7661       0.278    0.198   0.165   0.334   89.2s

TOP PERFORMERS BY METRIC
================================================================================
Metric       Best Model                               Score
NDCG@10      BAAI/bge-base-en-v1.5                   0.3456
F1@10        BAAI/bge-base-en-v1.5                   0.2456
MAP          BAAI/bge-base-en-v1.5                   0.1987
MRR          BAAI/bge-base-en-v1.5                   0.3789
```

## Domain Compatibility Guidelines

### ✅ Recommended Domain Combinations

**General Knowledge + Wikipedia Corpus**:
```bash
--datasets "quac" --corpus "clapnq_corpus.json"
# Expected Performance: High (semantic alignment)
# Use Case: Educational QA, general knowledge systems
```

**Technical QA + Technical Documentation**:
```bash
--datasets "tech_support_conversations.json" --corpus "cloud_docs.jsonl"
# Expected Performance: High (domain match)
# Use Case: Developer tools, technical support, DevOps
```

**Financial QA + Financial Corpus**:
```bash
--datasets "financial_conversations.csv" --corpus "fiqa_corpus.json" 
# Expected Performance: High (specialized domain)
# Use Case: FinTech applications, investment platforms
```

**Legal/Policy QA + Government Corpus**:
```bash
--datasets "legal_qa.json" --corpus "govt_corpus.json"
# Expected Performance: High (regulatory domain)
# Use Case: Legal research, compliance systems, policy analysis
```

### ⚠️ Domain Mismatch Warnings

**Avoid These Combinations** (will produce poor results):

```bash
# ❌ POOR: General knowledge with specialized corpus
--datasets "quac" --corpus "fiqa_corpus.json"
# Expected Performance: Low (~0.1-0.2 NDCG@10)
# Reason: Wikipedia questions, finance documents = semantic mismatch

# ❌ POOR: Technical questions with unrelated domain
--datasets "tech_qa.json" --corpus "medical_corpus.json"
# Expected Performance: Very Low (~0.05-0.15 NDCG@10)  
# Reason: No vocabulary or conceptual overlap

# ❌ POOR: Financial questions with government policy
--datasets "finance_conversations.csv" --corpus "govt_corpus.json"
# Expected Performance: Low (~0.1-0.2 NDCG@10)
# Reason: Different regulatory contexts and terminology
```

### Domain Matching Best Practices

**1. Vocabulary Alignment**:
- Ensure corpus contains terminology used in conversations
- Check for domain-specific jargon and concepts
- Verify semantic field overlap

**2. Context Relevance**:
- Match question types with corpus content types
- Align conversation complexity with document detail level
- Consider temporal relevance (recent vs. historical content)

**3. Performance Expectations**:
- **High Match (0.3+ NDCG@10)**: Same domain, aligned vocabulary
- **Medium Match (0.2-0.3 NDCG@10)**: Related domains, some overlap
- **Low Match (0.1-0.2 NDCG@10)**: Different domains, minimal overlap
- **Poor Match (<0.1 NDCG@10)**: Unrelated domains, no semantic connection

**4. Custom Corpus Creation**:
```bash
# Recommended corpus size by use case:
# - Development/Testing: 1,000-5,000 documents
# - Research/Analysis: 5,000-20,000 documents  
# - Production Evaluation: 20,000+ documents
```

### Domain-Specific Recommendations

**Finance Domain**:
- **Corpus Types**: SEC filings, financial reports, investment guides, regulatory documents
- **Compatible Datasets**: Financial QA, investment queries, regulatory questions
- **Key Models**: BGE-large, multi-qa-mpnet (perform well on financial content)

**Technical/Cloud Domain**:
- **Corpus Types**: API documentation, technical guides, troubleshooting docs, architecture specs
- **Compatible Datasets**: Technical support, developer QA, system administration
- **Key Models**: All-mpnet-base-v2, GTR-T5 models (handle technical terminology well)

**Legal/Government Domain**:
- **Corpus Types**: Legal documents, policy papers, regulatory texts, case law
- **Compatible Datasets**: Legal research, compliance QA, policy analysis
- **Key Models**: BGE-large, all-roberta-large (perform well on formal/legal text)

**Healthcare Domain**:
- **Corpus Types**: Medical literature, clinical guidelines, drug information, health policies
- **Compatible Datasets**: Medical QA, clinical support, health information
- **Key Models**: Specialized medical models or BGE-large for general medical content

## Advanced Usage

### Custom Evaluation Pipeline

```python
from enhanced_rag_benchmark import UltimateRAGBenchmark

# Initialize with external corpus support
benchmark = UltimateRAGBenchmark(batch_size=16)
benchmark.max_corpus_docs = 10000  # Limit corpus size

# Set specific models and datasets
benchmark.embedding_models = [
    "BAAI/bge-base-en-v1.5",
    "all-mpnet-base-v2", 
    "./fine-tuned-domain-model"
]
benchmark.datasets = ["domain_conversations.json"]

# Run evaluation with external corpus
results = benchmark.run_full_benchmark(
    output_dir="./domain_evaluation",
    max_conversations=50,
    create_graphs=True,
    corpus_path="./domain_corpus.jsonl"
)

# Access corpus-specific results
for result in results['individual_results']:
    print(f"{result['model']} on {result['dataset']} "
          f"with {result['corpus_size']} docs: "
          f"NDCG@10 = {result['metrics']['ndcg_10']:.3f}")
```

### Batch Domain Evaluation

```python
# Evaluate multiple domains
domain_configs = [
    {
        "name": "finance",
        "datasets": ["finance_qa.json"],
        "corpus": "finance_corpus.jsonl",
        "models": ["BAAI/bge-base-en-v1.5", "all-mpnet-base-v2"]
    },
    {
        "name": "technical", 
        "datasets": ["tech_support.csv"],
        "corpus": "technical_docs.json",
        "models": ["sentence-transformers/gtr-t5-base", "all-mpnet-base-v2"]
    }
]

for config in domain_configs:
    benchmark.embedding_models = config["models"]
    benchmark.datasets = config["datasets"] 
    
    results = benchmark.run_full_benchmark(
        output_dir=f"./evaluation_{config['name']}", 
        corpus_path=config["corpus"],
        max_conversations=50
    )
```

### Corpus Preprocessing

```python
# Load and preprocess large corpus
corpus_data = benchmark.load_external_corpus(
    corpus_path="./large_corpus.jsonl",
    max_docs=50000
)

print(f"Loaded {len(corpus_data)} documents")
print(f"Sample document: {list(corpus_data.values())[0][:100]}...")

# Save preprocessed corpus for reuse
import json
with open("./preprocessed_corpus.json", 'w') as f:
    json.dump(corpus_data, f, indent=2)
```

### Performance Analysis

```python
# Analyze corpus impact
def analyze_corpus_impact(results_with_corpus, results_without_corpus):
    for model in set(r['model'] for r in results_with_corpus):
        with_corpus = next(r for r in results_with_corpus if r['model'] == model)
        without_corpus = next(r for r in results_without_corpus if r['model'] == model)
        
        ndcg_improvement = (with_corpus['metrics']['ndcg_10'] - 
                           without_corpus['metrics']['ndcg_10'])
        
        print(f"{model}: NDCG@10 improvement = {ndcg_improvement:+.3f}")
```

## Performance Optimization

### Hardware Recommendations

**Minimum Configuration**:
- 8GB RAM
- 4 CPU cores  
- 50GB disk space (for large corpora)

**Recommended Configuration**:
- 16GB+ RAM
- 8+ CPU cores or GPU
- SSD storage
- 100GB+ disk space

**Optimal Configuration**:
- 32GB+ RAM
- GPU with 8GB+ VRAM
- NVMe SSD storage
- 200GB+ disk space for large-scale evaluation

### Memory Optimization for Large Corpora

```bash
# Large corpus (50K+ documents)
python enhanced_rag_benchmark.py \
  --corpus "large_corpus.jsonl" \
  --max_corpus_docs 50000 \
  --batch_size 8 \
  --max_conversations 25

# Memory-constrained systems
python enhanced_rag_benchmark.py \
  --corpus "corpus.json" \
  --max_corpus_docs 5000 \
  --batch_size 4 \
  --max_conversations 10

# GPU acceleration
python enhanced_rag_benchmark.py \
  --corpus "corpus.jsonl" \
  --batch_size 64 \
  --max_conversations 100
```

### Corpus Size Guidelines

| System RAM | Max Corpus Docs | Batch Size | Expected Time |
|------------|-----------------|------------|---------------|
| 8GB        | 5,000          | 4-8        | 2-4 hours     |
| 16GB       | 15,000         | 8-16       | 3-6 hours     |
| 32GB       | 50,000         | 16-32      | 5-10 hours    |
| 64GB+      | 100,000+       | 32-64      | 8-15 hours    |

### Processing Strategy for Large Evaluations

**Phase 1: Screening**
```bash
# Quick model screening with small corpus
python enhanced_rag_benchmark.py \
  --models "all-MiniLM-L6-v2" "all-mpnet-base-v2" \
  --corpus "sample_corpus.json" \
  --max_corpus_docs 1000 \
  --max_conversations 10
```

**Phase 2: Selection**
```bash
# Detailed evaluation with selected models
python enhanced_rag_benchmark.py \
  --models "all-mpnet-base-v2" "BAAI/bge-base-en-v1.5" \
  --corpus "full_corpus.jsonl" \
  --max_corpus_docs 25000 \
  --max_conversations 50
```

**Phase 3: Production**
```bash
# Comprehensive evaluation with best models
python enhanced_rag_benchmark.py \
  --models "BAAI/bge-base-en-v1.5" \
  --corpus "production_corpus.jsonl" \
  --max_conversations 100
```

## Troubleshooting

### Common Issues

**Memory Errors with Large Corpora**:
```bash
# Reduce corpus size
--max_corpus_docs 5000

# Reduce batch size
--batch_size 4

# Limit conversations
--max_conversations 10
```

**Corpus Loading Failures**:
```
Error: No valid documents found in corpus file

Solutions:
1. Check file format (JSON/JSONL/CSV)
2. Verify text fields exist ('text', 'content', 'passage')
3. Ensure UTF-8 encoding
4. Check file permissions
```

**Poor Performance with External Corpus**:
```
NDCG@10 < 0.1 indicates domain mismatch

Solutions:
1. Verify domain compatibility
2. Check vocabulary overlap
3. Use appropriate dataset-corpus combination
4. Consider corpus preprocessing/filtering
```

**Model Loading Issues**:
```bash
# Test model loading
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('model-name')"

# Use alternative models
--models "all-MiniLM-L6-v2"  # Reliable fallback
```

### Debug Information

**Corpus Validation**:
```python
# Check corpus structure
corpus = benchmark.load_external_corpus("corpus.json", max_docs=10)
print(f"Loaded {len(corpus)} documents")
for doc_id, text in list(corpus.items())[:3]:
    print(f"ID: {doc_id}, Text: {text[:100]}...")
```

**Dataset Validation**:
```python
# Check dataset structure  
data = benchmark.load_dataset_safe("dataset.json", max_conversations=5)
if data:
    print(f"Conversations: {len(data['conversations'])}")
    print(f"Built-in corpus: {len(data['corpus'])}")
```

### Performance Diagnostics

**Corpus Processing Time**:
- JSON: ~1000 docs/second
- JSONL: ~800 docs/second  
- CSV: ~500 docs/second

**Memory Usage Estimation**:
- 1K docs: ~100MB RAM
- 10K docs: ~800MB RAM
- 50K docs: ~4GB RAM
- 100K docs: ~8GB RAM

**Expected Evaluation Times**:
- Small corpus (1K docs): 5-15 min per model
- Medium corpus (10K docs): 15-45 min per model
- Large corpus (50K docs): 45-120 min per model

This **Ultimate RAG Benchmark** provides the most comprehensive solution for domain-aware multi-turn conversational evaluation, enabling realistic assessment of embedding model performance in specialized domains through external corpus integration. The system's flexibility in handling various data formats, combined with its sophisticated visualization and analysis capabilities, makes it the ideal tool for both research and production model selection.
