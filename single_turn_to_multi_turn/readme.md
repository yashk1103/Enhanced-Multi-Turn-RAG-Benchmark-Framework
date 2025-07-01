# Multi-Turn RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems on multi-turn conversational scenarios. This system automatically generates synthetic multi-turn conversations from single-turn datasets and evaluates how well RAG systems maintain context and performance across conversation turns.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Synthetic Conversation Generation](#synthetic-conversation-generation)
6. [Evaluation Metrics](#evaluation-metrics)
7. [BEIR Dataset Support](#beir-dataset-support)
8. [Command Line Usage](#command-line-usage)
9. [Output and Results](#output-and-results)
10. [Configuration Options](#configuration-options)
11. [Advanced Usage](#advanced-usage)
12. [Understanding the Results](#understanding-the-results)
13. [Customization](#customization)
14. [Troubleshooting](#troubleshooting)
15. [Performance Optimization](#performance-optimization)

## Overview

This framework addresses a critical gap in RAG evaluation: most existing datasets contain only single-turn queries, but real-world conversational AI systems must handle multi-turn interactions where context builds across turns. The system automatically transforms single-turn datasets into synthetic multi-turn conversations and evaluates how well RAG retrieval systems maintain performance as conversations progress.

### Key Capabilities

- **Synthetic Multi-Turn Generation**: Automatically converts single-turn queries into realistic multi-turn conversations
- **Context-Aware Evaluation**: Measures how well systems maintain context across conversation turns
- **BEIR Dataset Integration**: Works with all BEIR benchmark datasets out of the box
- **Template-Based Question Generation**: Uses sophisticated templates to create natural follow-up questions
- **Comprehensive Metrics**: Evaluates context preservation, turn-by-turn performance, and conversation success
- **Flexible Architecture**: Supports any sentence transformer model for retrieval evaluation

## Features

### Synthetic Conversation Generation

**Question Types Generated:**
- **Elaboration**: "Can you tell me more about X?"
- **Clarification**: "What exactly do you mean by X?"
- **Comparison**: "How does this compare to Y?"
- **Follow-up**: Context-aware follow-up questions

**Template Categories:**
- Follow-up templates (8 variations)
- Clarification templates (4 variations)
- Comparison templates (3 variations)
- Automatic topic extraction from queries and documents

### Evaluation Metrics

**Multi-Turn Specific Metrics:**
- **Context Preservation**: Measures whether performance maintains across turns
- **Conversation Success Rate**: Percentage of conversations with at least one successful retrieval
- **Turn-by-Turn Performance**: NDCG@10 and Recall@10 for each conversation turn
- **Final Turn Performance**: How well the system performs on the last turn of conversations

**Standard Retrieval Metrics:**
- NDCG@10 (Normalized Discounted Cumulative Gain)
- Recall@10 (Proportion of relevant documents retrieved)

### BEIR Dataset Support

**Compatible Datasets:**
- NFCorpus (Nutrition)
- TREC-COVID (COVID-19 research)
- SciFact (Scientific fact verification)
- MS MARCO (Web search)
- Natural Questions (Question answering)
- HotpotQA (Multi-hop reasoning)
- FiQA (Financial QA)
- ArguAna (Argument mining)
- Touche-2020 (Argument retrieval)
- CQADupStack (Community QA)
- Quora (Question similarity)
- DBPedia (Entity retrieval)
- SCIDOCS (Scientific document similarity)
- FEVER (Fact verification)
- Climate-FEVER (Climate fact verification)
- Signal-1M (News clustering)
- TREC-NEWS (News background linking)
- Robust04 (News retrieval)

## Installation

### Requirements

```bash
pip install sentence-transformers
pip install beir
pip install numpy
pip install scikit-learn
```

### Quick Install

```bash
pip install sentence-transformers beir numpy scikit-learn
```

### Verify Installation

```python
from beir import util
from sentence_transformers import SentenceTransformer
print("Installation successful!")
```

## Quick Start

### Basic Evaluation

```bash
python multiturn_rag_evaluator.py
```

This runs with default settings:
- Model: all-MiniLM-L6-v2
- Dataset: nfcorpus
- Number of turns: 3
- Output: ./multiturn_results

### Custom Model Evaluation

```bash
python multiturn_rag_evaluator.py --model "sentence-transformers/all-mpnet-base-v2"
```

### Different Dataset

```bash
python multiturn_rag_evaluator.py --dataset "scifact" --num_turns 4
```

### Complete Custom Run

```bash
python multiturn_rag_evaluator.py \
  --model "BAAI/bge-large-en-v1.5" \
  --dataset "msmarco" \
  --num_turns 5 \
  --batch_size 32 \
  --output_dir "./my_results"
```

## Synthetic Conversation Generation

### How It Works

The system transforms single-turn queries through a sophisticated multi-step process:

1. **Topic Extraction**: Identifies key concepts from original queries and relevant documents
2. **Template Selection**: Chooses appropriate question templates based on turn number and context
3. **Context Building**: Maintains cumulative context across conversation turns
4. **Query Classification**: Categorizes generated questions by type (elaboration, clarification, comparison)

### Generation Process

**Turn 0 (Initial)**: Original query from dataset
```
"What are the health benefits of omega-3 fatty acids?"
```

**Turn 1 (Follow-up)**: Generated elaboration question
```
Context: "What are the health benefits of omega-3 fatty acids?"
Generated: "Can you tell me more about omega-3?"
```

**Turn 2 (Clarification)**: Generated clarification question
```
Context: Previous turns...
Generated: "What exactly are fatty acids?"
```

**Turn 3 (Comparison)**: Generated comparison question
```
Context: Previous turns...
Generated: "How does this compare to omega-6?"
```

### Template System

**Follow-up Templates:**
- "Can you tell me more about {topic}?"
- "What else should I know about {topic}?"
- "How does {topic} relate to {context}?"
- "Can you explain {topic} in more detail?"
- "What are some examples of {topic}?"
- "Why is {topic} important?"
- "What are the benefits of {topic}?"
- "What problems does {topic} solve?"

**Clarification Templates:**
- "I didn't understand the part about {concept}"
- "Can you clarify what you mean by {concept}?"
- "What exactly is {concept}?"
- "How does {concept} work?"

**Comparison Templates:**
- "How does this compare to {alternative}?"
- "What's the difference between this and {alternative}?"
- "Which is better: this or {alternative}?"

### Topic Extraction

**From Queries:**
- Extracts words longer than 3 characters
- Removes common stop words
- Focuses on domain-specific terms

**From Documents:**
- Uses top 2 most relevant documents
- Extracts from title and text fields
- Limits to first 100 characters for efficiency
- Focuses on words longer than 4 characters

## Evaluation Metrics

### Context Preservation

**Definition**: Measures whether retrieval performance is maintained or improved as conversation context builds.

**Calculation**: 
- Compare final turn NDCG@10 with first turn NDCG@10
- Score = 1.0 if final_turn >= first_turn * 0.8, else 0.0
- Average across all conversations

**Interpretation**:
- 1.0: Perfect context preservation
- 0.8+: Good context utilization
- 0.5-0.8: Moderate context preservation
- <0.5: Poor context handling

### Conversation Success Rate

**Definition**: Percentage of conversations where at least one turn successfully retrieves relevant documents.

**Calculation**: 
- Success = any turn retrieves relevant documents
- Rate = successful_conversations / total_conversations

**Interpretation**:
- 1.0: All conversations successful
- 0.8+: High success rate
- 0.5-0.8: Moderate success
- <0.5: Low success rate

### Turn-by-Turn Performance

**NDCG@10 (Normalized Discounted Cumulative Gain)**:
- Measures ranking quality considering position
- Accounts for multiple relevance levels
- Range: 0.0 to 1.0 (higher is better)

**Recall@10**:
- Proportion of relevant documents found in top 10
- Range: 0.0 to 1.0 (higher is better)

### Final Turn Performance

**Definition**: NDCG@10 performance on the last turn of each conversation.

**Significance**: 
- Tests system's ability to handle complex, context-rich queries
- Measures cumulative context utilization
- Critical for conversational AI applications

## BEIR Dataset Support

### Automatic Dataset Download

The system automatically downloads and extracts BEIR datasets:

```python
# Automatic download process
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
data_path = util.download_and_unzip(url, data_dir)
```

### Dataset Structure

**Expected Format**:
```python
corpus = {
    "doc_id": {
        "title": "Document title",
        "text": "Document content..."
    }
}

queries = {
    "query_id": "Query text"
}

qrels = {
    "query_id": {
        "doc_id": relevance_score
    }
}
```

### Dataset Characteristics

**Small Datasets** (Quick testing):
- NFCorpus: 3.6K documents, 323 queries
- SciFact: 5K documents, 300 queries
- ArguAna: 8.7K documents, 1.4K queries

**Medium Datasets** (Balanced evaluation):
- TREC-COVID: 171K documents, 50 queries
- FiQA: 57K documents, 648 queries
- Touche-2020: 382K documents, 49 queries

**Large Datasets** (Comprehensive evaluation):
- MS MARCO: 8.8M documents, 6.9K queries
- Natural Questions: 2.7M documents, 3.5K queries
- HotpotQA: 5.2M documents, 7.4K queries

## Command Line Usage

### Basic Commands

```bash
# Default evaluation
python multiturn_rag_evaluator.py

# Specify model
python multiturn_rag_evaluator.py --model "all-mpnet-base-v2"

# Specify dataset
python multiturn_rag_evaluator.py --dataset "scifact"

# Set number of turns
python multiturn_rag_evaluator.py --num_turns 5

# Custom output directory
python multiturn_rag_evaluator.py --output_dir "./results"

# Adjust batch size
python multiturn_rag_evaluator.py --batch_size 32
```

### Model Selection

**Popular Models:**
```bash
# Fast models
python multiturn_rag_evaluator.py --model "all-MiniLM-L6-v2"

# Balanced models
python multiturn_rag_evaluator.py --model "all-mpnet-base-v2"

# High-performance models
python multiturn_rag_evaluator.py --model "BAAI/bge-large-en-v1.5"

# Specialized models
python multiturn_rag_evaluator.py --model "sentence-transformers/multi-qa-mpnet-base-cos-v1"
```

### Dataset Selection

**Domain-Specific Evaluation:**
```bash
# Scientific domains
python multiturn_rag_evaluator.py --dataset "scifact"
python multiturn_rag_evaluator.py --dataset "trec-covid"

# General web search
python multiturn_rag_evaluator.py --dataset "msmarco"
python multiturn_rag_evaluator.py --dataset "natural-questions"

# Specialized domains
python multiturn_rag_evaluator.py --dataset "fiqa"      # Financial
python multiturn_rag_evaluator.py --dataset "nfcorpus"  # Nutrition
```

### Performance Optimization

```bash
# Memory-constrained systems
python multiturn_rag_evaluator.py --batch_size 4

# High-performance systems
python multiturn_rag_evaluator.py --batch_size 64

# Quick testing
python multiturn_rag_evaluator.py --dataset "nfcorpus" --num_turns 2
```

## Output and Results

### File Structure

```
multiturn_results/
└── {dataset_name}_multiturn_results.json
```

### Results File Format

```json
{
  "dataset": "nfcorpus",
  "num_turns": 3,
  "metrics": {
    "context_preservation": 0.7500,
    "avg_ndcg": 0.2341,
    "avg_recall": 0.1875,
    "final_turn_ndcg": 0.2654,
    "conversation_success_rate": 0.8400,
    "total_conversations": 50,
    "total_turns": 150
  },
  "time_seconds": 127.3,
  "timestamp": "2024-01-15T10:30:45",
  "sample_conversations": [...]
}
```

### Console Output

```
Multi-Turn RAG Evaluation
============================================================
Dataset: nfcorpus
Number of turns: 3
============================================================
Loaded 3633 documents, 323 queries
Generating synthetic multi-turn conversations...
Encoding corpus...
Evaluating multi-turn retrieval...

Multi-Turn Retrieval Results:
Context Preservation: 0.7500
Turn-by-Turn NDCG@10: 0.2341
Turn-by-Turn Recall@10: 0.1875
Final Turn NDCG@10: 0.2654
Conversation Success Rate: 0.8400
Time: 127.3s

Results saved to: ./multiturn_results/nfcorpus_multiturn_results.json
```

### Sample Conversation Structure

```json
{
  "conversation_id": "PLAIN-1",
  "turns": [
    {
      "turn_id": 0,
      "query": "What are the health benefits of fish oil?",
      "contextualized_query": "What are the health benefits of fish oil?",
      "query_type": "initial",
      "ndcg_10": 0.3456,
      "recall_10": 0.2500,
      "retrieved_docs": ["doc_123", "doc_456", ...],
      "retrieved_relevant_count": 2
    },
    {
      "turn_id": 1,
      "query": "Can you tell me more about omega-3?",
      "contextualized_query": "What are the health benefits of fish oil? Can you tell me more about omega-3?",
      "query_type": "elaboration",
      "ndcg_10": 0.3789,
      "recall_10": 0.3000,
      "retrieved_docs": ["doc_789", "doc_012", ...],
      "retrieved_relevant_count": 3
    }
  ],
  "relevant_docs_count": 4
}
```

## Configuration Options

### Model Configuration

**Embedding Models**: Any sentence-transformers compatible model
- Local models: `"./path/to/model"`
- Hugging Face models: `"organization/model-name"`
- Short names: `"all-MiniLM-L6-v2"`

### Generation Parameters

**Number of Turns**: 2-10 turns per conversation
- 2-3 turns: Quick evaluation
- 4-5 turns: Standard evaluation
- 6+ turns: Comprehensive evaluation

**Batch Size**: Depends on available memory
- 4-8: Limited memory systems
- 16-32: Standard systems
- 64+: High-memory systems

### Dataset Limitations

**Conversation Limit**: Currently limited to 50 conversations per evaluation for demonstration
- Modify in code for full dataset evaluation
- Balances thoroughness with execution time

## Advanced Usage

### Custom Evaluation Pipeline

```python
from multiturn_rag_evaluator import MultiTurnRAGEvaluator

# Initialize evaluator
evaluator = MultiTurnRAGEvaluator("all-mpnet-base-v2", batch_size=16)

# Run evaluation
result = evaluator.evaluate_multiturn_retrieval(
    dataset_name="scifact",
    output_dir="./custom_results", 
    num_turns=4
)

# Access metrics
metrics = result['metrics']
print(f"Context Preservation: {metrics['context_preservation']:.2%}")
```

### Custom Template Generation

```python
from multiturn_rag_evaluator import SyntheticMultiTurnGenerator

# Initialize generator
generator = SyntheticMultiTurnGenerator()

# Add custom templates
generator.follow_up_templates.append("What research supports {topic}?")
generator.clarification_templates.append("Can you provide evidence for {concept}?")

# Generate conversations
multiturn_data = generator.create_multiturn_from_single(
    queries, corpus, qrels, num_turns=4
)
```

### Batch Dataset Evaluation

```python
datasets = ["nfcorpus", "scifact", "fiqa", "trec-covid"]
models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"]

for model in models:
    evaluator = MultiTurnRAGEvaluator(model)
    for dataset in datasets:
        result = evaluator.evaluate_multiturn_retrieval(
            dataset, f"./results/{model}", num_turns=3
        )
```

## Understanding the Results

### Metric Interpretation

**Context Preservation Scores:**
- **0.9-1.0**: Excellent context utilization
- **0.7-0.9**: Good context maintenance
- **0.5-0.7**: Moderate context handling
- **<0.5**: Poor context preservation

**Conversation Success Rates:**
- **0.9+**: Nearly all conversations successful
- **0.7-0.9**: Most conversations successful
- **0.5-0.7**: Mixed success
- **<0.5**: Many conversations fail

**NDCG@10 Benchmarks:**
- **0.4+**: Excellent retrieval performance
- **0.3-0.4**: Good retrieval performance
- **0.2-0.3**: Moderate retrieval performance
- **<0.2**: Poor retrieval performance

### Performance Patterns

**Typical Behavior:**
- First turn performance reflects single-turn capability
- Middle turns test context integration
- Final turn reveals cumulative context utilization

**Success Indicators:**
- Stable or improving performance across turns
- High conversation success rate
- Final turn performance ≥ first turn performance

**Failure Modes:**
- Declining performance across turns (context confusion)
- Low conversation success rate (poor retrieval)
- Large gap between first and final turn performance

## Customization

### Adding New Question Templates

```python
class CustomMultiTurnGenerator(SyntheticMultiTurnGenerator):
    def __init__(self):
        super().__init__()
        # Add domain-specific templates
        self.follow_up_templates.extend([
            "What research supports {topic}?",
            "What are the limitations of {topic}?",
            "How has {topic} evolved over time?"
        ])
```

### Custom Evaluation Metrics

```python
def custom_metric(self, conversations_results):
    # Implement custom evaluation logic
    scores = []
    for conv in conversations_results:
        # Calculate custom score
        score = custom_calculation(conv)
        scores.append(score)
    return np.mean(scores)

# Add to evaluator
evaluator._calculate_custom_metric = custom_metric
```

### Custom Dataset Integration

```python
def load_custom_dataset(self, data_path):
    # Load custom data format
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to BEIR format
    corpus = {}
    queries = {}
    qrels = {}
    
    # Implementation depends on your data format
    return corpus, queries, qrels
```

## Troubleshooting

### Common Issues

**Memory Errors:**
```bash
# Reduce batch size
python multiturn_rag_evaluator.py --batch_size 4

# Use smaller dataset for testing
python multiturn_rag_evaluator.py --dataset "nfcorpus"
```

**Dataset Download Failures:**
- Check internet connection
- Verify dataset name spelling
- Try different dataset if persistent issues

**Model Loading Issues:**
- Verify model name/path
- Check sentence-transformers compatibility
- Ensure sufficient disk space

**Performance Issues:**
- Use smaller datasets for testing
- Reduce number of turns
- Increase batch size if memory allows

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation

**Quick Test:**
```bash
python multiturn_rag_evaluator.py --dataset "nfcorpus" --num_turns 2
```

**Model Test:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("your-model-name")
embeddings = model.encode(["test sentence"])
print(f"Embedding shape: {embeddings.shape}")
```

## Performance Optimization

### Hardware Recommendations

**Minimum Requirements:**
- 8GB RAM
- 4 CPU cores
- 20GB disk space

**Recommended:**
- 16GB+ RAM
- 8+ CPU cores or GPU
- 100GB+ disk space (for large datasets)

**Optimal:**
- 32GB+ RAM
- GPU with 8GB+ VRAM
- SSD storage

### Optimization Strategies

**For Large Datasets:**
1. Increase batch sizes
2. Use GPU acceleration
3. Process in chunks
4. Cache embeddings

**For Memory-Constrained Systems:**
1. Reduce batch sizes
2. Use smaller models
3. Limit conversation count
4. Process one dataset at a time

**For Time-Constrained Evaluation:**
1. Use smaller datasets first
2. Reduce number of turns
3. Use faster models for screening
4. Parallel processing across datasets

### Batch Size Guidelines

| System Memory | Recommended Batch Size |
|---------------|------------------------|
| 8GB           | 4-8                    |
| 16GB          | 8-16                   |
| 32GB          | 16-32                  |
| 64GB+         | 32-64                  |

This framework provides a comprehensive solution for evaluating multi-turn RAG systems, bridging the gap between single-turn benchmarks and real-world conversational AI applications. The synthetic conversation generation creates realistic multi-turn scenarios while the evaluation metrics provide detailed insights into context preservation and retrieval performance across conversation turns.