
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
from datasets import load_dataset
import pandas as pd
import traceback
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

class UltimateRAGBenchmark:
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self.results = []
        
        # Define models to benchmark
        self.embedding_models = [
            # Fast models
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2", 
            
            # Balanced models
            "all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-mpnet-base-v2",
            
            # High performance models
            "sentence-transformers/all-roberta-large-v1",
            "sentence-transformers/gtr-t5-base",
            "sentence-transformers/gtr-t5-large",
            
            # Specialized models
            "sentence-transformers/msmarco-distilbert-base-v4",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            
            # Recent models
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5"
        ]
        
        # Define datasets to test
        self.datasets = ["quac", "topicqa"]
        
        # External corpus support
        self.external_corpus = None
        self.max_corpus_docs = None

    def _extract_doc_info(self, doc: Any, index: int, format_type: str) -> Tuple[str, str]:
        """Extract document ID and text from various document formats"""
        doc_id = f"{format_type}_doc_{index}"
        text = ""
        
        if isinstance(doc, dict):
            # Priority order for ID fields
            id_fields = ['_id', 'document_id', 'id', 'doc_id', 'idx', 'index']
            text_fields = ['text', 'content', 'passage', 'document', 'body', 
                          'description', 'answer', 'response', 'message', 'data']
            
            # Extract ID
            for field in id_fields:
                if field in doc and doc[field]:
                    doc_id = str(doc[field]).strip()
                    break
            
            # Extract text - try multiple strategies
            for field in text_fields:
                if field in doc and doc[field]:
                    candidate_text = str(doc[field]).strip()
                    if len(candidate_text) > len(text):
                        text = candidate_text
            
            # Fallback: if no text found, try other fields
            if not text or len(text) < 5:
                for key, value in doc.items():
                    if key not in id_fields and isinstance(value, str):
                        candidate_text = str(value).strip()
                        if len(candidate_text) > 10:  # Reasonable minimum
                            text = candidate_text
                            break
            
            # Last resort: convert entire dict to string (excluding ID fields)
            if not text or len(text) < 5:
                filtered_doc = {k: v for k, v in doc.items() if k not in id_fields}
                if filtered_doc:
                    text = str(filtered_doc)
        
        elif isinstance(doc, str):
            text = doc.strip()
        
        else:
            text = str(doc).strip()
        
        # Clean up text
        if text:
            # Remove excessive whitespace
            text = ' '.join(text.split())
            # Remove newlines and tabs
            text = text.replace('\n', ' ').replace('\t', ' ')
            # Remove HTML-like tags if present
            import re
            text = re.sub(r'<[^>]+>', '', text)
            # Remove extra spaces
            text = ' '.join(text.split())
        
        return doc_id, text

    def load_external_corpus(self, corpus_path: str, max_docs: int = None) -> Dict[str, str]:
        """Load external corpus from JSON, JSONL, or CSV file - handles all format variations"""
        print(f"Loading external corpus from {corpus_path}...")
        
        try:
            file_ext = corpus_path.lower().split('.')[-1]
            corpus = {}
            
            if file_ext == 'jsonl':
                # Handle JSONL format (each line is a JSON object)
                print("Detected JSONL format")
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if max_docs and len(corpus) >= max_docs:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            doc = json.loads(line)
                            doc_id, text = self._extract_doc_info(doc, i, "jsonl")
                            
                            if text and len(text.strip()) > 5:  # More lenient minimum length
                                corpus[doc_id] = text.strip()
                                
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {i+1}: {e}")
                            continue
            
            elif file_ext == 'csv':
                # Handle CSV format
                print("Detected CSV format")
                df = pd.read_csv(corpus_path)
                
                # Enhanced column detection
                text_fields = ['text', 'content', 'passage', 'document', 'body', 
                              'description', 'answer', 'response', 'message', 'data']
                id_fields = ['_id', 'document_id', 'id', 'doc_id', 'idx', 'index']
                
                # Find text column
                text_col = None
                for col in text_fields:
                    if col in df.columns:
                        text_col = col
                        break
                
                # If no standard text column, find column with longest average text
                if not text_col:
                    text_lengths = {}
                    for col in df.columns:
                        if df[col].dtype == 'object':  # Text columns
                            try:
                                avg_len = df[col].astype(str).str.len().mean()
                                if avg_len > 10:  # Only consider columns with substantial text
                                    text_lengths[col] = avg_len
                            except:
                                continue
                    
                    if text_lengths:
                        text_col = max(text_lengths, key=text_lengths.get)
                        print(f"Using column '{text_col}' as text source (longest average length: {text_lengths[text_col]:.1f})")
                
                # Find ID column
                id_col = None
                for col in id_fields:
                    if col in df.columns:
                        id_col = col
                        break
                
                if text_col:
                    for i, row in df.iterrows():
                        if max_docs and len(corpus) >= max_docs:
                            break
                        
                        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                        text = ' '.join(text.split())  # Clean whitespace
                        
                        if len(text.strip()) > 5:
                            if id_col and pd.notna(row[id_col]):
                                doc_id = str(row[id_col]).strip()
                            else:
                                doc_id = f"csv_doc_{i}"
                            corpus[doc_id] = text.strip()
                else:
                    print("Warning: Could not identify text column in CSV")
                    print(f"Available columns: {list(df.columns)}")
                    print("Please ensure your CSV has one of these text columns:")
                    print(f"  {', '.join(text_fields)}")
            
            else:
                # Handle JSON format (default)
                print("Detected JSON format")
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # List of documents format
                    for i, doc in enumerate(data):
                        if max_docs and len(corpus) >= max_docs:
                            break
                        
                        doc_id, text = self._extract_doc_info(doc, i, "json")
                        if text and len(text.strip()) > 5:
                            corpus[doc_id] = text.strip()
                
                elif isinstance(data, dict):
                    # Check if it's a direct mapping first
                    if all(isinstance(v, str) for v in data.values() if v):
                        # Direct mapping format: {"doc1": "text1", "doc2": "text2"}
                        for doc_id, text in data.items():
                            if max_docs and len(corpus) >= max_docs:
                                break
                            if text and len(str(text).strip()) > 5:
                                corpus[str(doc_id)] = str(text).strip()
                    else:
                        # Nested format or complex structure
                        count = 0
                        for key, value in data.items():
                            if max_docs and count >= max_docs:
                                break
                            
                            doc_id, text = self._extract_doc_info(value, count, f"json_{key}")
                            if not text and isinstance(value, str):
                                text = value.strip()
                                doc_id = str(key)
                            
                            if text and len(text.strip()) > 5:
                                corpus[doc_id] = text.strip()
                                count += 1
            
            print(f"Successfully loaded {len(corpus)} documents from external corpus ({file_ext.upper()} format)")
            
            if len(corpus) == 0:
                print("Warning: No valid documents found in corpus file")
                print("Supported formats and examples:")
                print("  JSON: [{\"_id\": \"123\", \"text\": \"content\"}, ...]")
                print("  JSONL: Each line like {\"document_id\": \"abc\", \"text\": \"content\"}")
                print("  CSV: Must have text/content column and optional id column")
                print("\nSupported field names:")
                print("  Text: text, content, passage, document, body, description, answer, response")
                print("  ID: _id, document_id, id, doc_id, idx, index")
            else:
                # Show sample of loaded documents
                sample_ids = list(corpus.keys())[:3]
                print(f"Sample documents loaded:")
                for doc_id in sample_ids:
                    text_preview = corpus[doc_id][:100] + "..." if len(corpus[doc_id]) > 100 else corpus[doc_id]
                    print(f"  ID {doc_id}: {text_preview}")
            
            return corpus
            
        except Exception as e:
            print(f"Error loading external corpus: {e}")
            print(f"File: {corpus_path}")
            print("Please check:")
            print("  - File exists and is readable")
            print("  - File format is valid JSON/JSONL/CSV")
            print("  - File encoding is UTF-8")
            traceback.print_exc()
            return {}

    def load_dataset_safe(self, dataset_name: str, max_conversations: int = 50):
        """Safely load datasets with error handling - supports ALL formats"""
        try:
            # Check if it's a custom dataset file
            if dataset_name.endswith(('.json', '.csv', '.jsonl')) or os.path.exists(dataset_name):
                return self._load_custom_dataset(dataset_name, max_conversations)
            elif dataset_name.lower() == "topicqa":
                return self._load_topicqa_dataset(max_conversations)
            elif dataset_name.lower() == "quac":
                return self._load_quac_dataset(max_conversations)
            else:
                print(f"Unsupported dataset: {dataset_name}")
                return None
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            traceback.print_exc()
            return None

    def _load_custom_dataset(self, dataset_path: str, max_conversations: int):
        """COMPREHENSIVE custom dataset loader - handles ALL CSV/JSON/JSONL formats"""
        print(f"Loading custom dataset from {dataset_path}...")
        
        try:
            file_ext = dataset_path.lower().split('.')[-1]
            
            if file_ext == 'csv':
                return self._load_csv_dataset(dataset_path, max_conversations)
            elif file_ext == 'jsonl':
                return self._load_jsonl_dataset(dataset_path, max_conversations)
            else:
                return self._load_json_dataset(dataset_path, max_conversations)
                
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            traceback.print_exc()
            return None

    def _load_csv_dataset(self, dataset_path: str, max_conversations: int):
        """Comprehensive CSV dataset loader"""
        df = pd.read_csv(dataset_path)
        print(f"CSV columns: {list(df.columns)}")
        
        conversations = []
        corpus = {}
        
        # Auto-detect CSV structure
        if 'conversation_id' in df.columns:
            # Multi-turn conversation format
            print("Detected multi-turn conversation format")
            grouped = df.groupby('conversation_id')
            count = 0
            for conv_id, group in grouped:
                if max_conversations and count >= max_conversations:
                    break
                
                turns = []
                for _, row in group.iterrows():
                    turn = {
                        'turn_id': row.get('turn_id', len(turns)),
                        'question': str(row.get('question', row.get('Q', row.get('query', '')))),
                        'answer': str(row.get('answer', row.get('A', row.get('response', '')))),
                        'context': str(row.get('context', ''))
                    }
                    turns.append(turn)
                
                if len(turns) >= 2:  # Multi-turn requirement
                    conversation = {
                        'conversation_id': str(conv_id),
                        'turns': turns,
                        'topic': str(group.iloc[0].get('topic', 'unknown'))
                    }
                    conversations.append(conversation)
                    count += 1
                    
                    # Add context to corpus if available
                    context_text = str(group.iloc[0].get('context', ''))
                    if context_text and len(context_text.strip()) > 10:
                        corpus[str(conv_id)] = context_text.strip()
        
        elif any(col in df.columns for col in ['question', 'Q', 'query']):
            # Single turn format - create artificial conversations
            print("Detected Q&A format - creating multi-turn conversations")
            question_col = next((col for col in ['question', 'Q', 'query'] if col in df.columns), None)
            answer_col = next((col for col in ['answer', 'A', 'response'] if col in df.columns), None)
            
            if question_col and answer_col:
                for i in range(0, min(len(df), max_conversations * 2), 2):
                    if i + 1 < len(df):
                        turns = [
                            {
                                'turn_id': 0,
                                'question': str(df.iloc[i][question_col]),
                                'answer': str(df.iloc[i][answer_col]),
                                'context': str(df.iloc[i].get('context', ''))
                            },
                            {
                                'turn_id': 1,
                                'question': str(df.iloc[i+1][question_col]),
                                'answer': str(df.iloc[i+1][answer_col]),
                                'context': str(df.iloc[i+1].get('context', ''))
                            }
                        ]
                        
                        conversation = {
                            'conversation_id': f'conv_{len(conversations)}',
                            'turns': turns,
                            'topic': str(df.iloc[i].get('topic', 'unknown'))
                        }
                        conversations.append(conversation)
                        
                        # Add context to corpus
                        context_text = str(df.iloc[i].get('context', ''))
                        if context_text and len(context_text.strip()) > 10:
                            corpus[f'conv_{len(conversations)-1}'] = context_text.strip()
        
        else:
            print("Warning: Could not detect CSV structure")
            print("Supported formats:")
            print("  Multi-turn: conversation_id, turn_id, question, answer, context")
            print("  Q&A pairs: question, answer, context")
        
        print(f"Successfully loaded {len(conversations)} conversations with {len(corpus)} corpus documents")
        return {'conversations': conversations, 'corpus': corpus}

    def _load_jsonl_dataset(self, dataset_path: str, max_conversations: int):
        """Comprehensive JSONL dataset loader"""
        print("Detected JSONL format")
        conversations = []
        corpus = {}
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            conv_count = 0
            qa_buffer = []
            
            for i, line in enumerate(f):
                if max_conversations and conv_count >= max_conversations:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Check if it's a conversation format
                    if 'turns' in data or 'conversation' in data:
                        # Direct conversation format
                        turns_data = data.get('turns', data.get('conversation', []))
                        if len(turns_data) >= 2:
                            turns = []
                            for t_idx, turn in enumerate(turns_data):
                                if isinstance(turn, dict):
                                    turns.append({
                                        'turn_id': t_idx,
                                        'question': str(turn.get('question', turn.get('Q', turn.get('query', '')))),
                                        'answer': str(turn.get('answer', turn.get('A', turn.get('response', '')))),
                                        'context': str(turn.get('context', ''))
                                    })
                            
                            if turns:
                                conversation = {
                                    'conversation_id': data.get('conversation_id', f'jsonl_conv_{conv_count}'),
                                    'turns': turns,
                                    'topic': str(data.get('topic', 'unknown'))
                                }
                                conversations.append(conversation)
                                conv_count += 1
                    
                    elif any(field in data for field in ['question', 'Q', 'query']):
                        # Single QA format - buffer for pairing
                        qa_item = {
                            'question': str(data.get('question', data.get('Q', data.get('query', '')))),
                            'answer': str(data.get('answer', data.get('A', data.get('response', '')))),
                            'context': str(data.get('context', '')),
                            'topic': str(data.get('topic', 'unknown'))
                        }
                        qa_buffer.append(qa_item)
                        
                        # Create conversation when we have 2 QAs
                        if len(qa_buffer) >= 2:
                            turns = [
                                {'turn_id': 0, **qa_buffer[0]},
                                {'turn_id': 1, **qa_buffer[1]}
                            ]
                            
                            conversation = {
                                'conversation_id': f'jsonl_conv_{conv_count}',
                                'turns': turns,
                                'topic': qa_buffer[0]['topic']
                            }
                            conversations.append(conversation)
                            conv_count += 1
                            
                            # Add context to corpus
                            if qa_buffer[0]['context'] and len(qa_buffer[0]['context'].strip()) > 10:
                                corpus[f'jsonl_conv_{conv_count-1}'] = qa_buffer[0]['context'].strip()
                            
                            qa_buffer = []  # Reset buffer
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {i+1}: {e}")
                    continue
        
        print(f"Successfully loaded {len(conversations)} conversations with {len(corpus)} corpus documents")
        return {'conversations': conversations, 'corpus': corpus}

    def _load_json_dataset(self, dataset_path: str, max_conversations: int):
        """Comprehensive JSON dataset loader"""
        print("Detected JSON format")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        corpus = {}
        
        if isinstance(data, dict):
            # Check for direct format
            if 'conversations' in data:
                conversations = data['conversations'][:max_conversations]
                corpus = data.get('corpus', {})
            elif 'data' in data:
                # Handle data wrapper
                conversations = self._convert_qa_to_conversations(data['data'], max_conversations)
            else:
                # Try to convert the dict itself
                conversations = self._convert_qa_to_conversations([data], max_conversations)
        
        elif isinstance(data, list):
            # List format - detect structure
            if len(data) > 0 and isinstance(data[0], dict):
                if 'turns' in data[0] or 'conversation' in data[0]:
                    # Direct conversation list
                    conversations = data[:max_conversations]
                else:
                    # QA list - convert to conversations
                    conversations = self._convert_qa_to_conversations(data, max_conversations)
        
        print(f"Successfully loaded {len(conversations)} conversations with {len(corpus)} corpus documents")
        return {'conversations': conversations, 'corpus': corpus}

    def _convert_qa_to_conversations(self, qa_data: List[Dict], max_conversations: int) -> List[Dict]:
        """Convert Q&A data to conversation format"""
        conversations = []
        
        for i in range(0, min(len(qa_data), max_conversations * 2), 2):
            if i + 1 < len(qa_data):
                turns = []
                for j, qa in enumerate([qa_data[i], qa_data[i+1]]):
                    turn = {
                        'turn_id': j,
                        'question': str(qa.get('question', qa.get('Q', qa.get('query', '')))),
                        'answer': str(qa.get('answer', qa.get('A', qa.get('response', '')))),
                        'context': str(qa.get('context', ''))
                    }
                    turns.append(turn)
                
                conversation = {
                    'conversation_id': f'json_conv_{len(conversations)}',
                    'turns': turns,
                    'topic': str(qa_data[i].get('topic', 'unknown'))
                }
                conversations.append(conversation)
        
        return conversations

    def _load_topicqa_dataset(self, max_conversations: int):
        """Load TopiOCQA dataset"""
        print("Loading TopiOCQA dataset...")
        dataset = load_dataset("McGill-NLP/TopiOCQA", trust_remote_code=True)
        
        # Group turns by conversation number  
        conversation_turns = {}
        count = 0
        
        for example in dataset['train']:
            conv_no = example['Conversation_no']
            turn_no = example['Turn_no']
            
            if conv_no not in conversation_turns:
                conversation_turns[conv_no] = {}
            
            conversation_turns[conv_no][turn_no] = {
                'question': str(example['Question']),
                'answer': str(example['Answer']),
                'topic': str(example['Topic']),
                'context': example.get('Context', []),
                'gold_passage': example.get('Gold_passage', {})
            }
            
            count += 1
            if count >= max_conversations * 15:
                break
        
        # Build conversations from grouped turns
        conversations = []
        corpus = {}
        
        for conv_no, turns_dict in list(conversation_turns.items())[:max_conversations]:
            sorted_turns = sorted(turns_dict.items(), key=lambda x: x[0])
            
            if len(sorted_turns) >= 2:
                turns = []
                for turn_no, turn_data in sorted_turns:
                    turns.append({
                        'turn_id': turn_no - 1,
                        'question': turn_data['question'],
                        'answer': turn_data['answer'],
                        'topic': turn_data['topic'],
                        'context': turn_data['context']
                    })
                
                conversations.append({
                    'conversation_id': conv_no,
                    'turns': turns,
                    'topic': sorted_turns[0][1]['topic']
                })
                
                # Add context to corpus - improved handling
                gold_passage = sorted_turns[0][1]['gold_passage']
                context_text = ""
                
                if isinstance(gold_passage, dict):
                    context_text = str(gold_passage.get('text', gold_passage.get('content', '')))
                elif isinstance(gold_passage, str):
                    context_text = gold_passage
                elif isinstance(gold_passage, list) and len(gold_passage) > 0:
                    context_text = ' '.join([str(item) for item in gold_passage])
                
                # Also add context from turns if available
                if not context_text:
                    for turn_no, turn_data in sorted_turns:
                        if turn_data['context']:
                            if isinstance(turn_data['context'], list):
                                context_text = ' '.join([str(item) for item in turn_data['context']])
                            else:
                                context_text = str(turn_data['context'])
                            break
                
                if context_text and len(context_text.strip()) > 10:
                    corpus[conv_no] = context_text.strip()
        
        print(f"Loaded {len(conversations)} conversations with {len(corpus)} corpus documents")
        return {'conversations': conversations, 'corpus': corpus}

    def _load_quac_dataset(self, max_conversations: int):
        """Load QuAC dataset with improved error handling"""
        print("Loading QuAC dataset...")
        
        try:
            dataset = load_dataset("quac", trust_remote_code=True)
            print(f"QuAC dataset loaded successfully")
            
            conversations = []
            corpus = {}
            
            # Debug: check the structure of the first example
            if len(dataset['validation']) > 0:
                first_example = dataset['validation'][0]
                print(f"First example keys: {list(first_example.keys())}")
                print(f"First example type: {type(first_example)}")
            
            # Convert to list to avoid iterator issues
            validation_data = list(dataset['validation'])
            print(f"Total validation examples: {len(validation_data)}")
            
            processed_count = 0
            for idx in range(min(max_conversations, len(validation_data))):
                try:
                    example = validation_data[idx]
                    conv_id = idx
                    
                    # Debug first few examples
                    if idx < 3:
                        print(f"Example {idx} type: {type(example)}")
                        if isinstance(example, dict):
                            print(f"Example {idx} has {len(example.get('questions', []))} questions")
                    
                    # Handle different possible structures
                    if isinstance(example, dict):
                        context = example.get('context', '')
                        questions = example.get('questions', [])
                        answers = example.get('answers', [])
                        title = example.get('wikipedia_page_title', f'conversation_{idx}')
                    else:
                        print(f"Unexpected example type at index {idx}: {type(example)}")
                        continue
                    
                    if not questions or not answers:
                        print(f"Skipping conversation {idx} - missing questions or answers")
                        continue
                    
                    turns = []
                    for i, (question, answer_data) in enumerate(zip(questions, answers)):
                        try:
                            # Handle different answer formats
                            if isinstance(answer_data, dict):
                                answer_text = answer_data.get('text', '')
                                if isinstance(answer_text, list) and len(answer_text) > 0:
                                    answer_text = answer_text[0]
                            elif isinstance(answer_data, list) and len(answer_data) > 0:
                                answer_text = str(answer_data[0])
                            else:
                                answer_text = str(answer_data)
                            
                            # Only include valid turns
                            if question and answer_text and answer_text != "CANNOTANSWER":
                                turns.append({
                                    'turn_id': i,
                                    'question': str(question),
                                    'answer': str(answer_text),
                                    'context': str(context)
                                })
                        except Exception as e:
                            print(f"Error processing turn {i} in conversation {idx}: {e}")
                            continue
                    
                    if len(turns) >= 2:  # Only include conversations with at least 2 turns
                        conversations.append({
                            'conversation_id': conv_id,
                            'turns': turns,
                            'title': str(title)
                        })
                        
                        # Add context to corpus
                        if context and len(str(context).strip()) > 10:
                            corpus[conv_id] = str(context).strip()
                        
                        processed_count += 1
                        
                    elif len(turns) == 1:
                        print(f"Conversation {idx} only has 1 turn, skipping")
                    else:
                        print(f"Conversation {idx} has no valid turns")
                
                except Exception as e:
                    print(f"Error processing conversation {idx}: {e}")
                    continue
            
            print(f"Processed {processed_count} conversations from {len(validation_data)} examples")
            print(f"Successfully loaded {len(conversations)} QuAC conversations with {len(corpus)} corpus documents")
            return {'conversations': conversations, 'corpus': corpus}
            
        except Exception as e:
            print(f"Failed to load QuAC dataset: {e}")
            traceback.print_exc()
            return None

    def calculate_ndcg_at_k(self, relevance_scores: List[float], k: int = 10) -> float:
        """Calculate NDCG@k"""
        scores = np.array(relevance_scores)[:k]
        if len(scores) == 0:
            return 0.0
        
        # DCG
        positions = np.arange(1, len(scores) + 1)
        dcg = np.sum(scores / np.log2(positions + 1))
        
        # IDCG
        ideal_scores = sorted(relevance_scores, reverse=True)
        ideal_dcg = np.sum(np.array(ideal_scores[:k]) / np.log2(np.arange(2, len(ideal_scores[:k]) + 2)))
        
        if ideal_dcg == 0:
            return 0.0
        return dcg / ideal_dcg

    def calculate_recall_at_k(self, retrieved_relevant: int, total_relevant: int, k: int = 10) -> float:
        """Calculate Recall@k"""
        if total_relevant == 0:
            return 0.0
        return retrieved_relevant / total_relevant

    def calculate_precision_at_k(self, retrieved_relevant: int, k: int = 10) -> float:
        """Calculate Precision@k"""
        if k == 0:
            return 0.0
        return retrieved_relevant / k

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 Score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_average_precision(self, relevance_scores: List[float]) -> float:
        """Calculate Average Precision"""
        if not relevance_scores:
            return 0.0
        
        relevant_docs = sum(1 for score in relevance_scores if score > 0)
        if relevant_docs == 0:
            return 0.0
        
        ap = 0.0
        relevant_found = 0
        
        for i, score in enumerate(relevance_scores):
            if score > 0:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                ap += precision_at_i
        
        return ap / relevant_docs

    def calculate_reciprocal_rank(self, relevance_scores: List[float]) -> float:
        """Calculate Reciprocal Rank"""
        for i, score in enumerate(relevance_scores):
            if score > 0:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_model_on_dataset(self, model_name: str, dataset_name: str, corpus_path: str = None, max_conversations: int = 30):
        """Evaluate a single model on a single dataset with optional external corpus"""
        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name} on {dataset_name.upper()}")
        if corpus_path:
            print(f"USING EXTERNAL CORPUS: {corpus_path}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            # Load model
            print(f"Loading model: {model_name}")
            model = SentenceTransformer(model_name)
            model_load_time = time.time() - start_time
            
            # Load dataset
            print(f"Loading dataset: {dataset_name}")
            data = self.load_dataset_safe(dataset_name, max_conversations)
            
            if not data:
                print(f"Failed to load dataset {dataset_name}")
                return None
            
            conversations = data['conversations']
            built_in_corpus = data['corpus']
            
            print(f"Loaded {len(conversations)} conversations")
            print(f"Built-in corpus documents: {len(built_in_corpus)}")
            
            # Load external corpus if specified
            final_corpus = built_in_corpus.copy()
            
            if corpus_path:
                external_corpus = self.load_external_corpus(corpus_path, self.max_corpus_docs)
                final_corpus.update(external_corpus)
                print(f"Combined corpus size: {len(final_corpus)} documents")
            
            # Encode corpus if available
            corpus_embeddings = None
            corpus_ids = None
            encoding_time = 0
            
            if final_corpus and len(final_corpus) > 0:
                corpus_encode_start = time.time()
                print("Encoding corpus...")
                
                corpus_texts = []
                corpus_ids = []
                
                for doc_id, doc_text in final_corpus.items():
                    if doc_text and len(str(doc_text).strip()) > 0:
                        corpus_texts.append(str(doc_text).strip())
                        corpus_ids.append(doc_id)
                
                if corpus_texts:
                    try:
                        corpus_embeddings = model.encode(corpus_texts, batch_size=self.batch_size, show_progress_bar=True)
                        encoding_time = time.time() - corpus_encode_start
                        print(f"Encoded {len(corpus_texts)} corpus documents")
                    except Exception as e:
                        print(f"Error encoding corpus: {e}")
                        corpus_embeddings = None
                else:
                    print("No valid corpus texts found")
            else:
                print("No corpus available for this evaluation")
            
            # Evaluate conversations
            print("Evaluating conversations...")
            eval_start = time.time()
            
            results = []
            for conv in conversations:
                if len(conv['turns']) < 2:
                    continue
                
                conv_result = self._evaluate_conversation(conv, corpus_embeddings, corpus_ids, model)
                results.append(conv_result)
            
            eval_time = time.time() - eval_start
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(results)
            
            total_time = time.time() - start_time
            
            # Compile results
            result = {
                'model': model_name,
                'dataset': dataset_name,
                'external_corpus': corpus_path if corpus_path else None,
                'conversations_evaluated': len(results),
                'corpus_size': len(final_corpus),
                'metrics': metrics,
                'timing': {
                    'model_load_time': model_load_time,
                    'encoding_time': encoding_time,
                    'evaluation_time': eval_time,
                    'total_time': total_time
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Print summary
            print(f"\nRESULTS for {model_name} on {dataset_name}:")
            print(f"  Conversations: {len(results)}")
            print(f"  Corpus Size: {len(final_corpus)}")
            print(f"  Avg Turns: {metrics.get('avg_turns_per_conv', 0):.1f}")
            print(f"  NDCG@10: {metrics.get('ndcg_10', 0):.4f}")
            print(f"  Recall@10: {metrics.get('recall_10', 0):.4f}")
            print(f"  Precision@10: {metrics.get('precision_10', 0):.4f}")
            print(f"  F1@10: {metrics.get('f1_10', 0):.4f}")
            print(f"  MAP: {metrics.get('map', 0):.4f}")
            print(f"  MRR: {metrics.get('mrr', 0):.4f}")
            print(f"  Context Coherence: {metrics.get('context_coherence', 0):.4f}")
            print(f"  Answer Relevance: {metrics.get('answer_relevance', 0):.4f}")
            print(f"  Total Time: {total_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"ERROR evaluating {model_name} on {dataset_name}: {e}")
            traceback.print_exc()
            return None

    def _evaluate_conversation(self, conversation: Dict, corpus_embeddings, corpus_ids, model) -> Dict:
        """Evaluate a single conversation with comprehensive metrics - FIXED THRESHOLDS"""
        turns_results = []
        context_history = ""
        
        for turn in conversation['turns']:
            current_query = turn['question']
            if context_history:
                contextualized_query = f"{context_history} {current_query}"
            else:
                contextualized_query = current_query
            
            # Encode query
            try:
                query_embedding = model.encode([contextualized_query], batch_size=1)
            except Exception as e:
                print(f"Error encoding query: {e}")
                query_embedding = None
            
            # Calculate retrieval metrics
            retrieval_metrics = {}
            if (corpus_embeddings is not None and 
                len(corpus_embeddings) > 0 and 
                query_embedding is not None):
                
                try:
                    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
                    top_indices = np.argsort(similarities)[::-1][:10]  # Top 10
                    
                    # FIXED: More reasonable similarity thresholds for real-world scenarios
                    high_threshold = 0.50   # Highly relevant
                    med_threshold = 0.35    # Moderately relevant  
                    low_threshold = 0.20    # Somewhat relevant
                    
                    relevance_scores = []
                    for idx in top_indices:
                        sim_score = similarities[idx]
                        if sim_score >= high_threshold:
                            relevance_score = 1.0  # Highly relevant
                        elif sim_score >= med_threshold:
                            relevance_score = 0.7  # Moderately relevant
                        elif sim_score >= low_threshold:
                            relevance_score = 0.3  # Somewhat relevant
                        else:
                            relevance_score = 0.0   # Not relevant
                        relevance_scores.append(relevance_score)
                    
                    # Calculate metrics with more realistic assumptions
                    retrieved_relevant = sum(1 for score in relevance_scores if score > 0)
                    
                    # More realistic total relevant calculation
                    corpus_size = len(corpus_embeddings)
                    if corpus_size <= 5:
                        total_relevant = max(1, corpus_size // 2)  # Small corpus: half might be relevant
                    elif corpus_size <= 20:
                        total_relevant = max(2, corpus_size // 3)  # Medium corpus: third might be relevant
                    else:
                        total_relevant = max(3, min(8, corpus_size // 8)) # Large corpus: up to 8 might be relevant
                    
                    # Ensure retrieved_relevant doesn't exceed total_relevant (fixes recall > 1.0)
                    retrieved_relevant = min(retrieved_relevant, total_relevant)
                    
                    retrieval_metrics = {
                        'ndcg_10': self.calculate_ndcg_at_k(relevance_scores, 10),
                        'recall_10': self.calculate_recall_at_k(retrieved_relevant, total_relevant, 10),
                        'precision_10': self.calculate_precision_at_k(retrieved_relevant, 10),
                        'ap': self.calculate_average_precision(relevance_scores),
                        'rr': self.calculate_reciprocal_rank(relevance_scores),
                        'max_similarity': float(np.max(similarities))
                    }
                    
                    # Calculate F1
                    precision = retrieval_metrics['precision_10']
                    recall = retrieval_metrics['recall_10']
                    retrieval_metrics['f1_10'] = self.calculate_f1_score(precision, recall)
                    
                    # Debug output for first few evaluations
                    if turn['turn_id'] == 0 and len(turns_results) == 0:
                        print(f"    Sample similarities: {similarities[:3]}")
                        print(f"    Retrieved relevant: {retrieved_relevant}/{total_relevant}")
                        print(f"    Top similarity: {similarities[top_indices[0]]:.3f}")
                        print(f"    Relevance scores: {relevance_scores[:3]}")
                    
                except Exception as e:
                    print(f"Error calculating retrieval metrics: {e}")
                    retrieval_metrics = {
                        'ndcg_10': 0.0, 'recall_10': 0.0, 'precision_10': 0.0,
                        'f1_10': 0.0, 'ap': 0.0, 'rr': 0.0, 'max_similarity': 0.0
                    }
            else:
                # If no corpus or embeddings, set all retrieval metrics to 0
                retrieval_metrics = {
                    'ndcg_10': 0.0, 'recall_10': 0.0, 'precision_10': 0.0,
                    'f1_10': 0.0, 'ap': 0.0, 'rr': 0.0, 'max_similarity': 0.0
                }
            
            # Calculate answer relevance
            answer_text = turn.get('answer', '')
            answer_relevance = 0.0
            if answer_text and answer_text != "UNANSWERABLE" and query_embedding is not None:
                try:
                    answer_embedding = model.encode([answer_text], batch_size=1)
                    answer_relevance = cosine_similarity(query_embedding, answer_embedding)[0][0]
                except Exception as e:
                    print(f"Error calculating answer relevance: {e}")
                    answer_relevance = 0.0
            
            turn_result = {
                'turn_id': turn['turn_id'],
                'question': current_query,
                'answer': answer_text,
                'answer_relevance': float(answer_relevance),
                'context_length': len(context_history.split()) if context_history else 0,
                **{k: float(v) for k, v in retrieval_metrics.items()}
            }
            
            turns_results.append(turn_result)
            context_history = contextualized_query
        
        return {
            'conversation_id': conversation['conversation_id'],
            'turns': turns_results,
            'topic': conversation.get('topic', conversation.get('title', 'unknown')),
            'total_turns': len(turns_results)
        }

    def _calculate_comprehensive_metrics(self, conversations_results: List[Dict]) -> Dict:
        """Calculate comprehensive aggregated metrics"""
        if not conversations_results:
            return {}
        
        all_turns = []
        turn_counts = []
        context_coherence_scores = []
        turn_progression_scores = []
        
        # Metric accumulators
        ndcg_scores = []
        recall_scores = []
        precision_scores = []
        f1_scores = []
        ap_scores = []
        rr_scores = []
        answer_relevance_scores = []
        max_similarity_scores = []
        
        for conv_result in conversations_results:
            turns = conv_result['turns']
            turn_counts.append(len(turns))
            
            for i, turn in enumerate(turns):
                all_turns.append(turn)
                
                # Collect retrieval metrics
                ndcg_scores.append(turn.get('ndcg_10', 0))
                recall_scores.append(turn.get('recall_10', 0))
                precision_scores.append(turn.get('precision_10', 0))
                f1_scores.append(turn.get('f1_10', 0))
                ap_scores.append(turn.get('ap', 0))
                rr_scores.append(turn.get('rr', 0))
                max_similarity_scores.append(turn.get('max_similarity', 0))
                
                # Answer relevance
                ans_rel = turn.get('answer_relevance', 0)
                if ans_rel > 0:
                    answer_relevance_scores.append(ans_rel)
                
                # Context coherence
                if i > 0:
                    prev_context_len = turns[i-1].get('context_length', 0)
                    curr_context_len = turn.get('context_length', 0)
                    coherence = min(1.0, curr_context_len / max(1, prev_context_len + 10))
                    context_coherence_scores.append(coherence)
                
                # Turn progression
                if i > 0:
                    prev_relevance = turns[i-1].get('answer_relevance', 0)
                    curr_relevance = turn.get('answer_relevance', 0)
                    progression = 1.0 if curr_relevance >= prev_relevance * 0.8 else 0.5
                    turn_progression_scores.append(progression)
        
        # Calculate averages with proper handling of empty lists
        def safe_mean(scores):
            return float(np.mean(scores)) if scores else 0.0
        
        return {
            'total_turns': len(all_turns),
            'avg_turns_per_conv': safe_mean(turn_counts),
            'ndcg_10': safe_mean(ndcg_scores),
            'recall_10': safe_mean(recall_scores),
            'precision_10': safe_mean(precision_scores),
            'f1_10': safe_mean(f1_scores),
            'map': safe_mean(ap_scores),  # MAP from Average Precision
            'mrr': safe_mean(rr_scores),  # MRR from Reciprocal Rank
            'context_coherence': safe_mean(context_coherence_scores),
            'turn_progression': safe_mean(turn_progression_scores),
            'answer_relevance': safe_mean(answer_relevance_scores),
            'retrieval_quality': safe_mean(max_similarity_scores),
            'conversations_evaluated': len(conversations_results)
        }

    def create_visualizations(self, all_results: List[Dict], output_dir: str):
        """Create ULTIMATE visualizations optimized for ALL scenarios"""
        if not all_results:
            print("No results to visualize")
            return
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame([{
            'Model': r['model'].split('/')[-1] if '/' in r['model'] else r['model'],  # Shorter names
            'Full_Model': r['model'],
            'Dataset': r['dataset'].upper(),
            'Corpus': r.get('external_corpus', 'Built-in') or 'Built-in',
            'NDCG@10': r['metrics'].get('ndcg_10', 0),
            'Recall@10': r['metrics'].get('recall_10', 0),
            'Precision@10': r['metrics'].get('precision_10', 0),
            'F1@10': r['metrics'].get('f1_10', 0),
            'MAP': r['metrics'].get('map', 0),
            'MRR': r['metrics'].get('mrr', 0),
            'Context_Coherence': r['metrics'].get('context_coherence', 0),
            'Answer_Relevance': r['metrics'].get('answer_relevance', 0),
            'Total_Time': r['timing']['total_time'],
            'Corpus_Size': r.get('corpus_size', 0)
        } for r in all_results])
        
        print(f"\nCreating visualizations in {output_dir}/graphs/...")
        graphs_dir = os.path.join(output_dir, 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Determine scenario
        num_models = df['Model'].nunique()
        num_datasets = df['Dataset'].nunique()
        num_corpora = df['Corpus'].nunique()
        
        print(f"Generating charts for {num_models} model(s), {num_datasets} dataset(s), {num_corpora} corpus/corpora")
        
        # 1. Main performance overview - adaptive
        self._create_metrics_overview(df, graphs_dir, num_models, num_datasets)
        
        # 2. Individual metric charts (if multiple models)
        if num_models > 1:
            self._create_individual_metric_charts(df, graphs_dir)
        
        # 3. Multi-metric comparison (if multiple models)
        if num_models > 1:
            self._create_multi_metric_chart(df, graphs_dir)
        
        # 4. Dataset-specific analysis (if multiple datasets)
        if num_datasets > 1:
            self._create_dataset_specific_charts(df, graphs_dir)
        
        # 5. Top performers chart (if multiple models)
        if num_models > 1:
            self._create_top_performers_chart(df, graphs_dir)
        
        # 6. Corpus impact analysis (if multiple corpora or external corpus)
        if num_corpora > 1 or any('Built-in' not in corpus for corpus in df['Corpus']):
            self._create_corpus_analysis(df, graphs_dir)
        
        # 7. Performance vs time analysis
        self._create_performance_time_chart(df, graphs_dir, num_models)
        
        # 8. Detailed metrics breakdown
        self._create_detailed_breakdown(df, graphs_dir, num_models)
        
        # 9. Create comprehensive PDF report
        self._create_pdf_report(df, graphs_dir, num_models, num_datasets, num_corpora)
        
        print(f"All visualizations saved to {graphs_dir}/")
        print("Generated charts:")
        print("  - metrics_overview.png")
        if num_models > 1:
            print("  - individual_metrics_comparison.png")
            print("  - multi_metric_comparison.png")
            print("  - top_performers.png")
        if num_datasets > 1:
            print("  - dataset_specific_charts.png")
        if num_corpora > 1:
            print("  - corpus_analysis.png")
        print("  - performance_time_analysis.png")
        print("  - detailed_breakdown.png")
        print("  - benchmark_report.pdf")

    def _create_metrics_overview(self, df: pd.DataFrame, output_dir: str, num_models: int, num_datasets: int):
        """Create adaptive main metrics overview"""
        key_metrics = ['NDCG@10', 'Recall@10', 'Precision@10', 'F1@10', 'MAP', 'MRR']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Overview', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            ax = axes[i]
            
            if num_models == 1:
                # Single model: show by dataset if multiple, otherwise single bar
                if num_datasets > 1:
                    dataset_perf = df.groupby('Dataset')[metric].mean()
                    colors = sns.color_palette("husl", len(dataset_perf))
                    bars = ax.bar(dataset_perf.index, dataset_perf.values, color=colors, alpha=0.8)
                    ax.set_xticklabels(dataset_perf.index, rotation=45, ha='right')
                else:
                    # Single model, single dataset
                    bars = ax.bar([metric], [df[metric].iloc[0]], color='steelblue', alpha=0.8)
            else:
                # Multiple models: show model comparison
                model_perf = df.groupby('Model')[metric].mean()
                colors = sns.color_palette("husl", len(model_perf))
                bars = ax.bar(range(len(model_perf)), model_perf.values, color=colors, alpha=0.8)
                ax.set_xticks(range(len(model_perf)))
                ax.set_xticklabels(model_perf.index, rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_individual_metric_charts(self, df: pd.DataFrame, output_dir: str):
        """Create individual bar charts for each metric"""
        key_metrics = ['NDCG@10', 'Recall@10', 'Precision@10', 'F1@10', 'MAP', 'MRR']
        
        # Average across datasets for each model
        avg_df = df.groupby('Model')[key_metrics].mean().sort_values('NDCG@10', ascending=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Individual Metric Performance Comparison', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        colors = sns.color_palette("husl", len(avg_df))
        
        for i, metric in enumerate(key_metrics):
            ax = axes[i]
            bars = ax.bar(range(len(avg_df)), avg_df[metric], color=colors)
            ax.set_title(f'{metric} Performance', fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(avg_df)))
            ax.set_xticklabels(avg_df.index, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'individual_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_multi_metric_chart(self, df: pd.DataFrame, output_dir: str):
        """Create multi-metric comparison chart"""
        key_metrics = ['NDCG@10', 'Recall@10', 'Precision@10', 'F1@10', 'MAP', 'MRR']
        
        # Get top 8 models by average NDCG
        avg_df = df.groupby('Model')[key_metrics].mean()
        top_models = avg_df.sort_values('NDCG@10', ascending=False).head(8)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x = np.arange(len(top_models))
        width = 0.12
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, metric in enumerate(key_metrics):
            offset = (i - len(key_metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, top_models[metric], width, 
                         label=metric, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:  # Only show label if bar is visible
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', 
                           fontsize=7, rotation=0)
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Multi-Metric Performance Comparison (Top 8 Models)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_models.index, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multi_metric_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_dataset_specific_charts(self, df: pd.DataFrame, output_dir: str):
        """Create dataset-specific comparison charts"""
        datasets = df['Dataset'].unique()
        key_metrics = ['NDCG@10', 'F1@10', 'MAP', 'MRR']
        
        for dataset in datasets:
            dataset_df = df[df['Dataset'] == dataset]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if len(dataset_df) == 1:
                # Single model for this dataset
                model_name = dataset_df['Model'].iloc[0]
                values = [dataset_df[metric].iloc[0] for metric in key_metrics]
                colors = sns.color_palette("husl", len(key_metrics))
                bars = ax.bar(key_metrics, values, color=colors, alpha=0.8)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                ax.set_title(f'{dataset} Dataset - {model_name}', fontsize=14, fontweight='bold')
                ax.set_ylabel('Score', fontweight='bold')
                
            else:
                # Multiple models for this dataset
                dataset_avg = dataset_df.groupby('Model')[key_metrics].mean().sort_values('NDCG@10', ascending=False)
                x = np.arange(len(dataset_avg))
                width = 0.2
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                
                for i, metric in enumerate(key_metrics):
                    offset = (i - len(key_metrics)/2 + 0.5) * width
                    bars = ax.bar(x + offset, dataset_avg[metric], width, 
                                 label=metric, color=colors[i], alpha=0.8)
                
                ax.set_xlabel('Models', fontweight='bold')
                ax.set_ylabel('Score', fontweight='bold')
                ax.set_title(f'{dataset} Dataset - Model Comparison', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(dataset_avg.index, rotation=45, ha='right')
                ax.legend()
            
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{dataset.lower()}_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _create_top_performers_chart(self, df: pd.DataFrame, output_dir: str):
        """Create top performers summary chart"""
        key_metrics = ['NDCG@10', 'Recall@10', 'Precision@10', 'F1@10', 'MAP', 'MRR']
        
        # Get top performer for each metric
        avg_df = df.groupby('Model')[key_metrics].mean()
        top_performers = {}
        top_scores = {}
        
        for metric in key_metrics:
            top_model = avg_df[metric].idxmax()
            top_score = avg_df[metric].max()
            top_performers[metric] = top_model
            top_scores[metric] = top_score
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = list(top_scores.keys())
        scores = list(top_scores.values())
        models = list(top_performers.values())
        
        colors = sns.color_palette("husl", len(metrics))
        bars = ax.bar(metrics, scores, color=colors, alpha=0.8)
        
        # Add value labels and model names
        for i, (bar, model, score) in enumerate(zip(bars, models, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   model[:15] + '...' if len(model) > 15 else model, 
                   ha='center', va='center', fontsize=8, 
                   rotation=90, color='white', fontweight='bold')
        
        ax.set_ylabel('Best Score', fontweight='bold')
        ax.set_title('Top Performers by Metric', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_performers.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_corpus_analysis(self, df: pd.DataFrame, output_dir: str):
        """Create corpus impact analysis"""
        key_metrics = ['NDCG@10', 'Recall@10', 'Precision@10', 'F1@10']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Chart 1: Performance by corpus type
        ax1 = axes[0]
        corpus_perf = df.groupby('Corpus')[key_metrics].mean()
        
        x = np.arange(len(corpus_perf))
        width = 0.2
        colors = sns.color_palette("husl", len(key_metrics))
        
        for i, metric in enumerate(key_metrics):
            offset = (i - len(key_metrics)/2 + 0.5) * width
            bars = ax1.bar(x + offset, corpus_perf[metric], width, 
                          label=metric, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Corpus Type', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Performance by Corpus Type', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(corpus_perf.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Chart 2: Corpus size impact
        ax2 = axes[1]
        scatter = ax2.scatter(df['Corpus_Size'], df['F1@10'], 
                             c=df['NDCG@10'], cmap='viridis', s=100, alpha=0.7)
        
        ax2.set_xlabel('Corpus Size', fontweight='bold')
        ax2.set_ylabel('F1@10 Score', fontweight='bold')
        ax2.set_title('Corpus Size Impact on Performance', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('NDCG@10 Score', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'corpus_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_time_chart(self, df: pd.DataFrame, output_dir: str, num_models: int):
        """Create adaptive performance vs time analysis chart"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Chart 1: Performance vs Time scatter
        ax1 = axes[0]
        
        if num_models == 1:
            # Single model: show by dataset if available
            if len(df) > 1:
                scatter = ax1.scatter(df['Total_Time'], df['NDCG@10'], 
                                     s=100, alpha=0.7, c=df['F1@10'], cmap='viridis')
                
                for i, row in df.iterrows():
                    ax1.annotate(row['Dataset'], (row['Total_Time'], row['NDCG@10']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
            else:
                # Single model, single dataset
                scatter = ax1.scatter([df['Total_Time'].iloc[0]], [df['NDCG@10'].iloc[0]], 
                                     s=200, alpha=0.8, c=['steelblue'])
                model_name = df['Model'].iloc[0]
                ax1.annotate(model_name[:15], (df['Total_Time'].iloc[0], df['NDCG@10'].iloc[0]),
                            xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        else:
            # Multiple models
            avg_df = df.groupby('Model').agg({
                'NDCG@10': 'mean',
                'F1@10': 'mean', 
                'Total_Time': 'mean'
            }).reset_index()
            
            scatter = ax1.scatter(avg_df['Total_Time'], avg_df['NDCG@10'], 
                                 s=100, alpha=0.7, c=avg_df['F1@10'], cmap='viridis')
            
            for i, model in enumerate(avg_df['Model']):
                ax1.annotate(model[:10] + '...' if len(model) > 10 else model, 
                            (avg_df['Total_Time'].iloc[i], avg_df['NDCG@10'].iloc[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Time (seconds)', fontweight='bold')
        ax1.set_ylabel('NDCG@10', fontweight='bold')
        ax1.set_title('Performance vs Speed Trade-off', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Chart 2: Time comparison
        ax2 = axes[1]
        
        if num_models == 1:
            # Single model: show time breakdown by dataset if available
            if len(df) > 1:
                times = df.groupby('Dataset')['Total_Time'].mean()
                colors = sns.color_palette("husl", len(times))
                bars = ax2.bar(times.index, times.values, color=colors, alpha=0.8)
                ax2.set_xticklabels(times.index, rotation=45, ha='right')
                ax2.set_title('Processing Time by Dataset', fontweight='bold')
            else:
                # Single dataset: show time as single bar
                time_val = df['Total_Time'].iloc[0]
                bars = ax2.bar(['Processing Time'], [time_val], color='steelblue', alpha=0.8)
                ax2.set_title('Processing Time', fontweight='bold')
        else:
            # Multiple models: time comparison
            avg_time = df.groupby('Model')['Total_Time'].mean().sort_values()
            colors = sns.color_palette("husl", len(avg_time))
            bars = ax2.bar(range(len(avg_time)), avg_time.values, color=colors, alpha=0.8)
            ax2.set_xticks(range(len(avg_time)))
            ax2.set_xticklabels([name[:8] + '...' if len(name) > 8 else name for name in avg_time.index], 
                               rotation=45, ha='right')
            ax2.set_title('Processing Time Comparison', fontweight='bold')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2.set_ylabel('Time (seconds)', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_time_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_detailed_breakdown(self, df: pd.DataFrame, output_dir: str, num_models: int):
        """Create detailed metrics breakdown"""
        all_metrics = ['NDCG@10', 'Recall@10', 'Precision@10', 'F1@10', 'MAP', 'MRR', 
                      'Context_Coherence', 'Answer_Relevance']
        
        if num_models == 1:
            # Single model: detailed profile
            fig, ax = plt.subplots(figsize=(12, 8))
            values = df[all_metrics].mean() if len(df) > 1 else df[all_metrics].iloc[0]
            colors = sns.color_palette("husl", len(all_metrics))
            
            bars = ax.bar(all_metrics, values, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, metric in zip(bars, all_metrics):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            model_name = df['Model'].iloc[0]
            ax.set_title(f'Complete Metrics Profile - {model_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_xticklabels(all_metrics, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
        else:
            # Multiple models: heatmap
            avg_df = df.groupby('Model')[all_metrics].mean()
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create heatmap
            sns.heatmap(avg_df.T, annot=True, fmt='.3f', cmap='viridis', 
                       ax=ax, cbar_kws={'label': 'Score'})
            
            ax.set_title('Comprehensive Metrics Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Models', fontweight='bold')
            ax.set_ylabel('Metrics', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detailed_breakdown.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_pdf_report(self, df: pd.DataFrame, output_dir: str, num_models: int, num_datasets: int, num_corpora: int):
        """Create adaptive PDF report"""
        pdf_path = os.path.join(output_dir, 'benchmark_report.pdf')
        
        with PdfPages(pdf_path) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.8, 'Ultimate RAG Benchmark Report', 
                   ha='center', va='center', fontsize=24, fontweight='bold')
            ax.text(0.5, 0.7, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.6, f'Models Tested: {num_models}', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.55, f'Datasets: {num_datasets}', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.5, f'Corpora: {num_corpora}', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.45, f'Total Evaluations: {len(df)}', 
                   ha='center', va='center', fontsize=12)
            
            # Add key results
            best_f1 = df.loc[df['F1@10'].idxmax()]
            ax.text(0.5, 0.35, 'Key Results:', 
                   ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 0.3, f'Best F1@10: {best_f1["F1@10"]:.3f} ({best_f1["Model"]})', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.25, f'Best NDCG@10: {df["NDCG@10"].max():.3f}', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.2, f'Avg Processing Time: {df["Total_Time"].mean():.1f}s', 
                   ha='center', va='center', fontsize=12)
            
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add charts to PDF
            chart_files = ['metrics_overview.png', 'detailed_breakdown.png', 'performance_time_analysis.png']
            
            if num_models > 1:
                chart_files.extend(['individual_metrics_comparison.png', 'multi_metric_comparison.png', 'top_performers.png'])
            
            if num_datasets > 1:
                for dataset in df['Dataset'].unique():
                    chart_file = f'{dataset.lower()}_comparison.png'
                    if os.path.exists(os.path.join(output_dir, chart_file)):
                        chart_files.append(chart_file)
            
            if num_corpora > 1:
                chart_files.append('corpus_analysis.png')
            
            for chart_file in chart_files:
                chart_path = os.path.join(output_dir, chart_file)
                if os.path.exists(chart_path):
                    img = plt.imread(chart_path)
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        print(f"PDF report created: {pdf_path}")

    def display_comparative_results(self, all_results: List[Dict]):
        """Display comparative results in a nice CLI table with corpus information"""
        if not all_results:
            print("No results to display")
            return
        
        # Prepare data for table
        table_data = []
        for result in all_results:
            metrics = result['metrics']
            corpus_info = os.path.basename(result.get('external_corpus', 'Built-in')) if result.get('external_corpus') else 'Built-in'
            
            row = [
                result['model'][:25] + '...' if len(result['model']) > 25 else result['model'],
                result['dataset'].upper(),
                corpus_info[:15] + '...' if len(corpus_info) > 15 else corpus_info,
                result.get('corpus_size', 0),
                f"{metrics.get('ndcg_10', 0):.3f}",
                f"{metrics.get('recall_10', 0):.3f}",
                f"{metrics.get('precision_10', 0):.3f}",
                f"{metrics.get('f1_10', 0):.3f}",
                f"{metrics.get('map', 0):.3f}",
                f"{metrics.get('mrr', 0):.3f}",
                f"{result['timing']['total_time']:.1f}s"
            ]
            table_data.append(row)
        
        headers = [
            "Model", "Dataset", "Corpus", "Corp_Size", "NDCG@10", "Recall@10", 
            "Precision@10", "F1@10", "MAP", "MRR", "Time"
        ]
        
        print(f"\n{'='*160}")
        print("ULTIMATE RAG BENCHMARK RESULTS")
        print(f"{'='*160}")
        print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))
        
        # Show top performers if multiple models
        if len(set(r['model'] for r in all_results)) > 1:
            df = pd.DataFrame([{
                'Model': r['model'],
                'NDCG@10': r['metrics'].get('ndcg_10', 0),
                'F1@10': r['metrics'].get('f1_10', 0),
                'MAP': r['metrics'].get('map', 0),
                'MRR': r['metrics'].get('mrr', 0)
            } for r in all_results])
            
            print(f"\n{'='*80}")
            print("TOP PERFORMERS BY METRIC")
            print(f"{'='*80}")
            
            top_performers = []
            for metric in ['NDCG@10', 'F1@10', 'MAP', 'MRR']:
                top_model = df.groupby('Model')[metric].mean().idxmax()
                top_score = df.groupby('Model')[metric].mean().max()
                top_performers.append([metric, top_model[:40], f"{top_score:.4f}"])
            
            print(tabulate(top_performers, headers=["Metric", "Best Model", "Score"], tablefmt="grid"))

    def run_full_benchmark(self, output_dir: str = "./benchmark_results", max_conversations: int = 25, 
                          create_graphs: bool = True, corpus_path: str = None):
        """Run comprehensive benchmark on all models and datasets with optional external corpus"""
        print(f"\nSTARTING ULTIMATE RAG BENCHMARK")
        print(f"{'='*80}")
        print(f"Models to test: {len(self.embedding_models)}")
        print(f"Datasets to test: {len(self.datasets)}")
        print(f"Total evaluations: {len(self.embedding_models) * len(self.datasets)}")
        print(f"Max conversations per evaluation: {max_conversations}")
        if corpus_path:
            print(f"External corpus: {corpus_path}")
            print(f"Max corpus docs: {self.max_corpus_docs if self.max_corpus_docs else 'No limit'}")
        print(f"Metrics: NDCG@10, Recall@10, Precision@10, F1@10, MAP, MRR, Context Coherence, Answer Relevance")
        print(f"Visualizations: {'Enabled' if create_graphs else 'Disabled'}")
        print(f"{'='*80}")
        
        benchmark_start = time.time()
        all_results = []
        
        for i, model_name in enumerate(self.embedding_models):
            print(f"\nMODEL {i+1}/{len(self.embedding_models)}: {model_name}")
            
            for j, dataset_name in enumerate(self.datasets):
                print(f"   Dataset {j+1}/{len(self.datasets)}: {dataset_name}")
                
                result = self.evaluate_model_on_dataset(model_name, dataset_name, corpus_path, max_conversations)
                
                if result:
                    all_results.append(result)
        
        total_benchmark_time = time.time() - benchmark_start
        
        # Display comparative results
        self.display_comparative_results(all_results)
        
        # Create comprehensive results
        benchmark_results = {
            'benchmark_info': {
                'models_tested': len(self.embedding_models),
                'datasets_tested': len(self.datasets),
                'successful_evaluations': len(all_results),
                'external_corpus': corpus_path,
                'max_corpus_docs': self.max_corpus_docs,
                'total_time_seconds': total_benchmark_time,
                'timestamp': datetime.now().isoformat(),
                'max_conversations_per_eval': max_conversations
            },
            'individual_results': all_results
        }
        
        # Save detailed results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        with open(os.path.join(output_dir, 'full_benchmark_results.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Create comparison tables
        self._create_comparison_tables(all_results, output_dir)
        
        # Create visualizations
        if create_graphs:
            self.create_visualizations(all_results, output_dir)
        
        print(f"\nBENCHMARK COMPLETED!")
        print(f"Total time: {total_benchmark_time/60:.1f} minutes")
        print(f"Results saved to {output_dir}/")
        if create_graphs:
            print(f"Visualizations saved to {output_dir}/graphs/")
            print(f"PDF report: {output_dir}/graphs/benchmark_report.pdf")
        
        return benchmark_results

    def _create_comparison_tables(self, results: List[Dict], output_dir: str):
        """Create enhanced comparison tables with corpus information"""
        if not results:
            return
        
        # Create comprehensive DataFrame
        data = []
        for result in results:
            row = {
                'Model': result['model'],
                'Dataset': result['dataset'],
                'External_Corpus': result.get('external_corpus', 'None'),
                'Corpus_Size': result.get('corpus_size', 0),
                'Conversations': result['conversations_evaluated'],
                'Avg_Turns': result['metrics'].get('avg_turns_per_conv', 0),
                'NDCG@10': result['metrics'].get('ndcg_10', 0),
                'Recall@10': result['metrics'].get('recall_10', 0),
                'Precision@10': result['metrics'].get('precision_10', 0),
                'F1@10': result['metrics'].get('f1_10', 0),
                'MAP': result['metrics'].get('map', 0),
                'MRR': result['metrics'].get('mrr', 0),
                'Context_Coherence': result['metrics'].get('context_coherence', 0),
                'Turn_Progression': result['metrics'].get('turn_progression', 0),
                'Answer_Relevance': result['metrics'].get('answer_relevance', 0),
                'Retrieval_Quality': result['metrics'].get('retrieval_quality', 0),
                'Total_Time': result['timing']['total_time']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save comprehensive CSV
        df.to_csv(os.path.join(output_dir, 'comprehensive_benchmark_results.csv'), index=False)
        
        # Create rankings for all metrics (only if multiple models)
        if df['Model'].nunique() > 1:
            ranking_metrics = ['NDCG@10', 'Recall@10', 'Precision@10', 'F1@10', 'MAP', 'MRR', 
                              'Context_Coherence', 'Answer_Relevance']
            
            rankings = {}
            for metric in ranking_metrics:
                rankings[metric] = df.groupby('Model')[metric].mean().sort_values(ascending=False).head(10)
            
            # Save enhanced rankings
            with open(os.path.join(output_dir, 'comprehensive_rankings.json'), 'w') as f:
                json.dump({k: v.to_dict() for k, v in rankings.items()}, f, indent=2)
        
        # Create dataset-specific results
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            dataset_df.to_csv(os.path.join(output_dir, f'{dataset}_comprehensive_results.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description="Ultimate Multi-Model Multi-Turn RAG Benchmark with Complete Visualizations")
    parser.add_argument("-o", "--output_dir", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--max_conversations", type=int, default=25, help="Max conversations per evaluation")
    parser.add_argument("--models", type=str, nargs='+', help="Specific models to test")
    parser.add_argument("--datasets", type=str, nargs='+', 
                       help="Datasets: 'quac', 'topicqa', or custom files (.json/.csv/.jsonl)")
    parser.add_argument("--corpus", type=str, help="External corpus file path (.json/.csv/.jsonl)")
    parser.add_argument("--max_corpus_docs", type=int, help="Maximum corpus documents to load")
    parser.add_argument("--no-graphs", action="store_true", help="Skip creating visualizations")
    
    args = parser.parse_args()
    
    benchmark = UltimateRAGBenchmark(args.batch_size)
    
    # Set external corpus parameters
    if args.max_corpus_docs:
        benchmark.max_corpus_docs = args.max_corpus_docs
    
    # Override models/datasets if specified
    if args.models:
        benchmark.embedding_models = args.models
        print(f"Using custom models: {args.models}")
    
    if args.datasets:
        benchmark.datasets = args.datasets
        print(f"Using custom datasets: {args.datasets}")
        
        # Validate custom dataset files
        for dataset in args.datasets:
            if dataset not in ['quac', 'topicqa'] and not os.path.exists(dataset):
                print(f"Warning: Dataset file not found: {dataset}")
    
    if args.corpus:
        if not os.path.exists(args.corpus):
            print(f"Error: Corpus file not found: {args.corpus}")
            return
        print(f"Using external corpus: {args.corpus}")
    
    # Run benchmark
    results = benchmark.run_full_benchmark(
        args.output_dir, 
        args.max_conversations, 
        not args.no_graphs,
        args.corpus
    )
    
    print(f"\nBenchmark complete! Check {args.output_dir} for detailed results.")
    if not args.no_graphs:
        print(f"Visualizations and PDF report available in {args.output_dir}/graphs/")
    
    # print("\nSupported Formats:")
    # print("DATASETS: 'quac', 'topicqa', custom .json/.csv/.jsonl")
    # print("CORPUS: .json/.csv/.jsonl with flexible field detection")
    # print("CSV: conversation_id,turn_id,question,answer,context OR question,answer,context")
    # print("JSON: {'conversations': [...]} OR [qa_pairs] OR nested structures")
    # print("JSONL: One conversation/QA per line")

if __name__ == "__main__":
    main()