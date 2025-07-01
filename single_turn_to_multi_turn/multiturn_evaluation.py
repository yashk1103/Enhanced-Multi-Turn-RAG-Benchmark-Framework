
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import random
from beir import util
from beir.datasets.data_loader import GenericDataLoader

class SyntheticMultiTurnGenerator:
    """Generate synthetic multi-turn conversations from single-turn data"""
    
    def __init__(self):
        self.follow_up_templates = [
            "Can you tell me more about {topic}?",
            "What else should I know about {topic}?",
            "How does {topic} relate to {context}?",
            "Can you explain {topic} in more detail?",
            "What are some examples of {topic}?",
            "Why is {topic} important?",
            "What are the benefits of {topic}?",
            "What problems does {topic} solve?",
        ]
        
        self.clarification_templates = [
            "I didn't understand the part about {concept}",
            "Can you clarify what you mean by {concept}?",
            "What exactly is {concept}?",
            "How does {concept} work?",
        ]
        
        self.comparison_templates = [
            "How does this compare to {alternative}?",
            "What's the difference between this and {alternative}?",
            "Which is better: this or {alternative}?",
        ]

    def create_multiturn_from_single(self, queries: Dict, corpus: Dict, qrels: Dict, num_turns: int = 3) -> Dict:
        """Convert single-turn queries into multi-turn conversations"""
        multiturn_data = []
        
        for query_id, query_text in queries.items():
            if query_id not in qrels:
                continue
                
            # Get relevant documents for context
            relevant_docs = list(qrels[query_id].keys())
            if not relevant_docs:
                continue
                
            # Extract topics/concepts from query and relevant docs
            topics = self._extract_topics(query_text, corpus, relevant_docs)
            
            # Create conversation turns
            conversation = {
                'conversation_id': query_id,
                'original_query': query_text,
                'turns': [],
                'relevant_docs': relevant_docs,
                'qrels': qrels[query_id]
            }
            
            # First turn is the original query
            conversation['turns'].append({
                'turn_id': 0,
                'query': query_text,
                'context': "",
                'query_type': 'initial'
            })
            
            # Generate follow-up turns
            context = query_text
            for turn in range(1, num_turns):
                follow_up = self._generate_follow_up(query_text, topics, context, turn)
                conversation['turns'].append({
                    'turn_id': turn,
                    'query': follow_up,
                    'context': context,
                    'query_type': self._classify_query_type(follow_up)
                })
                context += f" {follow_up}"
            
            multiturn_data.append(conversation)
        
        return {'conversations': multiturn_data, 'corpus': corpus}

    def _extract_topics(self, query: str, corpus: Dict, relevant_docs: List[str]) -> List[str]:
        """Extract key topics from query and relevant documents"""
        topics = []
        
        # Simple keyword extraction from query
        query_words = query.lower().split()
        topics.extend([word for word in query_words if len(word) > 3])
        
        # Extract topics from relevant documents (first 100 chars)
        for doc_id in relevant_docs[:2]:  # Use top 2 relevant docs
            if doc_id in corpus:
                # Handle BEIR corpus structure: {"title": "...", "text": "..."}
                doc_content = corpus[doc_id]
                if isinstance(doc_content, dict):
                    doc_text = (doc_content.get("title", "") + " " + doc_content.get("text", ""))[:100].lower()
                else:
                    doc_text = str(doc_content)[:100].lower()
                doc_words = doc_text.split()
                topics.extend([word for word in doc_words if len(word) > 4])
        
        # Remove duplicates and return top topics
        unique_topics = list(set(topics))
        return unique_topics[:5]

    def _generate_follow_up(self, original_query: str, topics: List[str], context: str, turn_num: int) -> str:
        """Generate follow-up questions"""
        if not topics:
            topics = ["this topic"]
        
        topic = random.choice(topics)
        
        if turn_num == 1:
            # Follow-up questions
            template = random.choice(self.follow_up_templates)
            return template.format(topic=topic, context=context)
        else:
            # Mix of clarification and comparison
            if random.random() < 0.5:
                template = random.choice(self.clarification_templates)
                return template.format(concept=topic)
            else:
                alternative = random.choice(topics) if len(topics) > 1 else "other options"
                template = random.choice(self.comparison_templates)
                return template.format(alternative=alternative)

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['more', 'else', 'additional']):
            return 'elaboration'
        elif any(word in query_lower for word in ['clarify', 'explain', 'what exactly']):
            return 'clarification'
        elif any(word in query_lower for word in ['compare', 'difference', 'better']):
            return 'comparison'
        else:
            return 'follow_up'

class MultiTurnRAGEvaluator:
    def __init__(self, model_name: str, batch_size: int = 16):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.generator = SyntheticMultiTurnGenerator()

    def evaluate_multiturn_retrieval(self, dataset_name: str, output_dir: str, num_turns: int = 3):
        """Evaluate multi-turn retrieval using synthetic conversations"""
        print(f"\nMulti-Turn RAG Evaluation")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Number of turns: {num_turns}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load BEIR dataset
        data_path = self._download_beir_dataset(dataset_name)
        if not data_path:
            return None
        
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        print(f"Loaded {len(corpus)} documents, {len(queries)} queries")
        
        # Generate multi-turn conversations
        print("Generating synthetic multi-turn conversations...")
        multiturn_data = self.generator.create_multiturn_from_single(
            queries, corpus, qrels, num_turns
        )
        
        # Encode corpus
        print("Encoding corpus...")
        corpus_texts = [corpus[doc_id] for doc_id in corpus.keys()]
        corpus_ids = list(corpus.keys())
        corpus_embeddings = self.model.encode(corpus_texts, batch_size=self.batch_size, show_progress_bar=True)
        
        # Evaluate each conversation
        print("Evaluating multi-turn retrieval...")
        results = []
        
        for conv in multiturn_data['conversations'][:50]:  # Limit for demo
            conv_results = self._evaluate_conversation(conv, corpus_embeddings, corpus_ids)
            results.append(conv_results)
        
        # Calculate metrics
        metrics = self._calculate_multiturn_metrics(results)
        
        total_time = time.time() - start_time
        
        # Print results
        print(f"\nMulti-Turn Retrieval Results:")
        print(f"Context Preservation: {metrics['context_preservation']:.4f}")
        print(f"Turn-by-Turn NDCG@10: {metrics['avg_ndcg']:.4f}")
        print(f"Turn-by-Turn Recall@10: {metrics['avg_recall']:.4f}")
        print(f"Final Turn NDCG@10: {metrics['final_turn_ndcg']:.4f}")
        print(f"Conversation Success Rate: {metrics['conversation_success_rate']:.4f}")
        print(f"Time: {total_time:.1f}s")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_data = {
            "dataset": dataset_name,
            "num_turns": num_turns,
            "metrics": metrics,
            "time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
            "sample_conversations": results[:5]  # Save sample results
        }
        
        output_file = os.path.join(output_dir, f"{dataset_name}_multiturn_results.json")
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return results_data

    def _download_beir_dataset(self, dataset_name: str, data_dir: str = "./beir_data"):
        """Download BEIR dataset"""
        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, data_dir)
            return data_path
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    def _evaluate_conversation(self, conversation: Dict, corpus_embeddings: np.ndarray, corpus_ids: List[str]) -> Dict:
        """Evaluate a single multi-turn conversation"""
        turns_results = []
        cumulative_context = ""
        
        relevant_docs = set(conversation['relevant_docs'])
        qrels = conversation['qrels']
        
        for turn in conversation['turns']:
            # Create contextualized query
            if turn['turn_id'] == 0:
                contextualized_query = turn['query']
            else:
                contextualized_query = f"{cumulative_context} {turn['query']}"
            
            # Encode query
            query_embedding = self.model.encode([contextualized_query], batch_size=1)
            
            # Compute similarities
            similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:10]
            
            # Calculate metrics for this turn
            retrieved_docs = [corpus_ids[idx] for idx in top_indices]
            retrieved_relevant = len(set(retrieved_docs) & relevant_docs)
            
            recall_10 = retrieved_relevant / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
            
            # NDCG@10
            relevance_scores = []
            for doc_id in retrieved_docs:
                if doc_id in qrels:
                    relevance_scores.append(qrels[doc_id])
                else:
                    relevance_scores.append(0.0)
            
            ndcg_10 = self._calculate_ndcg_at_k(relevance_scores, 10)
            
            turn_result = {
                'turn_id': turn['turn_id'],
                'query': turn['query'],
                'contextualized_query': contextualized_query,
                'query_type': turn['query_type'],
                'ndcg_10': ndcg_10,
                'recall_10': recall_10,
                'retrieved_docs': retrieved_docs[:5],  # Top 5 for analysis
                'retrieved_relevant_count': retrieved_relevant
            }
            
            turns_results.append(turn_result)
            cumulative_context = contextualized_query
        
        return {
            'conversation_id': conversation['conversation_id'],
            'turns': turns_results,
            'relevant_docs_count': len(relevant_docs)
        }

    def _calculate_multiturn_metrics(self, conversations_results: List[Dict]) -> Dict:
        """Calculate multi-turn specific metrics"""
        all_ndcg = []
        all_recall = []
        final_turn_ndcg = []
        context_preservation_scores = []
        successful_conversations = 0
        
        for conv_result in conversations_results:
            turns = conv_result['turns']
            
            # Collect all turn metrics
            turn_ndcgs = [turn['ndcg_10'] for turn in turns]
            turn_recalls = [turn['recall_10'] for turn in turns]
            
            all_ndcg.extend(turn_ndcgs)
            all_recall.extend(turn_recalls)
            
            # Final turn performance
            if turns:
                final_turn_ndcg.append(turns[-1]['ndcg_10'])
            
            # Context preservation: does performance maintain or improve over turns?
            if len(turns) > 1:
                first_turn_perf = turns[0]['ndcg_10']
                last_turn_perf = turns[-1]['ndcg_10']
                context_preservation = 1.0 if last_turn_perf >= first_turn_perf * 0.8 else 0.0
                context_preservation_scores.append(context_preservation)
            
            # Conversation success: at least one turn retrieves relevant docs
            if any(turn['retrieved_relevant_count'] > 0 for turn in turns):
                successful_conversations += 1
        
        return {
            'avg_ndcg': np.mean(all_ndcg) if all_ndcg else 0.0,
            'avg_recall': np.mean(all_recall) if all_recall else 0.0,
            'final_turn_ndcg': np.mean(final_turn_ndcg) if final_turn_ndcg else 0.0,
            'context_preservation': np.mean(context_preservation_scores) if context_preservation_scores else 0.0,
            'conversation_success_rate': successful_conversations / len(conversations_results) if conversations_results else 0.0,
            'total_conversations': len(conversations_results),
            'total_turns': len(all_ndcg)
        }

    def _calculate_ndcg_at_k(self, relevance_scores: List[float], k: int) -> float:
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

def main():
    parser = argparse.ArgumentParser(description="Multi-Turn RAG Evaluation")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--dataset", type=str, default="nfcorpus", help="BEIR dataset name")
    parser.add_argument("--output_dir", type=str, default="./multiturn_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--num_turns", type=int, default=3, help="Number of turns per conversation")
    
    args = parser.parse_args()
    
    evaluator = MultiTurnRAGEvaluator(args.model, args.batch_size)
    
    result = evaluator.evaluate_multiturn_retrieval(
        args.dataset, 
        args.output_dir, 
        args.num_turns
    )
    
    if result:
        print(f"\nMulti-turn evaluation completed successfully!")
        print(f"Check {args.output_dir} for detailed results")
        
        # Print quick summary
        metrics = result['metrics']
        print(f"\nQuick Summary:")
        print(f"  Context Preservation: {metrics['context_preservation']:.1%}")
        print(f"  Conversation Success: {metrics['conversation_success_rate']:.1%}")
        print(f"  Average NDCG@10: {metrics['avg_ndcg']:.4f}")
    else:
        print("Evaluation failed")

if __name__ == "__main__":
    main()