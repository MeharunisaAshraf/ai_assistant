import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Optional
import time
import requests
from abc import ABC, abstractmethod


class BaseIntentClassifier(ABC):
    """ Abstract base class for intent classifiers """

    @abstractmethod
    def classify_intent(self, query: str, **kwargs) -> Dict:
        pass


class SimilarityIntentClassifier(BaseIntentClassifier):
    def __init__(self, intents_data: Dict):
        """Simple intent classifier using sentence transformers + FAISS"""

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intents = intents_data['intents']

        # Build searchable texts and metadata
        self.texts = []
        self.metadata = []
        self._build_index()

        print(f"Loaded {len(self.texts)} intent patterns")

    def _build_index(self):
        """Build FAISS index from intent data"""

        for intent in self.intents:
            # Add description as main searchable text
            self.texts.append(intent['description'])
            self.metadata.append(intent)

            # Add each pattern as searchable text
            for pattern in intent['patterns']:
                self.texts.append(pattern)
                self.metadata.append(intent)

        # Create embeddings and FAISS index
        print("Creating embeddings...")
        embeddings = self.model.encode(self.texts, show_progress_bar=True)

        # Build FAISS index for cosine similarity
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

    def classify_intent(self, query: str, top_k: int = 3) -> Dict:
        """Find best matching intents for user query"""

        start_time = time.time()

        # Get query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar texts
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)

        # Group results by page_id to avoid duplicates
        seen_pages = set()
        results = []

        for sim, idx in zip(similarities[0], indices[0]):
            if sim < 0.5:  # Skip low similarity matches
                continue

            intent = self.metadata[idx]
            page_id = intent['page_id']

            # Skip if we already have this page
            if page_id in seen_pages:
                continue

            seen_pages.add(page_id)
            results.append({
                'page_id': page_id,
                'confidence': float(sim),
                'category': intent['category'],
                'intent_type': intent['intent_type'],
                'response': intent['response'],
                'next_steps': intent.get('next_steps', []),
                'quick_actions': intent.get('quick_actions', [])
            })

            if len(results) >= top_k:
                break

        end_time = time.time()

        return {
            'classifier_type': 'similarity',
            'results': results,
            'inference_time_s': round((end_time - start_time), 3),
            'query': query,
            'top_confidence': results[0]['confidence'] if results else 0.0
        }

    def add_intent(self, intent_data: Dict):
        """Add new intent to classifier"""

        self.intents.append(intent_data)

        # Add new texts
        new_texts = [intent_data['description']] + intent_data['patterns']
        new_metadata = [intent_data] * len(new_texts)

        # Create embeddings
        new_embeddings = self.model.encode(new_texts)
        faiss.normalize_L2(new_embeddings)

        # Add to index
        self.index.add(new_embeddings.astype('float32'))
        self.texts.extend(new_texts)
        self.metadata.extend(new_metadata)

        print(f"Added intent: {intent_data['page_id']}")


class GenerativeIntentClassifier(BaseIntentClassifier):
    def __init__(self, intents_data: Dict, ollama_url: str = "http://localhost:11434", model_name: str = "llama3.2:3b", temperature: float = 0.1):
        """ Generative intent classifier using Ollama Llama 3.2 3B """

        self.intents_data = intents_data
        self.intents = intents_data['intents']
        self.temperature = temperature
        self.ollama_url = ollama_url
        self.model_name = model_name

        self._verify_ollama_connection()
        self.prompt_template = self._build_prompt_template()

        print(f"Generative classifier initialized with {self.model_name}")
        print(f"Ollama URL: {self.ollama_url}")
        print(f"Total intents: {len(self.intents)}")

    def _verify_ollama_connection(self):
        """Verify Ollama is running and model is available"""

        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]

            if not any(self.model_name in name for name in model_names):
                print(f"Model {self.model_name} not found. Available models: {model_names}")
                print(f"Pulling {self.model_name}...")
                self._pull_model()

            print(f"Successfully connected to Ollama with {self.model_name}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.ollama_url}: {str(e)}")

    def _pull_model(self):
        """Pull the model if not available"""

        try:
            pull_data = {"name": self.model_name}
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json=pull_data,
                timeout=300  # 5 minutes timeout for model pull
            )
            response.raise_for_status()
            print(f"Successfully pulled {self.model_name}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to pull model {self.model_name}: {str(e)}")

    def _build_prompt_template(self) -> str:
        """Build prompt template"""

        # Create intent descriptions
        intent_descriptions = []
        for intent in self.intents:
            description = intent.get('description', intent['response'][:150] + '...')
            intent_descriptions.append(
                f"- {intent['page_id']}: {description}"
            )

        intents_text = '\n'.join(intent_descriptions)

        # System prompt
        system_prompt = f"""You are a property management assistant that STRICTLY classifies user queries to EXACT page intents.

AVAILABLE INTENTS (MUST USE EXACT page_id):
{intents_text}

RULES:
1. Match the query to ONE exact page_id from the list above.
2. NEVER modify the page_id.
3. Output ONLY valid JSON in this format: {{"page_id": "exact_page_id", "confidence": 0.85}}
4. The confidence should be between 0.0 and 1.0.
5. Do not include any explanation or additional text, only the JSON response.
"""

        return system_prompt

    def classify_intent(self, query: str, top_k: int = 3) -> Dict:
        """Classify user intent using Ollama"""

        start_time = time.time()
        try:
            # Build prompt
            system_prompt = self.prompt_template

            # Prepare Ollama request
            ollama_data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query to classify: {query}"}
                ],
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 50, # Max tokens to generate
                },
                "format": "json",  # This forces JSON output
                "stream": False
            }

            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=ollama_data,
                timeout=30
            )
            response.raise_for_status()

            # Parse Ollama response
            ollama_result = response.json()
            message = ollama_result.get('message', {})
            generated_text = message.get('content', '').strip()

            # Parse result
            result = json.loads(generated_text)
            intent_data = self._get_intent_data(result['page_id'])
            if not intent_data:
                raise ValueError(f"Invalid page_id: {result['page_id']}")

            classification_result = {
                'page_id': result['page_id'],
                'confidence': result.get('confidence', 0.8),
                'category': intent_data['category'],
                'intent_type': intent_data['intent_type'],
                'response': intent_data['response'],
                'next_steps': intent_data.get('next_steps', []),
                'quick_actions': intent_data.get('quick_actions', []),
                'method': 'generative'
            }

            end_time = time.time()
            return {
                'classifier_type': 'generative',
                'results': [classification_result],
                'inference_time_s': round(end_time - start_time, 3),
                'query': query,
                'top_confidence': classification_result['confidence'],
                'model_name': self.model_name
            }


        except Exception as e:
            print(f"Classification error: {e}")
            return {
                'classifier_type': 'generative',
                'results': [],
                'inference_time_s': round((time.time() - start_time), 3),
                'query': query,
                'top_confidence': 0.0,
                'model_name': self.model_name,
                'error': str(e)
            }

    def _get_intent_data(self, page_id: str) -> Optional[Dict]:
        """Get intent data by page_id"""

        for intent in self.intents:
            if intent['page_id'] == page_id:
                return intent
        return None

    def batch_classify(self, queries: List[str]) -> List[Dict]:
        """Classify multiple queries"""

        return [self.classify_intent(query) for query in queries]

    def add_new_intent(self, intent_data: Dict):
        """Add new intent"""

        self.intents.append(intent_data)
        self.prompt_template = self._build_prompt_template()
        print(f"Added new intent: {intent_data['page_id']}")

    def test_connection(self) -> bool:
        """Test if Ollama connection is working"""

        try:
            test_data = {
                "model": self.model_name,
                "prompt": "Test connection. Respond with: OK",
                "stream": False
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=test_data,
                timeout=10
            )
            response.raise_for_status()
            return True
        except:
            return False


class HybridIntentClassifier(BaseIntentClassifier):
    def __init__(self, intents_data: Dict, similarity_threshold: float = 0.85,
                 ollama_url: str = "http://localhost:11434", model_name: str = "llama3.2:3b",
                 temperature: float = 0.1):
        """ Hybrid intent classifier that uses similarity first, falls back to generative """

        self.intents_data = intents_data
        self.similarity_threshold = similarity_threshold

        # Initialize both classifiers
        print("Initializing Similarity Classifier...")
        self.similarity_classifier = SimilarityIntentClassifier(intents_data)

        print("Initializing Generative Classifier...")
        self.generative_classifier = GenerativeIntentClassifier(
            intents_data, ollama_url, model_name, temperature
        )

        print(f"Hybrid classifier ready with threshold: {similarity_threshold}")

    def classify_intent(self, query: str, top_k: int = 3) -> Dict:
        start_time = time.time()

        # Step 1: Try similarity classifier first
        similarity_result = self.similarity_classifier.classify_intent(query, top_k=1)

        # Check if similarity classifier has high confidence result
        if similarity_result['results'] and similarity_result['top_confidence'] >= self.similarity_threshold:

            end_time = time.time()
            return {
                'classifier_type': 'hybrid_similarity',
                'primary_method': 'similarity',
                'fallback_used': False,
                'results': similarity_result['results'][:top_k],
                'inference_time_s': round((end_time - start_time), 3),
                'similarity_time_s': similarity_result['inference_time_s'],
                'generative_time_s': 0,
                'query': query,
                'top_confidence': similarity_result['top_confidence'],
                'similarity_threshold': self.similarity_threshold,
                'decision_reason': f"Similarity confidence {similarity_result['top_confidence']:.3f} >= threshold {self.similarity_threshold}"
            }

        # Step 2: Fall back to generative classifier
        print(
            f"Similarity confidence {similarity_result['top_confidence']:.3f} < threshold {self.similarity_threshold}, using generative...")

        generative_result = self.generative_classifier.classify_intent(query, top_k)

        end_time = time.time()

        return {
            'classifier_type': 'hybrid_generative',
            'primary_method': 'generative',
            'fallback_used': True,
            'results': generative_result['results'],
            'inference_time_s': round((end_time - start_time), 3),
            'similarity_time_s': similarity_result['inference_time_s'],
            'generative_time_s': generative_result['inference_time_s'],
            'query': query,
            'top_confidence': generative_result['top_confidence'],
            'similarity_threshold': self.similarity_threshold,
            'decision_reason': f"Similarity confidence {similarity_result['top_confidence']:.3f} < threshold {self.similarity_threshold}, used generative",
            'model_name': generative_result.get('model_name', 'unknown'),
            'similarity_best_match': similarity_result['results'][0] if similarity_result['results'] else None
        }

    def batch_classify(self, queries: List[str], top_k: int = 3) -> List[Dict]:
        """Classify multiple queries using hybrid approach"""

        results = []
        similarity_count = 0
        generative_count = 0

        for query in queries:
            result = self.classify_intent(query, top_k)
            results.append(result)

            if result['primary_method'] == 'similarity':
                similarity_count += 1
            else:
                generative_count += 1

        print(f"Batch classification complete: {similarity_count} similarity, {generative_count} generative")
        return results

    def add_intent(self, intent_data: Dict):
        """Add new intent to both classifiers"""

        self.similarity_classifier.add_intent(intent_data)
        self.generative_classifier.add_new_intent(intent_data)
        print(f"Added intent to hybrid classifier: {intent_data['page_id']}")

    def update_similarity_threshold(self, new_threshold: float):
        """Update the similarity threshold"""

        old_threshold = self.similarity_threshold
        self.similarity_threshold = new_threshold
        print(f"Updated similarity threshold: {old_threshold} -> {new_threshold}")


# Example usage and testing
if __name__ == "__main__":
    # Load your intent data
    current_dir = Path(__file__).resolve().parent
    intents_path = current_dir.parent / 'data' / 'intents.json'
    with open(intents_path, 'r') as f:
        intents = json.load(f)

    # Initialize hybrid classifier
    classifier = HybridIntentClassifier(
        intents,
        similarity_threshold=0.85,
        ollama_url = "http://localhost:11434",
        model_name = "llama3.2:3b",
    )

    # Test queries
    test_queries = [
        "Need to add owners before uploading lease",
        "set rent amount",
        "add new amenities to my property",
        "how do I confirm the address?",
        "what fees can I add?",
    ]

    print("\n" + "=" * 60)
    print("TESTING HYBRID INTENT CLASSIFIER")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = classifier.classify_intent(query)
        print(result)