import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import uuid
import json
import os
from collections import defaultdict # Needed for hybrid search normalization
from simple_lexical_search import SimpleSearchEngine

# Define a type alias for clarity
ItemId = str
ItemData = Dict[str, Any]
Metadata = Dict[str, Any]
FilterType = Dict[str, Any]

class SimpleVectorStore:
    """
    A simple in-memory store for vectors, text, and metadata,
    supporting vector similarity, TF-IDF lexical search, filtering, updates,
    and saving/loading to JSON.
    """

    def __init__(self, vector_dim: Optional[int] = None, stopwords: Optional[set] = None):
        """
        Initializes the store.

        Args:
            vector_dim: Expected dimension for all vectors. If provided,
                        vectors added must match this dimension.
            stopwords: Optional set of stopwords for the lexical search engine.
        """
        self.data: Dict[ItemId, ItemData] = {}
        self.vector_dim: Optional[int] = vector_dim
        # Initialize the lexical search engine
        self.lexical_engine = SimpleSearchEngine(stopwords=stopwords)
        # print(f"Initialized SimpleVectorStore (Expected vector dim: {vector_dim or 'Any'})")

    # ----- Core Data Management -----

    def add_item(self,
                 vector: np.ndarray,
                 text: str,
                 metadata: Metadata,
                 item_id: Optional[ItemId] = None) -> ItemId:
        """Adds or overwrites an item in the store."""
        is_update = False
        if item_id is None:
            item_id = str(uuid.uuid4())
        elif item_id in self.data:
            print(f"Warning: Overwriting existing item with ID: {item_id}")
            is_update = True # Treat overwrite as an update for indexing

        # ... (Keep vector dimension validation as is) ...
        current_vector_dim = vector.shape[0] if isinstance(vector, np.ndarray) else None
        if current_vector_dim is None:
             raise ValueError("Input vector must be a numpy array.")

        if self.vector_dim is not None and current_vector_dim != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {current_vector_dim}")
        elif self.vector_dim is None and not self.data: # First item sets the dim if not preset
             self.vector_dim = current_vector_dim
             print(f"Inferred vector dimension from first item: {self.vector_dim}")
        elif self.vector_dim is None and self.data: # Check against inferred dim
             first_item_vec = next(iter(self.data.values()))["vector"]
             inferred_dim = first_item_vec.shape[0]
             if current_vector_dim != inferred_dim:
                 raise ValueError(f"Vector dimension mismatch. Expected {inferred_dim} (inferred), got {current_vector_dim}")
             self.vector_dim = inferred_dim # Solidify inferred dim
        elif self.vector_dim is not None and current_vector_dim != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {current_vector_dim}")


        # Store data first
        self.data[item_id] = {
            "vector": vector.astype(np.float32),
            "text": text,
            "metadata": metadata.copy()
        }

        # Index/Re-index in lexical engine
        # Note: SimpleSearchEngine's index_document now handles removal if ID exists
        self.lexical_engine.index_document(item_id, text)

        return item_id

    def get_item(self, item_id: ItemId) -> Optional[ItemData]:
        """Retrieves an item by its ID."""
        return self.data.get(item_id)

    def delete_item(self, item_id: ItemId) -> bool:
        """Deletes an item by its ID. Returns True if deleted, False otherwise."""
        if item_id in self.data:
            del self.data[item_id]
            # Also remove from the lexical index
            self.lexical_engine.remove_document(item_id)
            # If store becomes empty, reset inferred dimension? Optional.
            # if not self.data:
            #     self.vector_dim = None # Or keep the last known dimension? Decide based on desired behavior.
            return True
        return False

    def update_vector(self, item_id: ItemId, vector: np.ndarray) -> bool:
        """Updates the vector for a specific item."""
        # ... (Keep update_vector logic as is) ...
        if item_id in self.data:
            current_vector_dim = vector.shape[0] if isinstance(vector, np.ndarray) else None
            if current_vector_dim is None:
                 raise ValueError("Input vector must be a numpy array.")
            if self.vector_dim is not None and current_vector_dim != self.vector_dim:
                 raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {current_vector_dim}")
            self.data[item_id]["vector"] = vector.astype(np.float32)
            return True
        return False

    def update_text(self, item_id: ItemId, text: str) -> bool:
        """Updates the text for a specific item."""
        if item_id in self.data:
            self.data[item_id]["text"] = text
            # Re-index the document in the lexical engine
            self.lexical_engine.index_document(item_id, text)
            return True
        return False

    def update_metadata(self, item_id: ItemId, metadata_update: Metadata, replace: bool = False) -> bool:
        """Updates the metadata for a specific item. Merges by default."""
        # ... (Keep update_metadata logic as is - no impact on lexical index) ...
        if item_id in self.data:
            if replace:
                self.data[item_id]["metadata"] = metadata_update.copy()
            else:
                self.data[item_id]["metadata"].update(metadata_update)
            return True
        return False


    # ----- Filtering -----

    def _matches_filters(self, item_metadata: Metadata, filters: FilterType) -> bool:
        # ... (Keep _matches_filters logic as is) ...
        if not filters:
            return True
        for key, value in filters.items():
            try:
                if key.endswith('__gt'):
                    actual_key = key[:-4]
                    if not (actual_key in item_metadata and isinstance(item_metadata[actual_key], (int, float)) and item_metadata[actual_key] > value): return False
                elif key.endswith('__lt'):
                    actual_key = key[:-4]
                    if not (actual_key in item_metadata and isinstance(item_metadata[actual_key], (int, float)) and item_metadata[actual_key] < value): return False
                elif key.endswith('__gte'):
                    actual_key = key[:-5]
                    if not (actual_key in item_metadata and isinstance(item_metadata[actual_key], (int, float)) and item_metadata[actual_key] >= value): return False
                elif key.endswith('__lte'):
                    actual_key = key[:-5]
                    if not (actual_key in item_metadata and isinstance(item_metadata[actual_key], (int, float)) and item_metadata[actual_key] <= value): return False
                elif key.endswith('__in'):
                     actual_key = key[:-4]
                     if not (actual_key in item_metadata and isinstance(value, (list, tuple, set)) and item_metadata[actual_key] in value): return False
                elif key.endswith('__contains'):
                     actual_key = key[:-10]
                     if not (actual_key in item_metadata and hasattr(item_metadata[actual_key], '__contains__') and value in item_metadata[actual_key]): return False
                else:
                    if key not in item_metadata or item_metadata[key] != value: return False
            except (TypeError, KeyError): return False
        return True


    def _get_filtered_ids(self, filters: Optional[FilterType]) -> List[ItemId]:
        # ... (Keep _get_filtered_ids logic as is) ...
        if not filters:
            return list(self.data.keys())
        return [
            item_id for item_id, item_data in self.data.items()
            if self._matches_filters(item_data["metadata"], filters)
        ]

    # ----- Search Methods -----

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # ... (Keep _cosine_similarity logic as is) ...
        vec1 = vec1.astype(np.float32)
        vec2 = vec2.astype(np.float32)
        if vec1.shape != vec2.shape:
             raise ValueError(f"Cannot compute similarity for vectors with shapes {vec1.shape} and {vec2.shape}")
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))


    def search_vector(self,
                      query_vector: np.ndarray,
                      k: int = 5,
                      filters: Optional[FilterType] = None) -> List[Tuple[ItemId, float]]:
        # ... (Keep search_vector logic as is) ...
        query_vector = query_vector.astype(np.float32)
        query_dim = query_vector.shape[0]
        if self.vector_dim is not None and query_dim != self.vector_dim: raise ValueError(f"Query vector dimension mismatch. Expected {self.vector_dim}, got {query_dim}")
        elif self.vector_dim is None and self.data:
             first_item_vec = next(iter(self.data.values()))["vector"]
             inferred_dim = first_item_vec.shape[0]
             if query_dim != inferred_dim: raise ValueError(f"Query vector dimension mismatch. Expected {inferred_dim} (inferred), got {query_dim}")
        elif self.vector_dim is None and not self.data: return []

        candidate_ids = self._get_filtered_ids(filters)
        if not candidate_ids: return []

        results = []
        for item_id in candidate_ids:
            item_vector = self.data[item_id]["vector"]
            similarity = self._cosine_similarity(query_vector, item_vector)
            results.append((item_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def search_lexical(self,
                       query_text: str,
                       k: int = 5,
                       filters: Optional[FilterType] = None) -> List[Tuple[ItemId, float]]:
        """
        Performs lexical search using the integrated TF-IDF based SimpleSearchEngine.

        Args:
            query_text: The text string to search for.
            k: Max number of results to return.
            filters: Optional metadata filters to apply *after* retrieving search results.

        Returns:
            A list of tuples: (item_id, tf_idf_score), sorted by score descending.
        """
        candidate_ids = self._get_filtered_ids(filters)
        if not candidate_ids:
            return []
        candidate_id_set = set(candidate_ids) # For faster lookup

        # Perform search using the lexical engine - search across all indexed docs
        # Request more than k initially, as filtering happens afterwards.
        # Requesting all possible results ensures we don't miss candidates due to TF-IDF rank.
        all_lexical_results = self.lexical_engine.search(query_text, top_n=len(self.data))

        # Filter results based on the metadata filters
        filtered_results = [
            (item_id, score) for item_id, score in all_lexical_results
            if item_id in candidate_id_set
        ]

        # Return the top k of the filtered results
        # Sorting is already done by the search engine, but filtering might change order slightly if scores were identical.
        # Re-sorting ensures consistency, though often redundant.
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:k]


    def search_hybrid(self,
                      query_vector: np.ndarray,
                      query_text: str,
                      k: int = 5,
                      filters: Optional[FilterType] = None,
                      vector_weight: float = 0.5,
                      # lexical_scorer removed - we now use TF-IDF internally
                      ) -> List[Tuple[ItemId, float]]:
        """
        Performs a hybrid search combining vector similarity and TF-IDF lexical relevance.
        Scores are combined after normalizing both to a 0-1 range within the candidate set.

        Args:
            query_vector: The vector part of the query.
            query_text: The text part of the query.
            k: Number of results to return.
            filters: Optional metadata filters.
            vector_weight: Weight given to normalized vector similarity score (0.0 to 1.0).
                           Normalized lexical score weight will be (1.0 - vector_weight).

        Returns:
            A list of tuples: (item_id, combined_score), sorted by score descending.
        """
        if not (0.0 <= vector_weight <= 1.0):
            raise ValueError("vector_weight must be between 0.0 and 1.0")

        # --- Vector Dimension Checks (same as search_vector) ---
        query_vector = query_vector.astype(np.float32)
        query_dim = query_vector.shape[0]
        if self.vector_dim is not None and query_dim != self.vector_dim: raise ValueError(f"Query vector dimension mismatch. Expected {self.vector_dim}, got {query_dim}")
        elif self.vector_dim is None and self.data:
             first_item_vec = next(iter(self.data.values()))["vector"]
             inferred_dim = first_item_vec.shape[0]
             if query_dim != inferred_dim: raise ValueError(f"Query vector dimension mismatch. Expected {inferred_dim} (inferred), got {query_dim}")
        elif self.vector_dim is None and not self.data: return []
        # --- End Vector Dimension Checks ---

        candidate_ids = self._get_filtered_ids(filters)
        if not candidate_ids:
            return []

        vector_scores = {}
        lexical_scores = {}
        combined_results = []
        lexical_weight = 1.0 - vector_weight

        # --- Calculate Scores for Candidates ---
        min_vec_score, max_vec_score = float('inf'), float('-inf')
        min_lex_score, max_lex_score = float('inf'), float('-inf')

        # Preprocess query once for lexical scoring
        processed_query_terms = self.lexical_engine._preprocess(query_text)

        for item_id in candidate_ids:
            item_data = self.data[item_id]

            # 1. Vector Score
            vec_score = self._cosine_similarity(query_vector, item_data["vector"])
            vector_scores[item_id] = vec_score
            min_vec_score = min(min_vec_score, vec_score)
            max_vec_score = max(max_vec_score, vec_score)

            # 2. Lexical Score (TF-IDF)
            # Use the helper method from SimpleSearchEngine if available
            if hasattr(self.lexical_engine, '_get_tf_idf_score'):
                 lex_score = self.lexical_engine._get_tf_idf_score(processed_query_terms, item_id)
            else: # Fallback if method not added or import failed
                 lex_score = 0.0 # Or implement basic scoring here as fallback
            lexical_scores[item_id] = lex_score
            min_lex_score = min(min_lex_score, lex_score)
            max_lex_score = max(max_lex_score, lex_score)


        # --- Normalize and Combine Scores ---
        vec_range = max_vec_score - min_vec_score
        lex_range = max_lex_score - min_lex_score

        for item_id in candidate_ids:
            # Normalize vector score (cosine similarity -1 to 1 -> 0 to 1)
            # Note: Original code normalized (score + 1)/2. Let's stick to that.
            norm_vec_score = (vector_scores[item_id] + 1.0) / 2.0

            # Normalize lexical score (TF-IDF 0 to N -> 0 to 1 based on range in candidates)
            if lex_range > 0:
                norm_lex_score = (lexical_scores[item_id] - min_lex_score) / lex_range
            elif max_lex_score > 0: # All candidates have the same non-zero score
                norm_lex_score = 1.0
            else: # All candidates have zero score
                norm_lex_score = 0.0

            combined_score = (vector_weight * norm_vec_score) + (lexical_weight * norm_lex_score)
            combined_results.append((item_id, combined_score))

        # Sort by combined score (descending) and return top k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]


    # ----- Persistence -----

    def save(self, filename_base: str):
        """
        Saves the current state of the vector store (vectors, text, metadata)
        to a JSON file. The lexical index is NOT saved and will be rebuilt on load.

        Args:
            filename_base: The base name for the file (e.g., "my_store").
                           ".json" will be appended automatically.
        """
        filepath = filename_base + ".json"
        print(f"Saving vector store data to {filepath}...")
        # ... (Keep serialization logic as is) ...
        serializable_data = {}
        for item_id, item_data in self.data.items():
            serializable_data[item_id] = {
                "vector": item_data["vector"].tolist(),
                "text": item_data["text"],
                "metadata": item_data["metadata"]
            }
        save_package = {
            "vector_dim": self.vector_dim,
            "data": serializable_data
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_package, f, indent=4)
            print(f"Successfully saved {len(self.data)} items. Lexical index will be rebuilt on load.")
        except IOError as e:
            print(f"Error saving store to {filepath}: {e}")
        except TypeError as e:
             print(f"Error serializing data for saving: {e}. Ensure metadata contains JSON-compatible types.")


    @classmethod
    def load(cls, filename_base: str, stopwords: Optional[set] = None) -> 'SimpleVectorStore':
        """
        Loads a vector store from a JSON file and rebuilds the lexical index.

        Args:
            filename_base: The base name of the file to load (e.g., "my_store").
                           ".json" will be appended automatically.
            stopwords: Optional set of stopwords for the lexical search engine.

        Returns:
            A new SimpleVectorStore instance populated with the loaded data.

        Raises:
            FileNotFoundError, ValueError, IOError as before.
        """
        filepath = filename_base + ".json"
        print(f"Loading vector store data from {filepath}...")
        # ... (Keep file reading and basic validation as is) ...
        if not os.path.exists(filepath): raise FileNotFoundError(f"Save file not found: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f: load_package = json.load(f)
        except json.JSONDecodeError as e: raise ValueError(f"Error decoding JSON from {filepath}: {e}")
        except IOError as e: raise IOError(f"Error reading file {filepath}: {e}")
        if "vector_dim" not in load_package or "data" not in load_package: raise ValueError(f"Invalid file format in {filepath}. Missing 'vector_dim' or 'data' key.")

        loaded_vector_dim = load_package["vector_dim"]
        loaded_data = load_package["data"]

        # Create a new store instance with the loaded dimension AND stopwords
        store = cls(vector_dim=loaded_vector_dim, stopwords=stopwords)

        print("Populating store and rebuilding lexical index...")
        items_processed = 0
        # Populate the store's data dictionary first
        for item_id, item_data in loaded_data.items():
             # ... (Keep item data validation and numpy conversion as is) ...
            if not all(k in item_data for k in ["vector", "text", "metadata"]):
                 print(f"Warning: Skipping item {item_id} due to missing keys (vector, text, or metadata).")
                 continue
            try:
                vector_list = item_data["vector"]
                vector = np.array(vector_list, dtype=np.float32)

                # Validate dimension consistency
                if store.vector_dim is not None and vector.shape[0] != store.vector_dim:
                     raise ValueError(f"Inconsistent vector dimension for item {item_id}. Expected {store.vector_dim}, found {vector.shape[0]}.")
                elif store.vector_dim is None and store.data: # If dim wasn't set initially, check against first loaded
                     first_vec_dim = next(iter(store.data.values()))["vector"].shape[0]
                     if vector.shape[0] != first_vec_dim: raise ValueError(f"Inconsistent vector dimension for item {item_id}. Expected {first_vec_dim} (inferred), found {vector.shape[0]}.")
                     store.vector_dim = first_vec_dim # Set the inferred dim now
                elif store.vector_dim is None and not store.data: # First item being loaded when dim was None
                     store.vector_dim = vector.shape[0] # Infer and set dim from this item
                     print(f"Inferred vector dimension from loaded file's first item: {store.vector_dim}")

                # Add directly to the internal data dictionary
                store.data[item_id] = {
                    "vector": vector,
                    "text": item_data["text"],
                    "metadata": item_data["metadata"]
                }
                items_processed += 1
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error processing item {item_id} from {filepath}: {e}")

        # Now, rebuild the lexical index from the loaded data
        for item_id, item_data in store.data.items():
            store.lexical_engine.index_document(item_id, item_data['text'])

        print(f"Successfully loaded {len(store.data)} items. Vector dimension: {store.vector_dim or 'Inferred/Any'}. Lexical index rebuilt.")
        return store

    # ----- Utility -----

    def __len__(self) -> int:
        """Returns the number of items in the store."""
        return len(self.data)

    def list_ids(self) -> List[ItemId]:
        """Returns a list of all item IDs."""
        return list(self.data.keys())

# --- Example Usage (Updated) ---
if __name__ == "__main__":
    # Initialize store, optionally specifying vector dimension
    store = SimpleVectorStore(vector_dim=3) # Can also pass stopwords=set([...])

    # Add items
    id1 = store.add_item(np.array([0.1, 0.2, 0.7]), "The quick brown fox jumps over the lazy dog.", {"category": "pets", "year": 2023, "tags": ["feline", "jump"]})
    id2 = store.add_item(np.array([0.8, 0.1, 0.1]), "A second document, this one concerns dogs and foxes.", {"category": "pets", "year": 2022})
    id3 = store.add_item(np.array([0.5, 0.5, 0.0]), "Talking about birds and maybe lazy dogs too.", {"category": "pets", "year": 2023, "rating": 4.5})
    id4 = store.add_item(np.array([0.2, 0.7, 0.1]), "A document unrelated to pets, about python programming and code.", {"category": "tech", "year": 2023, "tags": ["code", "python"]})

    print(f"\nStore contains {len(store)} items.")

    # --- Updates ---
    print("\n--- Updates ---")
    store.update_text(id1, "This first document is updated, still about fluffy cats and quick foxes.") # Re-indexes this doc
    store.update_metadata(id2, {"tags": ["canine", "friendly"]}, replace=False) # No re-index needed
    store.update_vector(id4, np.array([0.1, 0.8, 0.1])) # No re-index needed
    print("Updated item 1 text (re-indexed), item 2 metadata, item 4 vector.")

    # --- Vector Search ---
    print("\n--- Vector Search (Query: close to dogs/item2) ---")
    query_vec = np.array([0.7, 0.2, 0.1])
    vector_results = store.search_vector(query_vec, k=3)
    print(f"Top {len(vector_results)} vector results: {vector_results}")

    # --- Vector Search with Filter ---
    print("\n--- Vector Search (Query: close to dogs/item2, Filter: year=2023) ---")
    vector_results_filtered = store.search_vector(query_vec, k=3, filters={"year": 2023})
    print(f"Top {len(vector_results_filtered)} filtered vector results: {vector_results_filtered}")

    # --- Lexical Search (Now TF-IDF) ---
    print("\n--- Lexical Search (Query: 'lazy dogs') ---")
    lexical_results = store.search_lexical("lazy dogs", k=3)
    print(f"Found {len(lexical_results)} lexical results (TF-IDF): {lexical_results}")
    # Display text for context
    for res_id, score in lexical_results:
        item = store.get_item(res_id)
        print(f"  - {res_id} (Score: {score:.4f}): {item['text'][:60]}...")


    # --- Lexical Search with Filter ---
    print("\n--- Lexical Search (Query: 'document python', Filter: category='tech') ---")
    lexical_results_filtered = store.search_lexical("document python", k=3, filters={"category": "tech"})
    print(f"Found {len(lexical_results_filtered)} filtered lexical results (TF-IDF): {lexical_results_filtered}")
    for res_id, score in lexical_results_filtered:
        item = store.get_item(res_id)
        print(f"  - {res_id} (Score: {score:.4f}): {item['text'][:60]}...")


    # --- Hybrid Search (Now uses normalized TF-IDF) ---
    print("\n--- Hybrid Search (Query Vec: like dogs, Query Text: 'quick fox', weight=0.5) ---")
    hybrid_results = store.search_hybrid(query_vec, "quick fox", k=3, vector_weight=0.5)
    print(f"Top {len(hybrid_results)} hybrid results: {hybrid_results}")
    for res_id, score in hybrid_results:
        item = store.get_item(res_id)
        vec_s = store._cosine_similarity(query_vec, item['vector'])
        lex_s = store.lexical_engine._get_tf_idf_score(store.lexical_engine._preprocess("quick fox"), res_id)
        print(f"  - {res_id} (Combined: {score:.4f}) (Vec ~ {(vec_s+1)/2:.3f}, Lex ~ {lex_s:.3f}): {item['text'][:60]}...")


     # --- Hybrid Search with Filter ---
    print("\n--- Hybrid Search (Query Vec: like dogs, Query Text: 'document', Filter: year=2023, weight=0.8) ---")
    hybrid_results_filtered = store.search_hybrid(query_vec, "document", k=3, filters={"year": 2023}, vector_weight=0.8)
    print(f"Top {len(hybrid_results_filtered)} filtered hybrid results: {hybrid_results_filtered}")
    for res_id, score in hybrid_results_filtered:
        item = store.get_item(res_id)
        print(f"  - {res_id} (Combined: {score:.4f}): {item['text'][:60]}...")


    # --- Deletion ---
    print("\n--- Deletion ---")
    deleted = store.delete_item(id4) # This should now remove from lexical index too
    print(f"Deleted item {id4}: {deleted}")
    print(f"Store now contains {len(store)} items.")

    # Verify lexical search no longer finds deleted item's content easily
    print("\n--- Lexical Search After Deletion (Query: 'python code') ---")
    lexical_after_delete = store.search_lexical("python code", k=3)
    print(f"Found {len(lexical_after_delete)} lexical results: {lexical_after_delete}") # Should be empty or not include id4

    # --- Persistence ---
    print("\n--- Persistence ---")
    SAVE_FILENAME = "my_integrated_vector_store"
    store.save(SAVE_FILENAME)

    # Load from file into a new instance
    try:
        print("\n--- Loading Store ---")
        loaded_store = SimpleVectorStore.load(SAVE_FILENAME)
        print(f"\nLoaded store contains {len(loaded_store)} items.")
        print(f"Vector dimension of loaded store: {loaded_store.vector_dim}")

        # Verify loaded data by fetching an item
        loaded_item1 = loaded_store.get_item(id1)
        if loaded_item1:
            print(f"Successfully retrieved item {id1} from loaded store.")
        else:
            print(f"Error: Failed to retrieve item {id1} from loaded store.")

        # Perform searches on the loaded store to ensure index was rebuilt
        print("\n--- Lexical Search on Loaded Store (Query: 'lazy dogs') ---")
        loaded_lexical_results = loaded_store.search_lexical("lazy dogs", k=3)
        print(f"Found {len(loaded_lexical_results)} lexical results (TF-IDF): {loaded_lexical_results}")

        print("\n--- Hybrid Search on Loaded Store (Query Vec: like dogs, Query Text: 'quick fox', weight=0.5) ---")
        loaded_hybrid_results = loaded_store.search_hybrid(query_vec, "quick fox", k=3, vector_weight=0.5)
        print(f"Top {len(loaded_hybrid_results)} hybrid results: {loaded_hybrid_results}")

        # Clean up the created file (optional)
        try:
            os.remove(SAVE_FILENAME + ".json")
            print(f"\nCleaned up {SAVE_FILENAME}.json")
        except OSError as e:
            print(f"Error removing file {SAVE_FILENAME}.json: {e}")

    except (FileNotFoundError, ValueError, IOError, NameError) as e: # Added NameError for dummy class case
        print(f"\nError during load test: {e}")