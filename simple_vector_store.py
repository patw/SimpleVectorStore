import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import uuid # For generating IDs if not provided
import json # For saving and loading
import os   # For file path operations

# Define a type alias for clarity
ItemId = str
ItemData = Dict[str, Any]
Metadata = Dict[str, Any]
FilterType = Dict[str, Any]

class SimpleVectorStore:
    """
    A simple in-memory store for vectors, text, and metadata,
    supporting vector similarity, lexical search, filtering, updates,
    and saving/loading to JSON.
    """

    def __init__(self, vector_dim: Optional[int] = None):
        """
        Initializes the store.

        Args:
            vector_dim: Expected dimension for all vectors. If provided,
                        vectors added must match this dimension.
        """
        self.data: Dict[ItemId, ItemData] = {}
        self.vector_dim: Optional[int] = vector_dim
        # Don't print during basic init, print happens after loading or first add
        # print(f"Initialized SimpleVectorStore (Expected vector dim: {vector_dim or 'Any'})")

    # ----- Core Data Management -----

    def add_item(self,
                 vector: np.ndarray,
                 text: str,
                 metadata: Metadata,
                 item_id: Optional[ItemId] = None) -> ItemId:
        """Adds or overwrites an item in the store."""
        if item_id is None:
            item_id = str(uuid.uuid4())
        elif item_id in self.data:
            print(f"Warning: Overwriting existing item with ID: {item_id}")

        # Validate vector dimension if specified
        current_vector_dim = vector.shape[0] if isinstance(vector, np.ndarray) else None
        if current_vector_dim is None:
             raise ValueError("Input vector must be a numpy array.")

        if self.vector_dim is not None and current_vector_dim != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {current_vector_dim}")
        elif self.vector_dim is None and not self.data: # First item sets the dim if not preset
             self.vector_dim = current_vector_dim
             print(f"Inferred vector dimension from first item: {self.vector_dim}")
        elif self.vector_dim is None and self.data: # Check against inferred dim
             # This case should ideally not happen if the first item set it,
             # but adding a check for robustness if store was somehow manipulated
             first_item_vec = next(iter(self.data.values()))["vector"]
             inferred_dim = first_item_vec.shape[0]
             if current_vector_dim != inferred_dim:
                 raise ValueError(f"Vector dimension mismatch. Expected {inferred_dim} (inferred), got {current_vector_dim}")
             self.vector_dim = inferred_dim # Solidify inferred dim
        elif self.vector_dim is not None and current_vector_dim != self.vector_dim:
             # This check is redundant with the first one but kept for clarity
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {current_vector_dim}")


        self.data[item_id] = {
            "vector": vector.astype(np.float32), # Store as consistent type
            "text": text,
            "metadata": metadata.copy() # Store a copy
        }
        return item_id

    def get_item(self, item_id: ItemId) -> Optional[ItemData]:
        """Retrieves an item by its ID."""
        return self.data.get(item_id)

    def delete_item(self, item_id: ItemId) -> bool:
        """Deletes an item by its ID. Returns True if deleted, False otherwise."""
        if item_id in self.data:
            del self.data[item_id]
            return True
        return False

    def update_vector(self, item_id: ItemId, vector: np.ndarray) -> bool:
        """Updates the vector for a specific item."""
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
            return True
        return False

    def update_metadata(self, item_id: ItemId, metadata_update: Metadata, replace: bool = False) -> bool:
        """Updates the metadata for a specific item. Merges by default."""
        if item_id in self.data:
            if replace:
                self.data[item_id]["metadata"] = metadata_update.copy()
            else:
                self.data[item_id]["metadata"].update(metadata_update)
            return True
        return False

    # ----- Filtering -----

    def _matches_filters(self, item_metadata: Metadata, filters: FilterType) -> bool:
        """Checks if item metadata matches the provided filters."""
        if not filters:
            return True
        for key, value in filters.items():
            # Handle special filter conditions first
            try:
                if key.endswith('__gt'):
                    actual_key = key[:-4]
                    if not (actual_key in item_metadata and isinstance(item_metadata[actual_key], (int, float)) and item_metadata[actual_key] > value):
                        return False
                elif key.endswith('__lt'):
                    actual_key = key[:-4]
                    if not (actual_key in item_metadata and isinstance(item_metadata[actual_key], (int, float)) and item_metadata[actual_key] < value):
                        return False
                elif key.endswith('__gte'):
                    actual_key = key[:-5]
                    if not (actual_key in item_metadata and isinstance(item_metadata[actual_key], (int, float)) and item_metadata[actual_key] >= value):
                        return False
                elif key.endswith('__lte'):
                    actual_key = key[:-5]
                    if not (actual_key in item_metadata and isinstance(item_metadata[actual_key], (int, float)) and item_metadata[actual_key] <= value):
                        return False
                elif key.endswith('__in'):
                     actual_key = key[:-4]
                     if not (actual_key in item_metadata and isinstance(value, (list, tuple, set)) and item_metadata[actual_key] in value):
                         return False
                elif key.endswith('__contains'): # Check if metadata value (list/str) contains filter value
                     actual_key = key[:-10]
                     if not (actual_key in item_metadata and hasattr(item_metadata[actual_key], '__contains__') and value in item_metadata[actual_key]):
                         return False
                # Default: Basic equality check
                else:
                    if key not in item_metadata or item_metadata[key] != value:
                       return False
            except (TypeError, KeyError): # Handle cases where comparisons or key access fail gracefully
                 return False
        return True


    def _get_filtered_ids(self, filters: Optional[FilterType]) -> List[ItemId]:
        """Returns a list of item IDs that match the filters."""
        if not filters:
            return list(self.data.keys())
        return [
            item_id for item_id, item_data in self.data.items()
            if self._matches_filters(item_data["metadata"], filters)
        ]

    # ----- Search Methods -----

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        # Ensure vectors are float32 for consistency
        vec1 = vec1.astype(np.float32)
        vec2 = vec2.astype(np.float32)

        if vec1.shape != vec2.shape:
             raise ValueError(f"Cannot compute similarity for vectors with shapes {vec1.shape} and {vec2.shape}")
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # Use np.dot for potentially better performance and clarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        # Clip to handle potential floating point inaccuracies slightly outside [-1, 1]
        return float(np.clip(similarity, -1.0, 1.0))


    def search_vector(self,
                      query_vector: np.ndarray,
                      k: int = 5,
                      filters: Optional[FilterType] = None) -> List[Tuple[ItemId, float]]:
        """
        Performs vector similarity search (cosine similarity).

        Args:
            query_vector: The vector to search for.
            k: Number of nearest neighbors to return.
            filters: Optional metadata filters to apply before searching.

        Returns:
            A list of tuples: (item_id, similarity_score), sorted by score descending.
        """
        query_vector = query_vector.astype(np.float32) # Ensure query is float32
        query_dim = query_vector.shape[0]
        if self.vector_dim is not None and query_dim != self.vector_dim:
             raise ValueError(f"Query vector dimension mismatch. Expected {self.vector_dim}, got {query_dim}")
        elif self.vector_dim is None and self.data:
             # Infer expected dim from existing data if not set
             first_item_vec = next(iter(self.data.values()))["vector"]
             inferred_dim = first_item_vec.shape[0]
             if query_dim != inferred_dim:
                  raise ValueError(f"Query vector dimension mismatch. Expected {inferred_dim} (inferred), got {query_dim}")
             # Optionally set self.vector_dim here if desired, or leave as None
             # self.vector_dim = inferred_dim
        elif self.vector_dim is None and not self.data:
             # Cannot search an empty store, and cannot verify dimension
             return []


        candidate_ids = self._get_filtered_ids(filters)
        if not candidate_ids:
            return []

        results = []
        for item_id in candidate_ids:
            item_vector = self.data[item_id]["vector"]
            similarity = self._cosine_similarity(query_vector, item_vector)
            results.append((item_id, similarity))

        # Sort by similarity score (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def search_lexical(self,
                       query_text: str,
                       k: int = 5,
                       filters: Optional[FilterType] = None) -> List[Tuple[ItemId, float]]:
        """
        Performs simple case-insensitive KEYWORD lexical search.
        Checks if ANY word from the query exists in the item's text.

        Args:
            query_text: The text string to search for keywords from.
            k: Max number of results to return (ranking is arbitrary here).
            filters: Optional metadata filters to apply before searching.

        Returns:
            A list of tuples: (item_id, score) - score is 1.0 for match, 0.0 otherwise.
            Limited by k, but not meaningfully ranked beyond matching/not matching.
        """
        candidate_ids = self._get_filtered_ids(filters)
        if not candidate_ids:
            return []

        # Split query into words and lowercase them.
        # Basic split, ignores punctuation attached to words for simplicity.
        # Consider using regex or a library like nltk for better tokenization if needed.
        query_words = set(word for word in query_text.lower().split() if word) # Use a set for faster lookup

        if not query_words: # Handle empty query
             return []

        results = []
        for item_id in candidate_ids:
            item_text = self.data[item_id].get("text", "")
            if isinstance(item_text, str):
                item_text_lower = item_text.lower()
                # Check if any query word is present in the item text
                # This is still basic substring checking for each word
                found_match = False
                for word in query_words:
                    if word in item_text_lower:
                        found_match = True
                        break # Found one word, no need to check others for a simple 1.0 score

                if found_match:
                    results.append((item_id, 1.0)) # Simple 1.0 score on any match
            else:
                 print(f"Warning: Item {item_id} has non-string or missing text field.")

        # No meaningful ranking here, just return up to k matches
        return results[:k]

    def search_hybrid(self,
                      query_vector: np.ndarray,
                      query_text: str,
                      k: int = 5,
                      filters: Optional[FilterType] = None,
                      vector_weight: float = 0.7,
                      lexical_scorer: Optional[Callable[[str, str], float]] = None
                      ) -> List[Tuple[ItemId, float]]:
        """
        Performs a hybrid search combining vector similarity and lexical relevance.

        Args:
            query_vector: The vector part of the query.
            query_text: The text part of the query.
            k: Number of results to return.
            filters: Optional metadata filters.
            vector_weight: Weight given to vector similarity score (0.0 to 1.0).
                           Lexical score weight will be (1.0 - vector_weight).
            lexical_scorer: Optional function (query_text, item_text) -> score (0-1).
                            Defaults to basic substring match score (1 if match, 0 else).

        Returns:
            A list of tuples: (item_id, combined_score), sorted by score descending.
        """
        if not (0.0 <= vector_weight <= 1.0):
            raise ValueError("vector_weight must be between 0.0 and 1.0")

        query_vector = query_vector.astype(np.float32) # Ensure query is float32
        query_dim = query_vector.shape[0]
        if self.vector_dim is not None and query_dim != self.vector_dim:
             raise ValueError(f"Query vector dimension mismatch. Expected {self.vector_dim}, got {query_dim}")
        elif self.vector_dim is None and self.data:
             first_item_vec = next(iter(self.data.values()))["vector"]
             inferred_dim = first_item_vec.shape[0]
             if query_dim != inferred_dim:
                  raise ValueError(f"Query vector dimension mismatch. Expected {inferred_dim} (inferred), got {query_dim}")
        elif self.vector_dim is None and not self.data:
             return [] # Cannot search empty store


        candidate_ids = self._get_filtered_ids(filters)
        if not candidate_ids:
            return []

        # Default lexical scorer: simple substring match
        if lexical_scorer is None:
            query_lower = query_text.lower()
            def default_scorer(q_text, i_text):
                 # Handle non-string item text gracefully
                 if isinstance(i_text, str):
                    return 1.0 if query_lower in i_text.lower() else 0.0
                 return 0.0
            lexical_scorer = default_scorer

        results = []
        lexical_weight = 1.0 - vector_weight

        for item_id in candidate_ids:
            item_data = self.data[item_id]
            vector_score = self._cosine_similarity(query_vector, item_data["vector"])
            lexical_score = lexical_scorer(query_text, item_data.get("text", "")) # Use .get for safety

            # Normalize vector score (cosine is -1 to 1, map to 0 to 1)
            normalized_vector_score = (vector_score + 1.0) / 2.0

            combined_score = (vector_weight * normalized_vector_score) + (lexical_weight * lexical_score)
            results.append((item_id, combined_score))

        # Sort by combined score (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    # ----- Persistence -----

    def save(self, filename_base: str):
        """
        Saves the current state of the vector store to a JSON file.

        Args:
            filename_base: The base name for the file (e.g., "my_store").
                           ".json" will be appended automatically.
        """
        filepath = filename_base + ".json"
        print(f"Saving vector store to {filepath}...")

        # Prepare data for JSON serialization
        serializable_data = {}
        for item_id, item_data in self.data.items():
            serializable_data[item_id] = {
                # Convert numpy array to list
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
                json.dump(save_package, f, indent=4) # Use indent for readability
            print(f"Successfully saved {len(self.data)} items.")
        except IOError as e:
            print(f"Error saving store to {filepath}: {e}")
        except TypeError as e:
             print(f"Error serializing data for saving: {e}. Ensure metadata contains JSON-compatible types.")


    @classmethod
    def load(cls, filename_base: str) -> 'SimpleVectorStore':
        """
        Loads a vector store from a JSON file.

        Args:
            filename_base: The base name of the file to load (e.g., "my_store").
                           ".json" will be appended automatically.

        Returns:
            A new SimpleVectorStore instance populated with the loaded data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file format is invalid or data is inconsistent.
        """
        filepath = filename_base + ".json"
        print(f"Loading vector store from {filepath}...")

        if not os.path.exists(filepath):
             raise FileNotFoundError(f"Save file not found: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                load_package = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {filepath}: {e}")
        except IOError as e:
            raise IOError(f"Error reading file {filepath}: {e}")


        # Validate basic structure
        if "vector_dim" not in load_package or "data" not in load_package:
            raise ValueError(f"Invalid file format in {filepath}. Missing 'vector_dim' or 'data' key.")

        loaded_vector_dim = load_package["vector_dim"]
        loaded_data = load_package["data"]

        # Create a new store instance with the loaded dimension
        store = cls(vector_dim=loaded_vector_dim)

        # Populate the store
        for item_id, item_data in loaded_data.items():
            if not all(k in item_data for k in ["vector", "text", "metadata"]):
                 print(f"Warning: Skipping item {item_id} due to missing keys (vector, text, or metadata).")
                 continue

            try:
                # Convert vector list back to numpy array
                vector_list = item_data["vector"]
                vector = np.array(vector_list, dtype=np.float32)

                # Validate dimension consistency within the loaded data
                if loaded_vector_dim is not None and vector.shape[0] != loaded_vector_dim:
                     raise ValueError(
                         f"Inconsistent vector dimension for item {item_id} in {filepath}. "
                         f"Expected {loaded_vector_dim}, found {vector.shape[0]}."
                     )
                elif loaded_vector_dim is None and store.data: # If dim wasn't set initially, check against first loaded
                     first_vec_dim = next(iter(store.data.values()))["vector"].shape[0]
                     if vector.shape[0] != first_vec_dim:
                          raise ValueError(
                              f"Inconsistent vector dimension for item {item_id} in {filepath}. "
                              f"Expected {first_vec_dim} (inferred from first loaded item), found {vector.shape[0]}."
                          )
                     store.vector_dim = first_vec_dim # Set the inferred dim now
                elif loaded_vector_dim is None and not store.data: # First item being loaded when dim was None
                     store.vector_dim = vector.shape[0] # Infer and set dim from this item
                     print(f"Inferred vector dimension from loaded file's first item: {store.vector_dim}")


                # Add directly to the internal data dictionary
                # (Bypasses add_item checks which are partially redundant here)
                store.data[item_id] = {
                    "vector": vector,
                    "text": item_data["text"],
                    "metadata": item_data["metadata"]
                }
            except (ValueError, TypeError) as e:
                # Catch errors during numpy array conversion or validation
                raise ValueError(f"Error processing item {item_id} from {filepath}: {e}")


        print(f"Successfully loaded {len(store.data)} items. Vector dimension: {store.vector_dim or 'Inferred/Any'}")
        return store


    # ----- Utility -----

    def __len__(self) -> int:
        """Returns the number of items in the store."""
        return len(self.data)

    def list_ids(self) -> List[ItemId]:
        """Returns a list of all item IDs."""
        return list(self.data.keys())

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize store, optionally specifying vector dimension
    store = SimpleVectorStore(vector_dim=3)

    # Add items
    id1 = store.add_item(np.array([0.1, 0.2, 0.7]), "This is the first document about cats.", {"category": "pets", "year": 2023, "tags": ["feline"]})
    id2 = store.add_item(np.array([0.8, 0.1, 0.1]), "A second document, this one concerns dogs.", {"category": "pets", "year": 2022})
    id3 = store.add_item(np.array([0.5, 0.5, 0.0]), "Talking about birds and maybe dogs too.", {"category": "pets", "year": 2023, "rating": 4.5})
    id4 = store.add_item(np.array([0.2, 0.7, 0.1]), "A document unrelated to pets, about programming.", {"category": "tech", "year": 2023, "tags": ["code", "python"]})

    print(f"\nStore contains {len(store)} items.")
    # print(f"Item IDs: {store.list_ids()}") # ID listing can be long

    # --- Updates ---
    print("\n--- Updates ---")
    store.update_text(id1, "This is the first document, updated to be about fluffy cats.")
    store.update_metadata(id2, {"tags": ["canine", "friendly"]}, replace=False) # Merge
    store.update_vector(id4, np.array([0.1, 0.8, 0.1]))
    print("Updated item 1 text, item 2 metadata, item 4 vector.")
    # print(f"Item 1 data: {store.get_item(id1)}")
    # print(f"Item 2 data: {store.get_item(id2)}")

    # --- Vector Search ---
    print("\n--- Vector Search (Query: close to dogs/item2) ---")
    query_vec = np.array([0.7, 0.2, 0.1])
    vector_results = store.search_vector(query_vec, k=3)
    print(f"Top {len(vector_results)} vector results: {vector_results}")

    # --- Vector Search with Filter ---
    print("\n--- Vector Search (Query: close to dogs/item2, Filter: year=2023) ---")
    vector_results_filtered = store.search_vector(query_vec, k=3, filters={"year": 2023})
    print(f"Top {len(vector_results_filtered)} filtered vector results: {vector_results_filtered}")

    # --- More Complex Filter ---
    print("\n--- Vector Search (Query: close to dogs/item2, Filter: year>=2023, category='pets') ---")
    vector_results_filtered_adv = store.search_vector(query_vec, k=3, filters={"year__gte": 2023, "category": "pets"})
    print(f"Top {len(vector_results_filtered_adv)} filtered vector results: {vector_results_filtered_adv}")

    print("\n--- Lexical Search (Query: 'document', Filter: tags contains 'python') ---")
    lexical_results_filtered_adv = store.search_lexical("document", k=3, filters={"tags__contains": "python"})
    print(f"Found {len(lexical_results_filtered_adv)} filtered lexical results: {lexical_results_filtered_adv}")


    # --- Lexical Search ---
    print("\n--- Lexical Search (Query: 'dogs') ---")
    lexical_results = store.search_lexical("dogs", k=3)
    print(f"Found {len(lexical_results)} lexical results: {lexical_results}")

    # --- Lexical Search with Filter ---
    print("\n--- Lexical Search (Query: 'document', Filter: category='pets') ---")
    lexical_results_filtered = store.search_lexical("document", k=3, filters={"category": "pets"})
    print(f"Found {len(lexical_results_filtered)} filtered lexical results: {lexical_results_filtered}")

    # --- Hybrid Search ---
    print("\n--- Hybrid Search (Query Vec: like dogs, Query Text: 'cats', weight=0.5) ---")
    hybrid_results = store.search_hybrid(query_vec, "cats", k=3, vector_weight=0.5)
    print(f"Top {len(hybrid_results)} hybrid results: {hybrid_results}")

     # --- Hybrid Search with Filter ---
    print("\n--- Hybrid Search (Query Vec: like dogs, Query Text: 'document', Filter: year=2023, weight=0.8) ---")
    hybrid_results_filtered = store.search_hybrid(query_vec, "document", k=3, filters={"year": 2023}, vector_weight=0.8)
    print(f"Top {len(hybrid_results_filtered)} filtered hybrid results: {hybrid_results_filtered}")

    # --- Deletion ---
    print("\n--- Deletion ---")
    deleted = store.delete_item(id4)
    print(f"Deleted item {id4}: {deleted}")
    print(f"Store now contains {len(store)} items.")
    # print(f"Item 4 data: {store.get_item(id4)}") # Should be None

    # --- Persistence ---
    print("\n--- Persistence ---")
    SAVE_FILENAME = "my_simple_vector_store"
    store.save(SAVE_FILENAME)

    # Clear the store (or just create a new variable) to test loading
    # store.data = {} # Optional: Clear in-memory data
    # store.vector_dim = None # Optional: Clear dimension
    # print("\nCleared in-memory store.")

    # Load from file into a new instance
    try:
        loaded_store = SimpleVectorStore.load(SAVE_FILENAME)
        print(f"\nLoaded store contains {len(loaded_store)} items.")
        print(f"Vector dimension of loaded store: {loaded_store.vector_dim}")

        # Verify loaded data by fetching an item
        loaded_item1 = loaded_store.get_item(id1)
        if loaded_item1:
            print(f"Successfully retrieved item {id1} from loaded store.")
            # print(f"Loaded Item 1 Text: {loaded_item1['text']}")
            # print(f"Loaded Item 1 Vector (type): {type(loaded_item1['vector'])}") # Should be numpy.ndarray
            # print(f"Loaded Item 1 Vector Dim: {loaded_item1['vector'].shape}")
        else:
            print(f"Error: Failed to retrieve item {id1} from loaded store.")

        # Perform a search on the loaded store
        print("\n--- Vector Search on Loaded Store (Query: close to dogs/item2) ---")
        loaded_vector_results = loaded_store.search_vector(query_vec, k=3)
        print(f"Top {len(loaded_vector_results)} vector results: {loaded_vector_results}")

        # Clean up the created file (optional)
        # try:
        #     os.remove(SAVE_FILENAME + ".json")
        #     print(f"\nCleaned up {SAVE_FILENAME}.json")
        # except OSError as e:
        #     print(f"Error removing file {SAVE_FILENAME}.json: {e}")

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"\nError during load test: {e}")