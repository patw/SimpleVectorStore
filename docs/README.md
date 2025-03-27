# SimpleVectorStore Documentation

## Overview

`SimpleVectorStore` is a lightweight, in-memory Python library for managing and searching items containing vector embeddings, associated text, and metadata. It's designed for simplicity and ease of use for scenarios where a full-fledged vector database is not required.

**Key Features:**

* **In-Memory Storage:** Holds all data (vectors, text, metadata) in Python dictionaries.
* **Vector Similarity Search:** Performs cosine similarity searches to find items with vectors similar to a query vector.
* **Lexical Search:** Offers basic case-insensitive substring search on the text field.
* **Hybrid Search:** Combines vector similarity and lexical relevance using weighted scoring.
* **Metadata Filtering:** Supports pre-filtering search results based on metadata criteria (equality, range checks, containment).
* **CRUD Operations:** Allows adding, retrieving, updating (vector, text, metadata), and deleting items.
* **Dimension Enforcement:** Optionally enforces a specific dimension for all vectors. Can also infer dimension from the first added item.
* **Persistence:** Saves the entire store to a JSON file and loads it back.

**Limitations:**

* **In-Memory:** Not suitable for datasets larger than available RAM.
* **Basic Search:** Does not implement advanced indexing (like HNSW) for large-scale, ultra-fast vector search. Lexical search is rudimentary.
* **Scalability:** Designed for single-process use; not inherently distributed or thread-safe for concurrent writes without external locking.

## Usage

Since `SimpleVectorStore` is provided as a single Python class, you can simply save the code as a `.py` file (e.g., `simple_vector_store.py`) and import it into your project.

```python
# Assuming the class is saved in simple_vector_store.py
from simple_vector_store import SimpleVectorStore
import numpy as np

# Or, if you have the class definition directly in your script:
# from your_module import SimpleVectorStore
```

## Core Concepts

* **Item:** A single entry in the store, uniquely identified by an `ItemId`.
* **ItemId:** A string identifier for an item (UUID generated if not provided).
* **Vector:** A NumPy array (`np.ndarray`) representing the item in a high-dimensional space (embedding). All vectors in a store should ideally have the same dimension.
* **Text:** A string associated with the item (e.g., the original text that was embedded).
* **Metadata:** A Python dictionary (`Dict[str, Any]`) containing arbitrary key-value pairs associated with the item. Values should be JSON-serializable if persistence is used.
* **Vector Dimension (`vector_dim`):** The number of elements in each vector. The store can enforce this dimension.

## Getting Started: Initialization

You can initialize the store with or without specifying the expected vector dimension.

```python
import numpy as np
from simple_vector_store import SimpleVectorStore # Or your import path

# Initialize with a specific vector dimension (recommended)
# This enforces that all added vectors must have this dimension.
store_fixed_dim = SimpleVectorStore(vector_dim=128)
print(f"Initialized store with fixed dimension: {store_fixed_dim.vector_dim}")

# Initialize without specifying a dimension
# The dimension will be inferred from the first item added.
store_infer_dim = SimpleVectorStore()
print(f"Initialized store with inferred dimension: {store_infer_dim.vector_dim}") # Will be None initially

# Add an item to infer dimension
store_infer_dim.add_item(
vector=np.random.rand(768).astype(np.float32),
text="First item sets the dimension.",
metadata={"source": "init"}
)
print(f"Dimension after adding first item: {store_infer_dim.vector_dim}") # Will now be 768
```

## Core Operations

### Adding Items (`add_item`)

Add a new item with its vector, text, and metadata. An ID is generated if not provided. If an existing ID is used, the item will be overwritten (with a warning).

```python
store = SimpleVectorStore(vector_dim=3)

# Add an item, letting the store generate an ID
item_id_1 = store.add_item(
vector=np.array([0.1, 0.9, 0.0]),
text="Information about apples.",
metadata={"category": "fruit", "color": "red", "year": 2023}
)
print(f"Added item with generated ID: {item_id_1}")

# Add an item with a specific ID
item_id_2 = store.add_item(
vector=np.array([0.8, 0.1, 0.1]),
text="All about oranges.",
metadata={"category": "fruit", "color": "orange", "year": 2022},
item_id="fruit-orange-001"
)
print(f"Added item with specific ID: {item_id_2}")

# Trying to add an item with mismatched dimension (will raise ValueError)
try:
store.add_item(
vector=np.array([0.5, 0.5]), # Only 2 dimensions
text="Error example",
metadata={},
item_id="error-item"
)
except ValueError as e:
print(f"Caught expected error: {e}")

print(f"Store now contains {len(store)} items.")
```

### Getting Items (`get_item`)

Retrieve the full data for an item using its ID. Returns `None` if the ID doesn't exist.

```python
item_data = store.get_item(item_id_1)
if item_data:
print(f"\nRetrieved item {item_id_1}:")
print(f" Text: {item_data['text']}")
print(f" Vector shape: {item_data['vector'].shape}")
print(f" Metadata: {item_data['metadata']}")
else:
print(f"Item {item_id_1} not found.")

non_existent_item = store.get_item("non-existent-id")
print(f"\nRetrieving non-existent item: {non_existent_item}") # Output: None
```

### Updating Items (`update_vector`, `update_text`, `update_metadata`)

Modify specific parts of an existing item. These methods return `True` if the update was successful (item found), `False` otherwise.

```python
# Update vector
updated = store.update_vector(item_id_1, np.array([0.2, 0.8, 0.0]))
print(f"\nUpdated vector for {item_id_1}: {updated}")

# Update text
updated = store.update_text(item_id_1, "More details about red apples.")
print(f"Updated text for {item_id_1}: {updated}")

# Update metadata (merge by default)
updated = store.update_metadata(item_id_2, {"season": "winter", "year": 2024})
print(f"Merged metadata for {item_id_2}: {updated}")
print(f" New metadata: {store.get_item(item_id_2)['metadata']}")

# Update metadata (replace)
updated = store.update_metadata(item_id_2, {"source": "web"}, replace=True)
print(f"Replaced metadata for {item_id_2}: {updated}")
print(f" New metadata after replace: {store.get_item(item_id_2)['metadata']}")

# Try updating a non-existent item
updated = store.update_text("fake-id", "new text")
print(f"Attempted update on fake-id: {updated}") # Output: False
```

### Deleting Items (`delete_item`)

Remove an item from the store by its ID. Returns `True` if deleted, `False` if the ID wasn't found.

```python
# Add a temporary item to delete
temp_id = store.add_item(np.array([0.5, 0.5, 0.0]), "Temporary item", {})
print(f"\nStore size before delete: {len(store)}")

deleted = store.delete_item(temp_id)
print(f"Deleted item {temp_id}: {deleted}")
print(f"Store size after delete: {len(store)}")

# Try deleting again
deleted_again = store.delete_item(temp_id)
print(f"Attempted delete again for {temp_id}: {deleted_again}") # Output: False
```

## Filtering

Filters allow you to narrow down the search space *before* similarity or lexical matching is performed. Filters operate on the `metadata` dictionary of items.

Filters are passed as a dictionary (`FilterType`) to the search methods (`search_vector`, `search_lexical`, `search_hybrid`).

**Supported Filter Syntax:**

* **Equality:** `{ "key": value }` - Matches items where `metadata["key"]` equals `value`.
* **Greater Than:** `{ "key__gt": value }` - Matches where `metadata["key"] > value`.
* **Less Than:** `{ "key__lt": value }` - Matches where `metadata["key"] < value`.
* **Greater Than or Equal:** `{ "key__gte": value }` - Matches where `metadata["key"] >= value`.
* **Less Than or Equal:** `{ "key__lte": value }` - Matches where `metadata["key"] <= value`.
* **In List:** `{ "key__in": [val1, val2, ...] }` - Matches where `metadata["key"]` is one of the values in the list.
* **Contains:** `{ "key__contains": value }` - Matches where `metadata["key"]` (which should be a list, string, or other container) contains `value`.

```python
# Example filter dictionary
filters = {
"category": "fruit", # Must be fruit
"year__gte": 2023, # Year must be 2023 or later
"color__in": ["red", "green"] # Color must be red or green
}

print(f"\nExample filter dictionary: {filters}")
# This filter would match item_id_1 in the previous examples (if color was red/green and year >= 2023)
# but not item_id_2 (year 2022, color orange).
```

## Search Methods

### Vector Search (`search_vector`)

Finds the `k` items whose vectors are most similar to the `query_vector` based on cosine similarity. Optionally applies filters first.

Returns a list of tuples `(ItemId, similarity_score)`, sorted by score (higher is more similar, range -1.0 to 1.0).

```python
store = SimpleVectorStore(vector_dim=3)
id1 = store.add_item(np.array([0.1, 0.8, 0.1]), "About cats", {"topic": "animals", "year": 2023})
id2 = store.add_item(np.array([0.7, 0.2, 0.1]), "About dogs", {"topic": "animals", "year": 2022})
id3 = store.add_item(np.array([0.6, 0.3, 0.1]), "More about dogs", {"topic": "animals", "year": 2023})
id4 = store.add_item(np.array([0.1, 0.1, 0.8]), "About computers", {"topic": "tech", "year": 2023})

query_vector = np.array([0.5, 0.4, 0.1]) # A vector somewhat close to dogs

# Simple vector search
print("\n--- Vector Search (Query: like dogs) ---")
results = store.search_vector(query_vector, k=2)
print(f"Top 2 results: {results}")
# Expected: Likely id3 and id2, depending on exact similarity

# Vector search with filter
print("\n--- Vector Search (Query: like dogs, Filter: year=2023) ---")
filtered_results = store.search_vector(
query_vector,
k=2,
filters={"year": 2023, "topic": "animals"}
)
print(f"Top 2 filtered results (year=2023, topic=animals): {filtered_results}")
# Expected: Only id3 should match the filter among the dog-like items
```

### Lexical Search (`search_lexical`)

Performs a simple, case-insensitive substring search on the `text` field of items. Optionally applies filters.

Returns a list of tuples `(ItemId, score)`. The score is currently fixed at `1.0` for any match and `0.0` otherwise. The list is limited to `k` results, but the order among matches is arbitrary (usually insertion order).

```python
# Lexical search
print("\n--- Lexical Search (Query: 'dogs') ---")
lex_results = store.search_lexical("dogs", k=3)
print(f"Found {len(lex_results)} lexical results for 'dogs': {lex_results}")
# Expected: id2 and id3

# Lexical search with filter
print("\n--- Lexical Search (Query: 'about', Filter: topic='tech') ---")
lex_filtered_results = store.search_lexical(
"about",
k=2,
filters={"topic": "tech"}
)
print(f"Found {len(lex_filtered_results)} filtered lexical results for 'about' (topic=tech): {lex_filtered_results}")
# Expected: Only id4
```

### Hybrid Search (`search_hybrid`)

Combines vector similarity and lexical relevance into a single score. Useful for queries where both semantic meaning (vector) and specific keywords (text) are important.

The final score is a weighted average:
`combined_score = (vector_weight * normalized_vector_score) + ((1 - vector_weight) * lexical_score)`

* `normalized_vector_score`: Cosine similarity mapped from [-1, 1] to [0, 1].
* `lexical_score`: Score from the lexical scorer (defaults to 1.0 for substring match, 0.0 otherwise). Can be customized.
* `vector_weight`: How much importance to give the vector similarity (0.0 to 1.0).

Returns a list of tuples `(ItemId, combined_score)`, sorted by score descending.

```python
# Hybrid search: Query vector like dogs, but text mentions "cats"
query_vector_dogs = np.array([0.6, 0.3, 0.1])
query_text_cats = "cats"

print("\n--- Hybrid Search (Vec: like dogs, Text: 'cats', weight=0.5) ---")
hybrid_results = store.search_hybrid(
query_vector=query_vector_dogs,
query_text=query_text_cats,
k=3,
vector_weight=0.5 # Equal weight to vector and text match
)
print(f"Top 3 hybrid results: {hybrid_results}")
# Expected: id1 (matches text) and id2/id3 (match vector) will likely appear,
# their order depends on the combined score.

# Hybrid search with filter and higher vector weight
print("\n--- Hybrid Search (Vec: like dogs, Text: 'about', Filter: year=2023, weight=0.8) ---")
hybrid_filtered_results = store.search_hybrid(
query_vector=query_vector_dogs,
query_text="about",
k=3,
filters={"year": 2023},
vector_weight=0.8 # More weight on vector similarity
)
print(f"Top 3 filtered hybrid results: {hybrid_filtered_results}")
# Expected: Filters narrow down to id1, id3, id4.
# id3 (dogs, 2023) likely scores highest due to vector similarity + text match.
# id1 (cats, 2023) gets score from text match + some vector score.
# id4 (tech, 2023) gets score from text match + low vector score.
# The exact order depends on weights and similarities.

# Custom Lexical Scorer Example (Optional)
def custom_exact_match_scorer(query_text, item_text):
"""Scores 1.0 only if texts match exactly (case-insensitive)."""
if isinstance(item_text, str):
return 1.0 if query_text.lower() == item_text.lower() else 0.0
return 0.0

print("\n--- Hybrid Search with Custom Scorer ---")
hybrid_custom_results = store.search_hybrid(
query_vector=query_vector_dogs,
query_text="About dogs", # Exact match for id2's text
k=2,
vector_weight=0.3,
lexical_scorer=custom_exact_match_scorer
)
print(f"Top 2 results with exact match scorer: {hybrid_custom_results}")
# id2 should get a high lexical score (1.0 * 0.7) here.
```

## Persistence (`save`, `load`)

Save the current state of the store to a JSON file and load it back later.

**Important:** Metadata values must be JSON-serializable (strings, numbers, lists, dicts, booleans, None). NumPy arrays (vectors) are automatically converted to lists for saving and back to arrays when loading.

### Saving the Store (`save`)

```python
SAVE_FILENAME_BASE = "my_vector_store_data"

print(f"\n--- Saving Store (contains {len(store)} items) ---")
try:
store.save(SAVE_FILENAME_BASE)
# This will create a file named "my_vector_store_data.json"
print(f"Store saved successfully to {SAVE_FILENAME_BASE}.json")
except Exception as e:
print(f"Error saving store: {e}")
```

### Loading the Store (`load`)

`load` is a *class method*. Call it on the `SimpleVectorStore` class itself, not an instance. It returns a *new* instance of the store populated with the loaded data.

```python
print("\n--- Loading Store ---")
try:
# Load the store from the saved file
loaded_store = SimpleVectorStore.load(SAVE_FILENAME_BASE)

print(f"Store loaded successfully from {SAVE_FILENAME_BASE}.json")
print(f"Loaded store contains {len(loaded_store)} items.")
print(f"Loaded store vector dimension: {loaded_store.vector_dim}")

# Verify by getting an item
loaded_item = loaded_store.get_item(id1)
if loaded_item:
print(f"Successfully retrieved item {id1} from loaded store:")
print(f" Text: {loaded_item['text']}")
print(f" Vector type: {type(loaded_item['vector'])}") # Should be

# Perform a search on the loaded store
print("\n--- Vector Search on Loaded Store ---")
loaded_results = loaded_store.search_vector(query_vector, k=2)
print(f"Top 2 results from loaded store: {loaded_results}")

except FileNotFoundError:
print(f"Error: Save file '{SAVE_FILENAME_BASE}.json' not found.")
except (ValueError, IOError, json.JSONDecodeError) as e:
print(f"Error loading store: {e}")

# Optional: Clean up the saved file
# import os
# try:
# os.remove(SAVE_FILENAME_BASE + ".json")
# print(f"\nCleaned up {SAVE_FILENAME_BASE}.json")
# except OSError as e:
# print(f"Error removing file: {e}")
```

## Utility Methods

### Get Store Size (`__len__`)

Use the standard `len()` function to get the number of items in the store.

```python
num_items = len(store) # Or loaded_store
print(f"\nThe store currently has {num_items} items.")
```

### List Item IDs (`list_ids`)

Get a list of all item IDs currently in the store.

```python
all_ids = store.list_ids() # Or loaded_store.list_ids()
print(f"\nAll item IDs in the store: {all_ids}")
```