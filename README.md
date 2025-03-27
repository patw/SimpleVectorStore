# SimpleVectorStore

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, in-memory Python library for managing and searching items containing vector embeddings, associated text, and metadata. Designed for simplicity and ease of use in scenarios where a full-featured vector database is overkill.

## Overview

`SimpleVectorStore` provides a straightforward way to store, retrieve, update, and search data points that combine semantic meaning (via vectors) with textual content and structured metadata. It supports vector similarity search (cosine), basic lexical search, hybrid search combining both, and flexible metadata filtering. The entire store can be easily saved to and loaded from a JSON file.

## Features

* **In-Memory:** Fast access as all data resides in RAM.
* **Vector Similarity Search:** Find items with similar vector embeddings using cosine similarity.
* **Lexical Search:** Perform simple case-insensitive substring searches on item text.
* **Hybrid Search:** Combine vector and lexical scores with adjustable weighting.
* **Metadata Filtering:** Pre-filter search candidates based on metadata criteria (equality, ranges (`gt`, `lt`, `gte`, `lte`), list membership (`in`), containment (`contains`)).
* **CRUD Operations:** Add, get, update (vector, text, metadata), and delete items easily.
* **Dimension Enforcement:** Optionally enforce a consistent vector dimension across all items.
* **Persistence:** Save the entire store state to a JSON file and load it back.
* **Simple API:** Designed with ease of use in mind.

## Installation

```
pip install simple-vector-store
```

```python
from simple_vector_store import SimpleVectorStore
import numpy as np
```

**Dependencies:**

* `numpy`: For numerical operations, especially vector handling. (`pip install numpy`)

## Usage

### Initialization

```python
# Initialize with a specific vector dimension (recommended)
store = SimpleVectorStore(vector_dim=3)

# Or initialize without a dimension (it will be inferred from the first item)
# store = SimpleVectorStore()
```

### Adding Items

```python
# Add items with vectors, text, and metadata
id1 = store.add_item(
vector=np.array([0.1, 0.9, 0.0]),
text="Information about apples.",
metadata={"category": "fruit", "color": "red", "year": 2023, "tags": ["juicy", "sweet"]}
)

id2 = store.add_item(
vector=np.array([0.8, 0.1, 0.1]),
text="All about oranges.",
metadata={"category": "fruit", "color": "orange", "year": 2022},
item_id="citrus-001" # You can provide your own IDs
)

id3 = store.add_item(
vector=np.array([0.1, 0.1, 0.8]),
text="Introduction to Python programming.",
metadata={"category": "tech", "language": "python", "year": 2023, "tags": ["code", "beginner"]}
)

print(f"Store now contains {len(store)} items.")
```

### Vector Search

Find items semantically similar to a query vector.

```python
query_vec = np.array([0.2, 0.7, 0.1]) # Query vector somewhat similar to 'apples'

# Find the top 2 most similar items
results = store.search_vector(query_vec, k=2)
print("\nVector Search Results:")
for item_id, score in results:
print(f" ID: {item_id}, Score: {score:.4f}, Text: {store.get_item(item_id)['text']}")
```

### Lexical Search

Find items containing specific keywords in their text.

```python
query_text = "about"

# Find up to 3 items containing the word "about"
results = store.search_lexical(query_text, k=3)
print("\nLexical Search Results:")
for item_id, score in results: # Score is 1.0 for match, 0.0 otherwise
print(f" ID: {item_id}, Score: {score:.1f}, Text: {store.get_item(item_id)['text']}")

```

### Hybrid Search

Combine vector and lexical relevance.

```python
query_vec_fruit = np.array([0.5, 0.5, 0.0]) # Generic fruit vector
query_text_specific = "oranges"

# Find items, weighting vector similarity 60% and text match 40%
results = store.search_hybrid(
query_vector=query_vec_fruit,
query_text=query_text_specific,
k=2,
vector_weight=0.6
)
print("\nHybrid Search Results:")
for item_id, score in results:
print(f" ID: {item_id}, Combined Score: {score:.4f}, Text: {store.get_item(item_id)['text']}")
```

### Filtering

Apply metadata filters *before* searching. Filters are passed as a dictionary to search methods.

```python
query_vec_tech = np.array([0.1, 0.2, 0.7]) # Query related to tech/python

# Vector search for tech items from 2023 or later containing the 'code' tag
filters = {
"category": "tech",
"year__gte": 2023,
"tags__contains": "code"
}

results = store.search_vector(query_vec_tech, k=5, filters=filters)
print("\nFiltered Vector Search Results (tech, >=2023, 'code' tag):")
if results:
for item_id, score in results:
print(f" ID: {item_id}, Score: {score:.4f}, Text: {store.get_item(item_id)['text']}")
else:
print(" No items matched the filters.")

# Lexical search for fruit items where color is 'red' or 'orange'
filters_fruit_color = {
"category": "fruit",
"color__in": ["red", "orange"]
}
lex_results = store.search_lexical("about", k=5, filters=filters_fruit_color)
print("\nFiltered Lexical Search Results (fruit, red/orange):")
if lex_results:
for item_id, score in lex_results:
print(f" ID: {item_id}, Text: {store.get_item(item_id)['text']}")
else:
print(" No items matched the filters.")
```

### Updates & Deletion

```python
# Update text
store.update_text(id1, "Detailed information about crisp red apples.")

# Update metadata (merge by default)
store.update_metadata(id3, {"difficulty": "easy"})

# Update vector
store.update_vector(id3, np.array([0.05, 0.05, 0.9]))

# Delete an item
deleted = store.delete_item(id2)
print(f"\nDeleted item {id2}: {deleted}")
print(f"Store now contains {len(store)} items.")
```

### Persistence

Save the store's state to a JSON file and load it back.

```python
SAVE_FILE = "my_store_backup"

# Save the store
try:
store.save(SAVE_FILE)
print(f"\nStore saved to {SAVE_FILE}.json")
except Exception as e:
print(f"Error saving store: {e}")

# --- Later, or in another script ---

# Load the store (this is a class method, returns a new instance)
try:
loaded_store = SimpleVectorStore.load(SAVE_FILE)
print(f"\nStore loaded from {SAVE_FILE}.json")
print(f"Loaded store has {len(loaded_store)} items.")
print(f"Loaded store vector dimension: {loaded_store.vector_dim}")

# Verify loaded data
item = loaded_store.get_item(id1)
if item:
print(f"Retrieved item {id1} from loaded store: {item['text']}")

except FileNotFoundError:
print(f"Save file {SAVE_FILE}.json not found.")
except Exception as e:
print(f"Error loading store: {e}")

# Optional: Clean up the file
# import os
# os.remove(SAVE_FILE + ".json")
```

## API Reference (Key Methods)

* `__init__(self, vector_dim=None)`: Initialize the store.
* `add_item(self, vector, text, metadata, item_id=None)`: Add/overwrite an item.
* `get_item(self, item_id)`: Retrieve an item's data.
* `delete_item(self, item_id)`: Remove an item.
* `update_vector(self, item_id, vector)`: Update an item's vector.
* `update_text(self, item_id, text)`: Update an item's text.
* `update_metadata(self, item_id, metadata_update, replace=False)`: Update/replace an item's metadata.
* `search_vector(self, query_vector, k=5, filters=None)`: Perform cosine similarity search.
* `search_lexical(self, query_text, k=5, filters=None)`: Perform substring text search.
* `search_hybrid(self, query_vector, query_text, k=5, filters=None, vector_weight=0.7, lexical_scorer=None)`: Perform weighted hybrid search.
* `save(self, filename_base)`: Save store state to `filename_base.json`.
* `load(cls, filename_base)`: Class method to load store state from `filename_base.json`.
* `__len__(self)`: Get the number of items.
* `list_ids(self)`: Get a list of all item IDs.

## Limitations

* **In-Memory Only:** The entire dataset must fit into available RAM. Not suitable for very large datasets.
* **Basic Search Performance:** Vector search involves a linear scan. It will become slow with a very large number of items. No approximate nearest neighbor (ANN) indexing is implemented. Lexical search is ~~also basic~~ kinda awesome now.
* **Scalability:** Primarily designed for single-process use. Concurrent writes are not inherently thread-safe without external locking mechanisms.
* **JSON Serialization:** Metadata must contain JSON-serializable types for persistence to work correctly.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file (if one exists) or the [MIT License text](https://opensource.org/licenses/MIT) for details.