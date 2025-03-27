# SimpleVectorStore Examples

## Overview

In this directory you'll find the mflix.json database which can be used for testing SimpleVectorStore.  It contains ~6800 summarized movies from the awesome MongDB mflix data set.  The vectors are from Voyage.ai's excellent ```voyage-3``` embedding model, and are 1024 dimensions.  To run the sample code ```mflix_search.py``` you will need to sign up with [Voyage.ai](https://www.voyageai.com/) to get an API key and modify the ```config.json``` file.

### Example Usage

```python
from simple_vector_store import SimpleVectorStore
store = SimpleVectorStore.load("mflix")
print(len(store))
```

