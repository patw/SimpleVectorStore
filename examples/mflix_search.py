from simple_vector_store import SimpleVectorStore
import voyageai # The best embedding models ever!
import json
import numpy as np

# Load the config file
with open("config.json", 'r') as file:
    config = json.load(file)

# Vectors. Soon...
vo = voyageai.Client(api_key=config["voyage_api_key"])

# Load up the search engine
store = SimpleVectorStore.load("mflix")

# Ask a question...
QUESTION = "Movie about ghost catching in new york with Bill Murray"
qestion_embedding = vo.embed([QUESTION], model="voyage-3").embeddings[0]

print("Vector only...")
vector_results = store.search_vector(query_vector=np.array(qestion_embedding), k=3)
for result in vector_results:
    id = result[0]
    score = result[1]
    item = store.get_item(id)
    print(str(score) + " - " +  item["metadata"]["title"])

print("Lexical only...")
lex_results = store.search_lexical(query_text=QUESTION, k=3)
for result in lex_results:
    id = result[0]
    score = result[1]
    item = store.get_item(id)
    print(str(score) + " - " +  item["metadata"]["title"])

print("Hybrid...")
hybrid_results = store.search_hybrid(query_vector=np.array(qestion_embedding), query_text=QUESTION, k=5, vector_weight=0.9)
for result in hybrid_results:
    id = result[0]
    score = result[1]
    item = store.get_item(id)
    print(str(score) + " - " +  item["metadata"]["title"])