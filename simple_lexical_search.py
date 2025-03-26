import re
import math
from collections import Counter, defaultdict

# -----------------------------------------------------------------------------
# Simple Porter Stemmer (Pure Python Implementation)
# Based on the original algorithm by Martin Porter
# This is a compact version; many variations exist.
# -----------------------------------------------------------------------------
class PorterStemmer:
    def __init__(self):
        self.vowels = "aeiou"

    def _is_consonant(self, word, i):
        if word[i] in self.vowels:
            return False
        if word[i] == 'y':
            if i == 0:
                return True
            else:
                return not self._is_consonant(word, i - 1)
        return True

    def _measure(self, word):
        """Measure the number of consonant sequences in the word"""
        m = 0
        n = len(word)
        i = 0
        while i < n:
            while i < n and not self._is_consonant(word, i):
                i += 1
            if i >= n:
                break
            m += 1
            while i < n and self._is_consonant(word, i):
                i += 1
        return m

    def _contains_vowel(self, stem):
        for i in range(len(stem)):
            if not self._is_consonant(stem, i):
                return True
        return False

    def _ends_double_consonant(self, word):
        if len(word) >= 2:
            return self._is_consonant(word, len(word) - 1) and \
                   self._is_consonant(word, len(word) - 2)
        return False

    def _ends_cvc(self, word):
        n = len(word)
        if n < 3:
            return False
        return (self._is_consonant(word, n - 3) and
                not self._is_consonant(word, n - 2) and
                self._is_consonant(word, n - 1) and
                word[n - 1] not in 'wxy')

    def _step1a(self, word):
        if word.endswith('sses'):
            return word[:-2]
        if word.endswith('ies'):
            return word[:-2]
        if word.endswith('ss'):
            return word # No change
        if word.endswith('s'):
            # Check if the stem contains a vowel before the 's'
            has_vowel_before_s = False
            for char in word[:-2]: # Check the part before the potential single 's'
                if char in self.vowels:
                    has_vowel_before_s = True
                    break
            # Heuristic: only remove 's' if it's likely plural/verb form, not part of the word itself
            # This simple check helps avoid over-stemming short words or words ending in 's' naturally.
            # More complex rules exist, but this keeps it simpler.
            if len(word) > 2 and has_vowel_before_s:
                 return word[:-1]
        return word

    def _step1b(self, word):
        flag = False
        if word.endswith('eed'):
            if self._measure(word[:-3]) > 0:
                return word[:-1]
            else:
                return word
        elif word.endswith('ed'):
            stem = word[:-2]
            if self._contains_vowel(stem):
                word = stem
                flag = True
        elif word.endswith('ing'):
            stem = word[:-3]
            if self._contains_vowel(stem):
                word = stem
                flag = True

        if flag:
            if word.endswith(('at', 'bl', 'iz')):
                return word + 'e'
            elif self._ends_double_consonant(word) and word[-1] not in 'lsz':
                return word[:-1]
            elif self._measure(word) == 1 and self._ends_cvc(word):
                return word + 'e'
        return word

    def _step1c(self, word):
        if word.endswith('y') and self._contains_vowel(word[:-1]):
             return word[:-1] + 'i'
        return word

    def _step2(self, word):
        m = self._measure(word)
        if m > 0:
            if word.endswith('ational'): return word[:-7] + 'ate'
            if word.endswith('tional'): return word[:-6] + 'tion'
            if word.endswith('enci'): return word[:-4] + 'ence'
            if word.endswith('anci'): return word[:-4] + 'ance'
            if word.endswith('izer'): return word[:-4] + 'ize'
            if word.endswith('abli'): return word[:-4] + 'able' # Should be bli
            if word.endswith('alli'): return word[:-4] + 'al'
            if word.endswith('entli'): return word[:-5] + 'ent'
            if word.endswith('eli'): return word[:-3] + 'e'
            if word.endswith('ousli'): return word[:-5] + 'ous'
            if word.endswith('ization'): return word[:-7] + 'ize'
            if word.endswith('ation'): return word[:-5] + 'ate'
            if word.endswith('ator'): return word[:-4] + 'ate'
            if word.endswith('alism'): return word[:-5] + 'al'
            if word.endswith('iveness'): return word[:-7] + 'ive'
            if word.endswith('fulness'): return word[:-7] + 'ful'
            if word.endswith('ousness'): return word[:-7] + 'ous'
            if word.endswith('aliti'): return word[:-5] + 'al'
            if word.endswith('iviti'): return word[:-5] + 'ive'
            if word.endswith('biliti'): return word[:-6] + 'ble'
            if word.endswith('logi'): return word[:-4] + 'log' # Added for common case
        return word

    def _step3(self, word):
        m = self._measure(word)
        if m > 0:
            if word.endswith('icate'): return word[:-5] + 'ic'
            if word.endswith('ative'): return word[:-5]
            if word.endswith('alize'): return word[:-5] + 'al'
            if word.endswith('iciti'): return word[:-5] + 'ic'
            if word.endswith('ical'): return word[:-4] + 'ic'
            if word.endswith('ful'): return word[:-3]
            if word.endswith('ness'): return word[:-4]
        return word

    def _step4(self, word):
        m = self._measure(word)
        if m > 1: # Only apply if m > 1
            if word.endswith('al'): return word[:-2]
            if word.endswith('ance'): return word[:-4]
            if word.endswith('ence'): return word[:-4]
            if word.endswith('er'): return word[:-2]
            if word.endswith('ic'): return word[:-2]
            if word.endswith('able'): return word[:-4]
            if word.endswith('ible'): return word[:-4]
            if word.endswith('ant'): return word[:-3]
            if word.endswith('ement'): return word[:-5]
            if word.endswith('ment'): return word[:-4]
            if word.endswith('ent'): return word[:-3]
            if word.endswith(('ion')) and len(word) > 3 and word[-4] in ('s', 't'):
                 return word[:-3] # Condition: stem ends in s or t
            if word.endswith('ou'): return word[:-2]
            if word.endswith('ism'): return word[:-3]
            if word.endswith('ate'): return word[:-3]
            if word.endswith('iti'): return word[:-3]
            if word.endswith('ous'): return word[:-3]
            if word.endswith('ive'): return word[:-3]
            if word.endswith('ize'): return word[:-3]
        return word

    def _step5a(self, word):
        m = self._measure(word)
        if word.endswith('e'):
            if m > 1:
                return word[:-1]
            elif m == 1 and not self._ends_cvc(word[:-1]):
                return word[:-1]
        return word

    def _step5b(self, word):
        m = self._measure(word)
        if m > 1 and self._ends_double_consonant(word) and word.endswith('l'):
             return word[:-1]
        return word

    def stem(self, word):
        """Stems a word using the Porter algorithm."""
        if len(word) <= 2: # Short words are not stemmed
            return word

        word = word.lower() # Ensure lowercase

        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5a(word)
        word = self._step5b(word)
        return word

# -----------------------------------------------------------------------------
# Simple Search Engine Class
# -----------------------------------------------------------------------------
class SimpleSearchEngine:
    def __init__(self, stopwords=None):
        # ... (Keep __init__ as is) ...
        self.documents = {} # {doc_id: original_text}
        self.doc_term_freqs = {} # {doc_id: Counter(term: freq)}
        self.inverted_index = defaultdict(set) # {term: {doc_id1, doc_id2, ...}}
        self.stemmer = PorterStemmer()

        if stopwords is None:
            # A basic set of English stopwords
            self.stopwords = set([
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
                'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of',
                'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there',
                'these', 'they', 'this', 'to', 'was', 'will', 'with', 'i',
                'you', 'your', 'me', 'my', 'he', 'she', 'him', 'her', 'we',
                'our', 'us'
            ])
        else:
            self.stopwords = set(stopwords)

    def _tokenize(self, text):
        # ... (Keep _tokenize as is) ...
        return re.findall(r'\b\w+\b', text.lower())


    def _preprocess(self, text):
        # ... (Keep _preprocess as is) ...
        tokens = self._tokenize(text)
        stopped_tokens = [token for token in tokens if token not in self.stopwords]
        stemmed_tokens = [self.stemmer.stem(token) for token in stopped_tokens]
        return stemmed_tokens

    def index_document(self, doc_id, text):
        # ... (Keep index_document as is, maybe remove the print warning or make it optional) ...
        if doc_id in self.documents:
            # If overwriting, first remove the old index entries
            self.remove_document(doc_id) # Call remove before re-indexing

        self.documents[doc_id] = text
        processed_terms = self._preprocess(text)

        if not processed_terms:
            self.doc_term_freqs[doc_id] = Counter() # Store empty counter
            return # No terms to index

        term_counts = Counter(processed_terms)
        self.doc_term_freqs[doc_id] = term_counts

        for term in term_counts:
            self.inverted_index[term].add(doc_id)

    def remove_document(self, doc_id):
        """Removes a document and its terms from the index."""
        if doc_id not in self.documents:
            return False # Document doesn't exist

        # Get the terms associated with the document *before* deleting frequency info
        term_counts = self.doc_term_freqs.get(doc_id, Counter())

        # Remove from primary storage
        del self.documents[doc_id]
        if doc_id in self.doc_term_freqs:
            del self.doc_term_freqs[doc_id]

        # Remove doc_id from inverted index for each term it contained
        terms_to_check = list(term_counts.keys()) # Iterate over a copy of keys
        for term in terms_to_check:
            if term in self.inverted_index:
                if doc_id in self.inverted_index[term]:
                    self.inverted_index[term].remove(doc_id)
                # Optional: Clean up terms that no longer point to any documents
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
        return True


    def index_documents(self, documents_dict):
        # ... (Keep index_documents as is) ...
        for doc_id, text in documents_dict.items():
            self.index_document(doc_id, text)


    def _calculate_tf(self, term, doc_id):
        # ... (Keep _calculate_tf as is) ...
        term_counts = self.doc_term_freqs.get(doc_id, Counter())
        total_terms_in_doc = sum(term_counts.values())
        if total_terms_in_doc == 0:
            return 0
        return term_counts.get(term, 0) / total_terms_in_doc

    def _calculate_idf(self, term):
        # ... (Keep _calculate_idf as is) ...
        num_documents = len(self.documents)
        if num_documents == 0:
            return 0
        docs_containing_term = len(self.inverted_index.get(term, set()))
        return math.log(num_documents / (1 + docs_containing_term))

    def _get_tf_idf_score(self, query_terms, doc_id):
        """Calculates the TF-IDF score for a specific document given processed query terms."""
        score = 0.0
        unique_query_terms = set(query_terms) # Avoid double counting IDF for repeated query terms

        for term in unique_query_terms:
            tf = self._calculate_tf(term, doc_id)
            if tf > 0: # Only calculate IDF if term is actually in the doc (TF > 0)
                 idf = self._calculate_idf(term)
                 score += tf * idf
        return score

    def search(self, query, top_n=10):
        # ... (Modify slightly to use the _get_tf_idf_score helper) ...
        if not self.documents:
            return []

        processed_query_terms = self._preprocess(query)
        if not processed_query_terms:
            return []

        doc_scores = defaultdict(float)
        relevant_doc_ids = set()

        # Find all potentially relevant documents first
        for term in set(processed_query_terms): # Use unique terms
            if term in self.inverted_index:
                relevant_doc_ids.update(self.inverted_index[term])

        if not relevant_doc_ids:
            return [] # No documents contain any of the query terms

        # Calculate scores only for relevant documents
        for doc_id in relevant_doc_ids:
            doc_scores[doc_id] = self._get_tf_idf_score(processed_query_terms, doc_id)

        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

        return sorted_docs[:top_n]

    def get_document(self, doc_id):
        # ... (Keep get_document as is) ...
        return self.documents.get(doc_id)

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Create some sample documents
    docs = {
        "doc1": "The quick brown fox jumps over the lazy dog.",
        "doc2": "A fast brown fox leaps over a sleepy dog.",
        "doc3": "The dog is lazy, the fox is quick.",
        "doc4": "Never jump over the lazy dog quickly.",
        "doc5": "Coding in Python is fun. Python is versatile."
    }

    # 2. Initialize the search engine
    search_engine = SimpleSearchEngine()

    # 3. Index the documents
    search_engine.index_documents(docs)

    # --- You can inspect the internal state if interested ---
    # print("--- Inverted Index ---")
    # for term, doc_ids in search_engine.inverted_index.items():
    #     print(f"'{term}': {doc_ids}")
    # print("\n--- Document Term Frequencies ---")
    # for doc_id, freqs in search_engine.doc_term_freqs.items():
    #     print(f"{doc_id}: {dict(freqs)}")
    # print("-" * 20)
    # --- End inspection ---

    # 4. Perform searches
    print("\n--- Search Results ---")

    query1 = "quick fox"
    results1 = search_engine.search(query1)
    print(f"Search for '{query1}':")
    for doc_id, score in results1:
        print(f"  - {doc_id} (Score: {score:.4f}): {search_engine.get_document(doc_id)}")

    print("-" * 10)

    query2 = "lazy dog"
    results2 = search_engine.search(query2)
    print(f"Search for '{query2}':")
    for doc_id, score in results2:
        print(f"  - {doc_id} (Score: {score:.4f}): {search_engine.get_document(doc_id)}")

    print("-" * 10)

    query3 = "python programming" # 'programming' won't be found after stemming 'coding'
    results3 = search_engine.search(query3)
    print(f"Search for '{query3}':")
    if results3:
        for doc_id, score in results3:
            print(f"  - {doc_id} (Score: {score:.4f}): {search_engine.get_document(doc_id)}")
    else:
        print("  No results found.")

    print("-" * 10)

    query4 = "versatile code" # 'code' stems from 'coding', 'versatile' is exact
    results4 = search_engine.search(query4)
    print(f"Search for '{query4}':")
    if results4:
        for doc_id, score in results4:
            print(f"  - {doc_id} (Score: {score:.4f}): {search_engine.get_document(doc_id)}")
    else:
        print("  No results found.")