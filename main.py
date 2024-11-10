import gensim.downloader as api
from scipy import spatial
import numpy as np
from numpy.linalg import norm
import string

model = api.load('glove-wiki-gigaword-50')
model['tree']

def get_embedding(sentence, model=model):
     # Set all words to lowercase and remove punctuation
    sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the sentence into words (assuming spaces are used to separate words)
    words = sentence.split()
    
    # Initialize a list to store word embeddings
    word_embeddings = []
    
    # Iterate over each word in the sentence
    for word in words:
        if word in model:  # Check if the word is in the model's vocabulary
            word_embeddings.append(model[word])  # Get the embedding for the word
    
    # If no valid word embeddings were found, return a zero vector
    if not word_embeddings:
        return np.zeros(model.vector_size)
    
    # Convert the list of word embeddings to a NumPy array and take the average
    sentence_embedding = np.mean(np.array(word_embeddings), axis=0)
    
    return sentence_embedding
    pass


# We will use cosine similarity. It is probably the most common metric for comparing two vectors.
# The cosine similarity of two vectors is the cosine of the angle between them.
# Cosine similarity = 1 - cosine distance
def calculate_distance(sentence1, sentence2, model=model):
      # Get the embedding vectors for both sentences
    embedding1 = get_embedding(sentence1, model)
    embedding2 = get_embedding(sentence2, model)
    
    # Compute cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    magnitude1 = norm(embedding1)
    magnitude2 = norm(embedding2)
    
    # If either of the magnitudes is zero (e.g., if sentence embeddings are all zeros), return zero similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    
    return cosine_similarity
    pass

# Example words
sentence1 = "I took my dog to the park"
sentence2 = "My cat is in nature"

# Calculate distance between embeddings
distance = calculate_distance(sentence1, sentence2, model)
print(f"The distance between '{sentence1}' and '{sentence2}' is: {distance}")

from langchain_community.chat_models import ChatOpenAI
import os
from rag import RagEngine 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()