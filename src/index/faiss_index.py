# src/index/faiss_index.py

# FAISS index module for RAG system
# Implements vector-based document retrieval using FAISS
# Uses sample documents for demonstration

import faiss
from src.ingest.loader import load_policies
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FAISSIndex:
    """
    FAISS Index for document retrieval.
    Loads policy documents, creates embeddings, and builds a FAISS index.
    
    Attributes:
        index (faiss.Index): The FAISS index for document retrieval.
        section_map (dict): Mapping of FAISS indices to document sections.
        model (SentenceTransformer): The sentence transformer model for embeddings.
    """
    def __init__(self, policy_dir="./data/raw_docs", model_name='all-MiniLM-L6-v2'):
        # Load policy documents
        self.sections = load_policies(policy_dir)
        if not self.sections:
            logger.warning("No policy sections loaded. FAISS index will be empty.")
            self.index = None
            self.section_map = {}
            self.model = None
            return
        
        self.model = self.__load_model(model_name)
        self.embeddings = self.__create_embeddings([section["text"] for section in self.sections])
        self.index = self.__build_faiss_index(self.embeddings)
        self.section_map = {i: self.sections[i] for i in range(len(self.sections))}

    def __load_model(self, model_name):
        """
        Load the sentence transformer model.

        Args:
            model_name (str): Name of the pre-trained model.
        Returns:
            SentenceTransformer: The loaded model.
        """
        try:
            return SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
        
    def __create_embeddings(self, texts):
        """
        Create embeddings for the given texts.

        Args:
            texts (list[str]): List of document texts.
        Returns:
            np.ndarray: Array of embeddings.
        """
        if not texts:
            return np.array([], dtype='float32')
        try:
            return self.model.encode(texts, convert_to_numpy=True).astype('float32')
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return np.array([], dtype='float32')
        
    def __build_faiss_index(self, embeddings):
        """
        Build the FAISS index from embeddings.

        Args:
            embeddings (np.ndarray): Array of document embeddings.
        Returns:
            faiss.Index: The built FAISS index.
        """
        if embeddings.size == 0:
            logger.warning("Empty embeddings array. FAISS index will not be created.")
            return None
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    def get_index(self):
        """
        Returns the FAISS index for document retrieval.
        """
        return self.index
    
    def get_section_map(self):
        """
        Returns the mapping of FAISS indices to document sections.
        """
        return self.section_map
    
    def get_model(self):
        """
        Returns the sentence transformer model used for embeddings.
        """
        return self.model

# Test usage
if __name__ == "__main__":
    faiss_index = FAISSIndex()
    index = faiss_index.get_index()
    if index is not None:
        print(f"Index contains {index.ntotal} documents.")
    else:
        print("FAISS index is empty.")