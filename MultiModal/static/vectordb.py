import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L12-v2"
        )

        # Set environment variables for disk-based storage
        storage_dir = "chroma_storage"
        os.environ["CHROMA_DB_DIR"] = storage_dir
        os.environ["CHROMA_DB_IN_MEMORY"] = "False"

        # Initialize ChromaDB client
        settings = Settings(is_persistent=True)
        self.chroma_client = chromadb.Client(settings=settings)

        # Initialize collection as None
        self.collection = None
        self.text_embeddings = None

    def create_collection(self, collection_name):
        # Create or load the collection
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def delete_collection(self, collection_name):
        # Delete the collection
        self.chroma_client.delete_collection(name=collection_name)

    # Method to populate the vector store with embeddings from a dataset
    def populate_vectors(self, dataset):
        if self.collection is None:
            raise RuntimeError(
                "Collection is not initialized. Call create_collection() first."
            )

        for i in dataset:
            timestamps = f"{i['timestamp']}"
            captions = f"{i['caption']}"
            transcriptions = f"{i['transcription']}"
            combined_text = f"Captions: {captions} | Transcriptions: {transcriptions} | Timestamps: {timestamps}"
            timestamp_embeddings = self.embedding_model.encode(timestamps).tolist()
            self.text_embeddings = self.embedding_model.encode(
                combined_text
            ).tolist()
            if (transcriptions):
                self.collection.add(
                    documents=[combined_text],
                    embeddings=[timestamp_embeddings],
                    ids=[f"{i['timestamp']}"],
                )
            self.collection.add(
                documents=[combined_text],
                embeddings=[self.text_embeddings],
                ids=[f"id_{i['timestamp']}_caption"],
            )

    # Method to search the ChromaDB collection for relevant context based on a query
    def search_context(self, query, n_results=3):
        if self.collection is None:
            raise RuntimeError(
                "Collection is not initialized. Call create_collection() first."
            )

        query_embeddings = self.embedding_model.encode(query).tolist()
        return self.collection.query(
            query_embeddings=query_embeddings, n_results=n_results
        )


vector_store = VectorStore()
# Example usage:
# vector_store = VectorStore('my_collection', 'src/chroma_storage')
# vector_store.create_collection('my_collection')  # Initialize or load the collection
# vector_store.populate_vectors(dataset)
# results = vector_store.search_context('your query')
