import chromadb

class ChromaDB():
    def __init__(self, port):
        self.chroma_client = chromadb.HttpClient(host="localhost", port=port)

        self.collection = self.chroma_client.get_or_create_collection(name="video_summaries")
    
    def add(self, documents, metadata):
        self.collection.add(documents=documents, metadatas=[metadata])
    
    def query(self, text):
        results = self.collection.query(query_texts=[text], n_results=5)
        return results