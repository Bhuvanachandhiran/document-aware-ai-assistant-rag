from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

class RAGAssistant:
    def __init__(self):
        # Embedding model (free)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Lightweight text generation model
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            framework="pt"
        )

        self.index = None
        self.documents = []

    def ingest_text(self, text):
        chunks = [chunk.strip() for chunk in text.split(".") if len(chunk) > 20]
        self.documents = chunks

        embeddings = self.embedder.encode(chunks)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def ask(self, question):
        if self.index is None:
            return "Please ingest a document before asking questions."

        question_embedding = self.embedder.encode([question])
        _, indices = self.index.search(
            np.array(question_embedding), k=3
        )

        context = " ".join([self.documents[i] for i in indices[0]])

        prompt = f"""
        Answer the question using the context below.
        If the answer is not present, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """

        response = self.generator(prompt, max_length=200)
        return response[0]["generated_text"]