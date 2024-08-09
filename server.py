from concurrent import futures
import grpc
import vector_database_pb2
import vector_database_pb2_grpc
import fitz
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Initialize models
nlp = spacy.load('en_core_web_sm')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
d = 384  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)

# Store documents, vectors, summaries, and chunk indices
documents = []
vectors = []
summaries = []
chunk_indices = []

class VectorDatabaseServicer(vector_database_pb2_grpc.VectorDatabaseServicer):
    def UploadDocument(self, request, context):
        try:
            global documents, vectors, summaries, chunk_indices

            # Validate the file name
            if not request.file_name.endswith(".pdf"):
                context.set_details("Invalid file format. Only PDF files are supported.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return vector_database_pb2.UploadDocumentResponse(message="Failed")

            # Save the uploaded PDF data to a file
            pdf_path = request.file_name
            with open(pdf_path, "wb") as f:
                f.write(request.pdf_data)

            # Validate the content of the PDF
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                context.set_details("Empty or unreadable PDF content.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return vector_database_pb2.UploadDocumentResponse(message="Failed")

            preprocessed_text = preprocess_text(text)

            # Chunk the preprocessed text
            chunks = chunk_text(preprocessed_text)
            doc_summaries = []

            for chunk_idx, chunk in enumerate(chunks):
                # Summarize each chunk
                summary = summarize_chunk(chunk)
                doc_summaries.append(summary)

                # Vectorize each chunk
                vector = text_to_vector(chunk)
                vectors.append(vector)
                chunk_indices.append((len(documents), chunk_idx))  # (document_index, chunk_index)

            documents.append(chunks)
            summaries.append(' '.join(doc_summaries))

            # Add all vectors to the FAISS index at once
            add_vectors_to_index(vectors[-len(chunks):])

            return vector_database_pb2.UploadDocumentResponse(message="Document uploaded successfully")

        except Exception as e:
            context.set_details(f"Internal error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return vector_database_pb2.UploadDocumentResponse(message="Failed")

    def SearchDocuments(self, request, context):
        try:
            if not request.query:
                context.set_details("Query string is empty.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return vector_database_pb2.SearchDocumentsResponse(results=[])

            if request.top_k <= 0:
                context.set_details("top_k must be a positive integer.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return vector_database_pb2.SearchDocumentsResponse(results=[])

            query = request.query
            top_k = request.top_k

            preprocessed_query = preprocess_text(query)
            query_vector = text_to_vector(preprocessed_query)

            most_similar_chunk_indices = search_vectors(query_vector, k=top_k)

            results = []
            for idx in most_similar_chunk_indices:
                if 0 <= idx < len(chunk_indices):
                    doc_idx, chunk_idx = chunk_indices[idx]
                    result = vector_database_pb2.SearchResult(
                        document_index=doc_idx,
                        chunk_index=chunk_idx,
                        chunk_text=documents[doc_idx][chunk_idx],
                        document_summary=summaries[doc_idx]
                    )
                    results.append(result)

            return vector_database_pb2.SearchDocumentsResponse(results=results)

        except Exception as e:
            context.set_details(f"Internal error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return vector_database_pb2.SearchDocumentsResponse(results=[])

    def SummarizeText(self, request, context):
        try:
            if not request.text.strip():
                context.set_details("Text for summarization is empty.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return vector_database_pb2.SummarizeTextResponse(summary="")

            text = request.text
            summary = summarize_chunk(text)
            return vector_database_pb2.SummarizeTextResponse(summary=summary)

        except Exception as e:
            context.set_details(f"Internal error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return vector_database_pb2.SummarizeTextResponse(summary="")

# Utility functions
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    doc = nlp(text)
    processed_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_
        if len(lemma) > 2 and not lemma.isdigit():
            processed_tokens.append(lemma)
    return ' '.join(processed_tokens)

def chunk_text(text, max_length=512):
    """Splits the text into chunks of a specified max length."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1  # Add 1 for the space
        if current_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def summarize_chunk(chunk, max_length=100, min_length=40):
    """Summarizes a text chunk."""
    summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def text_to_vector(text):
    return model.encode(text)

def add_vectors_to_index(vectors):
    index.add(np.array(vectors))

def search_vectors(query_vector, k=5):
    D, I = index.search(np.array([query_vector]), k)
    return I[0]

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vector_database_pb2_grpc.add_VectorDatabaseServicer_to_server(VectorDatabaseServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
