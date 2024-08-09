# PDF Vector Database with gRPC API

This project implements a vector-based search engine for PDF documents using Sentence Transformers and FAISS. The project includes a gRPC API that allows users to upload PDF documents, search through them, and summarize text.

## Features

- **PDF Text Extraction**: Extracts text from PDF files.
- **Text Preprocessing**: Cleans and preprocesses the extracted text.
- **Chunking**: Splits large documents into smaller, manageable chunks.
- **Text Summarization**: Summarizes text using a pre-trained BART model.
- **Vectorization**: Converts text into high-dimensional vectors using Sentence Transformers.
- **FAISS Indexing**: Stores and searches vectors efficiently using FAISS.
- **gRPC API**: Provides endpoints for document upload, search, and summarization.

## Tool Selection

### 1. **PyMuPDF (Fitz) for PDF Text Extraction**
   - **Why?** PyMuPDF is a lightweight and fast library for PDF processing. It provides easy-to-use methods for extracting text from PDF files, which is crucial as the first step in this pipeline.
   - **Context**: In the context of this assignment, extracting text from PDF documents accurately and efficiently is essential, and PyMuPDF handles this task well.

### 2. **SpaCy for Text Preprocessing**
   - **Why?** SpaCy is a powerful NLP library that offers efficient tokenization, lemmatization, and stop-word removal. It’s well-suited for preprocessing tasks that require both speed and accuracy.
   - **Context**: Preprocessing the extracted text to remove noise and prepare it for vectorization is critical. SpaCy was chosen for its robust performance in these tasks.

### 3. **Sentence Transformers for Vectorization**
   - **Why?** Sentence Transformers, specifically the `all-MiniLM-L6-v2` model, provides an excellent trade-off between speed and accuracy for converting text into vectors. It captures semantic meaning effectively, which is vital for vector-based search.
   - **Context**: This model was selected because it allows us to generate high-quality embeddings quickly, which is crucial for creating a responsive search engine that operates over large text data.

### 4. **BART Model from Hugging Face for Text Summarization**
   - **Why?** The BART model is a state-of-the-art model for summarization tasks. It’s pre-trained on a large corpus and fine-tuned for summarization, making it highly effective for generating concise summaries of long texts.
   - **Context**: Summarization is an optional but valuable feature in this project, allowing users to quickly understand the content of a document. The BART model was chosen for its reliability and performance in producing meaningful summaries.

### 5. **FAISS for Vector Indexing and Search**
   - **Why?** FAISS (Facebook AI Similarity Search) is a highly optimized library for efficient similarity search over high-dimensional vectors. It supports various indexing methods to balance between search speed and memory usage.
   - **Context**: FAISS was chosen for its ability to handle large-scale vector searches with low latency, making it ideal for building a scalable and efficient search engine.

## Installation

### Prerequisites

- Python 3.7+
- `pip` (Python package installer)

### Dependencies

Install the required Python packages:

```bash
pip install grpcio grpcio-tools fitz spacy sentence-transformers transformers faiss-cpu numpy
```

Additionally, download the required SpaCy model:

```bash
python -m spacy download en_core_web_sm
```
### gRPC Code Generation

To generate the necessary gRPC code from the .proto file, run:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vector_database.proto
```
This will generate two files: vector_database_pb2.py and vector_database_pb2_grpc.py.

### Running the gRPC Server

To start the gRPC server, run:

```bash
python server.py
```
The server will start listening for incoming requests on localhost:50051.

## Using the gRPC API

### Modifying the Client Code

After starting the server, you will need to modify the client code ('client.py') based on the task you want to perform. Whether you want to upload a document, summarize its content, or search through the documents, you'll adjust the relevant parts of the 'client.py' script.

### Uploading a Document

To upload a PDF document to the server, use the 'upload_document' function in the client script:

```bash
upload_document(stub, 'path/to/sample1.pdf', 'sample1.pdf')
```

### Summarizing a PDF Document

To summarize the content of a PDF, extract the text and send it to the summarization endpoint:

```bash
pdf_text = extract_text_from_pdf('path/to/sample1.pdf')
summarize_text(stub, pdf_text)
```

### Searching Documents

To search through the uploaded documents based on a query, use the search_documents function:

```bash
search_documents(stub, 'math', 4)
```
In the context of vector-based search, k=4 refers to the number of most similar results (or "nearest neighbors") you want to retrieve from the database.

### Running the Client Script

Once you have modified the client code according to your needs, run the client script:

```bash
python client.py
```

## Error Handling

The API includes proper error handling and input validation:

- **Invalid File Format**: Only PDF files are accepted.
- **Empty Content**: PDF files must contain readable text.
- **Empty Queries**: Search queries cannot be empty.
- **Invalid Parameters**: Parameters such as `top_k` must be positive integers.

Error messages are returned via gRPC status codes and details.












