import grpc
import vector_database_pb2
import vector_database_pb2_grpc

def upload_document(stub, pdf_path, file_name):
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    request = vector_database_pb2.UploadDocumentRequest(pdf_data=pdf_data, file_name=file_name)
    response = stub.UploadDocument(request)
    print(f"UploadDocument Response for {file_name}:", response.message)

def search_documents(stub, query, top_k):
    request = vector_database_pb2.SearchDocumentsRequest(query=query, top_k=top_k)
    response = stub.SearchDocuments(request)
    for result in response.results:
        print(f"Document Index: {result.document_index}, Chunk Index: {result.chunk_index}")
        print(f"Chunk Text: {result.chunk_text}")
        print(f"Document Summary: {result.document_summary}\n")

def summarize_text(stub, text):
    request = vector_database_pb2.SummarizeTextRequest(text=text)
    response = stub.SummarizeText(request)
    print("SummarizeText Response:", response.summary)

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = vector_database_pb2_grpc.VectorDatabaseStub(channel)

        # Upload documents
        upload_document(stub, 'sample1.pdf', 'sample1.pdf')
        #upload_document(stub, 'sample2.pdf', 'sample2.pdf')
        #upload_document(stub, 'todotask.docx', 'todotask.docx' )
        

        # Search documents
        search_documents(stub, 'math', -2)

        # Summarize text
        summarize_text(stub, 'This is a long text that needs to be summarized.')

if __name__ == '__main__':
    run()