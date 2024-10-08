# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import vector_database_pb2 as vector__database__pb2


class VectorDatabaseStub(object):
    """The VectorDatabase service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.UploadDocument = channel.unary_unary(
                '/vector_database.VectorDatabase/UploadDocument',
                request_serializer=vector__database__pb2.UploadDocumentRequest.SerializeToString,
                response_deserializer=vector__database__pb2.UploadDocumentResponse.FromString,
                )
        self.SearchDocuments = channel.unary_unary(
                '/vector_database.VectorDatabase/SearchDocuments',
                request_serializer=vector__database__pb2.SearchDocumentsRequest.SerializeToString,
                response_deserializer=vector__database__pb2.SearchDocumentsResponse.FromString,
                )
        self.SummarizeText = channel.unary_unary(
                '/vector_database.VectorDatabase/SummarizeText',
                request_serializer=vector__database__pb2.SummarizeTextRequest.SerializeToString,
                response_deserializer=vector__database__pb2.SummarizeTextResponse.FromString,
                )


class VectorDatabaseServicer(object):
    """The VectorDatabase service definition.
    """

    def UploadDocument(self, request, context):
        """Uploads a PDF document and processes it.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SearchDocuments(self, request, context):
        """Searches through documents based on a query.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SummarizeText(self, request, context):
        """Summarizes a text.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_VectorDatabaseServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'UploadDocument': grpc.unary_unary_rpc_method_handler(
                    servicer.UploadDocument,
                    request_deserializer=vector__database__pb2.UploadDocumentRequest.FromString,
                    response_serializer=vector__database__pb2.UploadDocumentResponse.SerializeToString,
            ),
            'SearchDocuments': grpc.unary_unary_rpc_method_handler(
                    servicer.SearchDocuments,
                    request_deserializer=vector__database__pb2.SearchDocumentsRequest.FromString,
                    response_serializer=vector__database__pb2.SearchDocumentsResponse.SerializeToString,
            ),
            'SummarizeText': grpc.unary_unary_rpc_method_handler(
                    servicer.SummarizeText,
                    request_deserializer=vector__database__pb2.SummarizeTextRequest.FromString,
                    response_serializer=vector__database__pb2.SummarizeTextResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'vector_database.VectorDatabase', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class VectorDatabase(object):
    """The VectorDatabase service definition.
    """

    @staticmethod
    def UploadDocument(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/vector_database.VectorDatabase/UploadDocument',
            vector__database__pb2.UploadDocumentRequest.SerializeToString,
            vector__database__pb2.UploadDocumentResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SearchDocuments(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/vector_database.VectorDatabase/SearchDocuments',
            vector__database__pb2.SearchDocumentsRequest.SerializeToString,
            vector__database__pb2.SearchDocumentsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SummarizeText(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/vector_database.VectorDatabase/SummarizeText',
            vector__database__pb2.SummarizeTextRequest.SerializeToString,
            vector__database__pb2.SummarizeTextResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
