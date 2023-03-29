import argparse
import time
import grpc
import anyasand.pb.egoisticlily_pb2
from concurrent import futures
from anyasand.pb.egoisticlily_pb2_grpc import EgoisticLilyServiceServicer, add_EgoisticLilyServiceServicer_to_server
from anyasand.converter import Converter

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class EgoisticLilyGateway(EgoisticLilyServiceServicer):
    def __init__(self, dnn_model, db_path):
        self._converter = Converter(dnn_model, db_path)

    def Convert(self, request, context):
        ret_str = self._converter.convert(request.in_str)
        response = anyasand.pb.egoisticlily_pb2.ConvertResp(status=200, out_str=ret_str)
        return response


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model', help='model path', default="anya.mdl")
    arg_parser.add_argument('-d', '--dic_db', help='dictionary db path', default="anya-dic.db")
    arg_parser.add_argument('-p', '--port', help='server port number', default='50055')
    args = arg_parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_EgoisticLilyServiceServicer_to_server(EgoisticLilyGateway(args.model, args.dic_db), server)

    # portの設定
    port_str = '[::]:' + args.port
    server.add_insecure_port(port_str)
    server.start()
    print("ANYA Server Start!  Port %d" % int(args.port))
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)

    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()

