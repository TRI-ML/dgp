import os
import subprocess
from pathlib import Path

_ROOT_DIRPATH = Path(__file__).parent.absolute()


def build_proto(proto_version: str):
    assert proto_version in {"proto3", "proto4"}, "Unsupported proto version."
    from grpc_tools import command
    command.build_package_protos(_ROOT_DIRPATH)


if __name__ == "__main__":
    proto_version = os.getenv("PROTO_VERSION", "proto3")
    build_proto(proto_version)
