# Copyright 2019 Toyota Research Institute. All rights reserved.
from dgp.utils.validator_utils import ValidContent

SCHEMA_VALIDATION = {
    "dgp.proto.DatasetMetadata.creation_date": ValidContent(pattern=r'(\d+-\d+-\d+)'),
    "dgp.proto.DatasetMetadata.creator": ValidContent(pattern="@"),
    "dgp.proto.DatasetMetadata.description": ValidContent(max_len=200),
    "dgp.proto.RemotePath.value": ValidContent(pattern="s3://", check_exists_s3=False)
    # TODO: add more schemas
}
