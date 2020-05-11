# Copyright 2019 Toyota Research Institute. All rights reserved.
import os
import re

import boto3


class ValidationLengthError(Exception):
    """Thrown if length is outside of the range defined in schema."""


class ValidationPatternError(Exception):
    """Thrown if the string does not match the pattern defined in schema."""


class ValidationRepeatedError(Exception):
    """Thrown if the number of items in repeated fields is outside of the range defined in schema."""


class ValidationS3Error(Exception):
    """Thrown if the S3 path is invalid or the item does not exist in S3."""


class ValidationValueError(Exception):
    """Thrown if the value is outside of the range defined in schema."""


class ValidContent:
    """A class serves as a struct contains auxiliary schema to check the content of a protobuf field.
    A ValidContent instance validates if the content of a field satisfies all constraints set in this schema.
    For example, to check if a numeric content is greater or equal to 1 and less or equal to 100, please instantiate
        ValidContent(minimum=1, maximum=100)
    To check if a string content matches a pattern of s3 path and if the item exists in s3, do
        ValidContent(pattern="s3://", check_exists_s3=True)
    See dgp.proto.auxiliary.DATASET for examples.

    Parameters
    ----------
    maximum: int/float
        Maximum numeric value.
    minimum: int/float
        Minimum numeric value.
    max_len: int
        Maximum length.
    min_len: int
        Minimum length.
    max_items: int
        Maximum number of items in a repeated field.
    min_items: int
        Minimum number of items in a repeated field.
    pattern: str
        A pattern has to be matched.
    check_exists_s3: bool
        To check if the content exists in s3.
    """
    def __init__(
        self,
        maximum=None,
        minimum=None,
        max_len=None,
        min_len=None,
        max_items=None,
        min_items=None,
        pattern=None,
        check_exists_s3=None
    ):
        self.maximum = maximum
        self.minimum = minimum
        self.max_len = max_len
        self.min_len = min_len
        self.max_items = max_items
        self.min_items = min_items
        self.pattern = pattern
        if pattern and check_exists_s3 is None and re.search("s3://", pattern):
            self.check_exists_s3 = True
        else:
            self.check_exists_s3 = check_exists_s3

    def validate(self, name, content):
        if isinstance(content, str) and not content:
            raise ValidationValueError("{} is an empty string".format(name))
        if self.maximum is not None and content > self.maximum:
            raise ValidationValueError("{} is greater than the maximum of {}".format(name, self.maximum))
        if self.minimum is not None and content < self.minimum:
            raise ValidationValueError("{} is less than the minimum of {}".format(name, self.minimum))
        if self.max_len is not None and len(content) > self.max_len:
            raise ValidationLengthError("{} is too long. Maximum length is ".format(name, self.max_len))
        if self.min_len is not None and len(content) < self.min_len:
            raise ValidationLengthError("{} is too short. Minimum length is ".format(name, self.min_len))
        if self.pattern is not None and not re.search(self.pattern, content):
            raise ValidationPatternError("{} does not match the pattern of {}".format(name, self.pattern))
        if self.check_exists_s3 and not self.exists_s3_object(content):
            raise ValidationS3Error("{} does not exists in S3".format(content))

    @staticmethod
    def exists_s3_object(s3_path):
        """Check the existence of a remote file/bucket.
        Parameters
        ----------
        s3_path : string
            S3 path to a bucket ot file.

        Returns
        -------
        exists: bool
            True if exists.
        """
        s3 = boto3.session.Session().resource("s3")
        path_division = s3_path[len("s3://"):].split("/")
        bucket = s3.Bucket(path_division[0])
        if len(path_division) == 1:
            return bucket in s3.buckets.all()
        else:
            file_path = os.path.join(*path_division[1:])
            return len(list(bucket.objects.filter(Prefix=file_path))) != 0
