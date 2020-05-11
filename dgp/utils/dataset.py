# Copyright 2019-2020 Toyota Research Institute.  All rights reserved.
"""Dataset related utility functions for listing relevant files within a dataset and datum"""


def list_datum_files(datum):
    """Retrieve paths of all file paths described in a datum.

    Parameters
    ----------
    datum: dbp.proto.dataset_pb2.Sample.Datum
        The datum to be parsed

    Returns
    -------
    filepaths: list
        A list of all files described by the datum.
    """
    datum_attr = datum.datum.WhichOneof('datum_oneof')
    filepaths = []
    try:
        datum_value = getattr(datum.datum, datum_attr)
    except Exception as e:
        raise Exception("Unsupported datum type {}. Error: {}".format(datum_attr, e.message))
    filepaths.append(datum_value.filename)
    for _, filename in datum_value.annotations.items():
        filepaths.append(filename)
    return filepaths
