# Copyright 2019 Toyota Research Institute. All rights reserved.
import base64
import math
from importlib import import_module

from google.protobuf import descriptor
from google.protobuf.json_format import (_FLOAT_TYPES, _INFINITY, _INT64_TYPES,
                                         _NAN, _NEG_INFINITY, Parse)


def validate_protobuf(dataset_path, message):
    """Check if a protobuf file is DGP compliant. Throws exceptions if invalid.

    Parameters
    ----------
    dataset_path: string
        Path to the dataset file (.json) to be validated.
    message: string
        Target message name to be validated (dgp.proto.dataset.Dataset).
    """
    modules = message.split('.')
    assert len(modules) >= 4, '{} needs to be at least 4-tuple valued'.format(message)
    try:
        top_module = modules[0]
        proto, message_name = modules[-2], modules[-1]
        compiled_proto_module = '{}_pb2'.format(proto)
        module_object = import_module("{}.{}".format('.'.join(modules[:-2]), compiled_proto_module))
        target_message = getattr(module_object, message_name)
    except Exception as e:
        raise ValueError('Failed to parse {} proto message: {}'.format(message, e.message))

    if not dataset_path.endswith((".json", ".pb", ".prb")):
        raise IOError("{} is not a supported file format. Supported file extenstions: .json, .pb, .prb")

    is_json = dataset_path.endswith(".json")
    with open(dataset_path, "r" if is_json else "rb") as dataset_file:
        if is_json:
            message = Parse(dataset_file.read(), target_message())
        else:
            message = target_message()
            target_message().ParseFromString(dataset_file.read())

        schema = getattr('{}.validation', top_module, 'SCHEMA_VALIDATION')
        validate_message(message, schema)

    print("{} is valid".format(dataset_path))


def validate_message(message, schema):
    """Validate a protobuf message instance. Throws exception if a field value does not match the schema.
    Parameters
    ----------
    message: protobuf message instance
        The protocol buffers message instance to be validated.
    schema: dict
        A dictionary containing field names to NamedTuples of content schema.
    """
    for field, value in message.ListFields():
        if _is_map_entry(field):
            v_field = field.message_type.fields_by_name['value']
            for key in value:
                _validate_field(v_field, value[key], schema)
        elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
            for v in value:
                _validate_field(field, v, schema)
        else:
            _validate_field(field, value, schema)


def _is_map_entry(field):
    """Returns True if the field is a map entry, vice versa.
    Parameters
    ----------
    field: FieldDescriptor
        Field.

    Returns
    -------
    exists: bool
        True if the filed is a map entry.
    """
    return (
        field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and field.message_type.has_options
        and field.message_type.GetOptions().map_entry
    )


def _validate_content(full_name, content, name_to_schema):
    """Validate a single field content if the field name is in the auxiliary schema.
    Parameters
    ----------
    full_name: str
        Full name of the field.
    content: Any
        Field value.
    """
    if full_name in name_to_schema:
        schema = name_to_schema[full_name]
        schema.validate(full_name, content)


def _validate_field(field, value, schema):
    """Traverse fields, convert field value and call _validate_content to check
    if contents satisfy the auxiliary schema.
    Parameters
    ----------
    field: FieldDescriptor
        Field.
    value: Any
        Value.
    schema: Dict
        Auxiliary content schema imported from dgp.proto.auxiliary_schema
    """
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        return validate_message(value, schema)

    field_value = value
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
        enum_value = field.enum_type.values_by_number.get(value, None)
        if enum_value is not None:
            field_value = enum_value.name
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
        if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
            field_value = base64.b64encode(value).decode('utf-8')
        else:
            field_value = value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
        field_value = bool(value)
    elif field.cpp_type in _INT64_TYPES:
        field_value = str(value)
    elif field.cpp_type in _FLOAT_TYPES:
        if math.isinf(value):
            field_value = _NEG_INFINITY if value < 0.0 else _INFINITY
        if math.isnan(value):
            field_value = _NAN

    _validate_content(field.full_name, field_value, schema)
