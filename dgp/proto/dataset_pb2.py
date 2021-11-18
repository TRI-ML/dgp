# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dgp/proto/dataset.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from dgp.proto import remote_pb2 as dgp_dot_proto_dot_remote__pb2
from dgp.proto import scene_pb2 as dgp_dot_proto_dot_scene__pb2
from dgp.proto import statistics_pb2 as dgp_dot_proto_dot_statistics__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='dgp/proto/dataset.proto',
  package='dgp.proto',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x17\x64gp/proto/dataset.proto\x12\tdgp.proto\x1a\x19google/protobuf/any.proto\x1a\x16\x64gp/proto/remote.proto\x1a\x15\x64gp/proto/scene.proto\x1a\x1a\x64gp/proto/statistics.proto\"\xec\x05\n\x08Ontology\x12\x35\n\nname_to_id\x18\x01 \x03(\x0b\x32!.dgp.proto.Ontology.NameToIdEntry\x12\x35\n\nid_to_name\x18\x02 \x03(\x0b\x32!.dgp.proto.Ontology.IdToNameEntry\x12\x33\n\x08\x63olormap\x18\x03 \x03(\x0b\x32!.dgp.proto.Ontology.ColormapEntry\x12\x31\n\x07isthing\x18\x04 \x03(\x0b\x32 .dgp.proto.Ontology.IsthingEntry\x12=\n\rsupercategory\x18\x05 \x03(\x0b\x32&.dgp.proto.Ontology.SupercategoryEntry\x12\x42\n\x10segmentation_ids\x18\x06 \x03(\x0b\x32(.dgp.proto.Ontology.SegmentationIdsEntry\x12\x11\n\tignore_id\x18\x07 \x01(\x03\x1a/\n\rNameToIdEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1a/\n\rIdToNameEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a(\n\x05\x43olor\x12\t\n\x01r\x18\x01 \x01(\x05\x12\t\n\x01g\x18\x02 \x01(\x05\x12\t\n\x01\x62\x18\x03 \x01(\x05\x1aJ\n\rColormapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12(\n\x05value\x18\x02 \x01(\x0b\x32\x19.dgp.proto.Ontology.Color:\x02\x38\x01\x1a.\n\x0cIsthingEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x08:\x02\x38\x01\x1a\x34\n\x12SupercategoryEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x36\n\x14SegmentationIdsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\"\xbf\x03\n\x0f\x44\x61tasetMetadata\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x15\n\rcreation_date\x18\x03 \x01(\t\x12\x0f\n\x07\x63reator\x18\x04 \x01(\t\x12*\n\x0b\x62ucket_path\x18\x05 \x01(\x0b\x32\x15.dgp.proto.RemotePath\x12\'\n\x08raw_path\x18\x06 \x01(\x0b\x32\x15.dgp.proto.RemotePath\x12\x13\n\x0b\x64\x65scription\x18\x07 \x01(\t\x12\x38\n\x06origin\x18\x08 \x01(\x0e\x32(.dgp.proto.DatasetMetadata.DatasetOrigin\x12\"\n\x1a\x61vailable_annotation_types\x18\t \x03(\x05\x12\x30\n\nstatistics\x18\n \x01(\x0b\x32\x1c.dgp.proto.DatasetStatistics\x12\x18\n\x10\x66rame_per_second\x18\x0b \x01(\x02\x12&\n\x08metadata\x18\x0c \x01(\x0b\x32\x14.google.protobuf.Any\")\n\rDatasetOrigin\x12\n\n\x06PUBLIC\x10\x00\x12\x0c\n\x08INTERNAL\x10\x01\"\xc7\x01\n\x0cSceneDataset\x12,\n\x08metadata\x18\x01 \x01(\x0b\x32\x1a.dgp.proto.DatasetMetadata\x12>\n\x0cscene_splits\x18\x02 \x03(\x0b\x32(.dgp.proto.SceneDataset.SceneSplitsEntry\x1aI\n\x10SceneSplitsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.dgp.proto.SceneFiles:\x02\x38\x01\"\xbe\x01\n\x06\x41gents\x12,\n\x08metadata\x18\x01 \x01(\x0b\x32\x1a.dgp.proto.DatasetMetadata\x12:\n\ragents_splits\x18\x02 \x03(\x0b\x32#.dgp.proto.Agents.AgentsSplitsEntry\x1aJ\n\x11\x41gentsSplitsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.dgp.proto.AgentFiles:\x02\x38\x01\"\x1f\n\nAgentFiles\x12\x11\n\tfilenames\x18\x01 \x03(\t*?\n\x0c\x44\x61tasetSplit\x12\t\n\x05TRAIN\x10\x00\x12\x07\n\x03VAL\x10\x01\x12\x08\n\x04TEST\x10\x02\x12\x11\n\rTRAIN_OVERFIT\x10\x03\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_any__pb2.DESCRIPTOR,dgp_dot_proto_dot_remote__pb2.DESCRIPTOR,dgp_dot_proto_dot_scene__pb2.DESCRIPTOR,dgp_dot_proto_dot_statistics__pb2.DESCRIPTOR,])

_DATASETSPLIT = _descriptor.EnumDescriptor(
  name='DatasetSplit',
  full_name='dgp.proto.DatasetSplit',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TRAIN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VAL', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TEST', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TRAIN_OVERFIT', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1769,
  serialized_end=1832,
)
_sym_db.RegisterEnumDescriptor(_DATASETSPLIT)

DatasetSplit = enum_type_wrapper.EnumTypeWrapper(_DATASETSPLIT)
TRAIN = 0
VAL = 1
TEST = 2
TRAIN_OVERFIT = 3


_DATASETMETADATA_DATASETORIGIN = _descriptor.EnumDescriptor(
  name='DatasetOrigin',
  full_name='dgp.proto.DatasetMetadata.DatasetOrigin',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PUBLIC', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INTERNAL', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1298,
  serialized_end=1339,
)
_sym_db.RegisterEnumDescriptor(_DATASETMETADATA_DATASETORIGIN)


_ONTOLOGY_NAMETOIDENTRY = _descriptor.Descriptor(
  name='NameToIdEntry',
  full_name='dgp.proto.Ontology.NameToIdEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dgp.proto.Ontology.NameToIdEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='dgp.proto.Ontology.NameToIdEntry.value', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=517,
  serialized_end=564,
)

_ONTOLOGY_IDTONAMEENTRY = _descriptor.Descriptor(
  name='IdToNameEntry',
  full_name='dgp.proto.Ontology.IdToNameEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dgp.proto.Ontology.IdToNameEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='dgp.proto.Ontology.IdToNameEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=566,
  serialized_end=613,
)

_ONTOLOGY_COLOR = _descriptor.Descriptor(
  name='Color',
  full_name='dgp.proto.Ontology.Color',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='r', full_name='dgp.proto.Ontology.Color.r', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='g', full_name='dgp.proto.Ontology.Color.g', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='b', full_name='dgp.proto.Ontology.Color.b', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=615,
  serialized_end=655,
)

_ONTOLOGY_COLORMAPENTRY = _descriptor.Descriptor(
  name='ColormapEntry',
  full_name='dgp.proto.Ontology.ColormapEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dgp.proto.Ontology.ColormapEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='dgp.proto.Ontology.ColormapEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=657,
  serialized_end=731,
)

_ONTOLOGY_ISTHINGENTRY = _descriptor.Descriptor(
  name='IsthingEntry',
  full_name='dgp.proto.Ontology.IsthingEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dgp.proto.Ontology.IsthingEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='dgp.proto.Ontology.IsthingEntry.value', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=733,
  serialized_end=779,
)

_ONTOLOGY_SUPERCATEGORYENTRY = _descriptor.Descriptor(
  name='SupercategoryEntry',
  full_name='dgp.proto.Ontology.SupercategoryEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dgp.proto.Ontology.SupercategoryEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='dgp.proto.Ontology.SupercategoryEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=781,
  serialized_end=833,
)

_ONTOLOGY_SEGMENTATIONIDSENTRY = _descriptor.Descriptor(
  name='SegmentationIdsEntry',
  full_name='dgp.proto.Ontology.SegmentationIdsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dgp.proto.Ontology.SegmentationIdsEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='dgp.proto.Ontology.SegmentationIdsEntry.value', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=835,
  serialized_end=889,
)

_ONTOLOGY = _descriptor.Descriptor(
  name='Ontology',
  full_name='dgp.proto.Ontology',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name_to_id', full_name='dgp.proto.Ontology.name_to_id', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='id_to_name', full_name='dgp.proto.Ontology.id_to_name', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='colormap', full_name='dgp.proto.Ontology.colormap', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='isthing', full_name='dgp.proto.Ontology.isthing', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='supercategory', full_name='dgp.proto.Ontology.supercategory', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='segmentation_ids', full_name='dgp.proto.Ontology.segmentation_ids', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ignore_id', full_name='dgp.proto.Ontology.ignore_id', index=6,
      number=7, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_ONTOLOGY_NAMETOIDENTRY, _ONTOLOGY_IDTONAMEENTRY, _ONTOLOGY_COLOR, _ONTOLOGY_COLORMAPENTRY, _ONTOLOGY_ISTHINGENTRY, _ONTOLOGY_SUPERCATEGORYENTRY, _ONTOLOGY_SEGMENTATIONIDSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=141,
  serialized_end=889,
)


_DATASETMETADATA = _descriptor.Descriptor(
  name='DatasetMetadata',
  full_name='dgp.proto.DatasetMetadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='dgp.proto.DatasetMetadata.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='version', full_name='dgp.proto.DatasetMetadata.version', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='creation_date', full_name='dgp.proto.DatasetMetadata.creation_date', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='creator', full_name='dgp.proto.DatasetMetadata.creator', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bucket_path', full_name='dgp.proto.DatasetMetadata.bucket_path', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='raw_path', full_name='dgp.proto.DatasetMetadata.raw_path', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='description', full_name='dgp.proto.DatasetMetadata.description', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='origin', full_name='dgp.proto.DatasetMetadata.origin', index=7,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='available_annotation_types', full_name='dgp.proto.DatasetMetadata.available_annotation_types', index=8,
      number=9, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='statistics', full_name='dgp.proto.DatasetMetadata.statistics', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='frame_per_second', full_name='dgp.proto.DatasetMetadata.frame_per_second', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='metadata', full_name='dgp.proto.DatasetMetadata.metadata', index=11,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DATASETMETADATA_DATASETORIGIN,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=892,
  serialized_end=1339,
)


_SCENEDATASET_SCENESPLITSENTRY = _descriptor.Descriptor(
  name='SceneSplitsEntry',
  full_name='dgp.proto.SceneDataset.SceneSplitsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dgp.proto.SceneDataset.SceneSplitsEntry.key', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='dgp.proto.SceneDataset.SceneSplitsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1468,
  serialized_end=1541,
)

_SCENEDATASET = _descriptor.Descriptor(
  name='SceneDataset',
  full_name='dgp.proto.SceneDataset',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metadata', full_name='dgp.proto.SceneDataset.metadata', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scene_splits', full_name='dgp.proto.SceneDataset.scene_splits', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_SCENEDATASET_SCENESPLITSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1342,
  serialized_end=1541,
)


_AGENTS_AGENTSSPLITSENTRY = _descriptor.Descriptor(
  name='AgentsSplitsEntry',
  full_name='dgp.proto.Agents.AgentsSplitsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dgp.proto.Agents.AgentsSplitsEntry.key', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='dgp.proto.Agents.AgentsSplitsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1660,
  serialized_end=1734,
)

_AGENTS = _descriptor.Descriptor(
  name='Agents',
  full_name='dgp.proto.Agents',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metadata', full_name='dgp.proto.Agents.metadata', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='agents_splits', full_name='dgp.proto.Agents.agents_splits', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_AGENTS_AGENTSSPLITSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1544,
  serialized_end=1734,
)


_AGENTFILES = _descriptor.Descriptor(
  name='AgentFiles',
  full_name='dgp.proto.AgentFiles',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='filenames', full_name='dgp.proto.AgentFiles.filenames', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1736,
  serialized_end=1767,
)

_ONTOLOGY_NAMETOIDENTRY.containing_type = _ONTOLOGY
_ONTOLOGY_IDTONAMEENTRY.containing_type = _ONTOLOGY
_ONTOLOGY_COLOR.containing_type = _ONTOLOGY
_ONTOLOGY_COLORMAPENTRY.fields_by_name['value'].message_type = _ONTOLOGY_COLOR
_ONTOLOGY_COLORMAPENTRY.containing_type = _ONTOLOGY
_ONTOLOGY_ISTHINGENTRY.containing_type = _ONTOLOGY
_ONTOLOGY_SUPERCATEGORYENTRY.containing_type = _ONTOLOGY
_ONTOLOGY_SEGMENTATIONIDSENTRY.containing_type = _ONTOLOGY
_ONTOLOGY.fields_by_name['name_to_id'].message_type = _ONTOLOGY_NAMETOIDENTRY
_ONTOLOGY.fields_by_name['id_to_name'].message_type = _ONTOLOGY_IDTONAMEENTRY
_ONTOLOGY.fields_by_name['colormap'].message_type = _ONTOLOGY_COLORMAPENTRY
_ONTOLOGY.fields_by_name['isthing'].message_type = _ONTOLOGY_ISTHINGENTRY
_ONTOLOGY.fields_by_name['supercategory'].message_type = _ONTOLOGY_SUPERCATEGORYENTRY
_ONTOLOGY.fields_by_name['segmentation_ids'].message_type = _ONTOLOGY_SEGMENTATIONIDSENTRY
_DATASETMETADATA.fields_by_name['bucket_path'].message_type = dgp_dot_proto_dot_remote__pb2._REMOTEPATH
_DATASETMETADATA.fields_by_name['raw_path'].message_type = dgp_dot_proto_dot_remote__pb2._REMOTEPATH
_DATASETMETADATA.fields_by_name['origin'].enum_type = _DATASETMETADATA_DATASETORIGIN
_DATASETMETADATA.fields_by_name['statistics'].message_type = dgp_dot_proto_dot_statistics__pb2._DATASETSTATISTICS
_DATASETMETADATA.fields_by_name['metadata'].message_type = google_dot_protobuf_dot_any__pb2._ANY
_DATASETMETADATA_DATASETORIGIN.containing_type = _DATASETMETADATA
_SCENEDATASET_SCENESPLITSENTRY.fields_by_name['value'].message_type = dgp_dot_proto_dot_scene__pb2._SCENEFILES
_SCENEDATASET_SCENESPLITSENTRY.containing_type = _SCENEDATASET
_SCENEDATASET.fields_by_name['metadata'].message_type = _DATASETMETADATA
_SCENEDATASET.fields_by_name['scene_splits'].message_type = _SCENEDATASET_SCENESPLITSENTRY
_AGENTS_AGENTSSPLITSENTRY.fields_by_name['value'].message_type = _AGENTFILES
_AGENTS_AGENTSSPLITSENTRY.containing_type = _AGENTS
_AGENTS.fields_by_name['metadata'].message_type = _DATASETMETADATA
_AGENTS.fields_by_name['agents_splits'].message_type = _AGENTS_AGENTSSPLITSENTRY
DESCRIPTOR.message_types_by_name['Ontology'] = _ONTOLOGY
DESCRIPTOR.message_types_by_name['DatasetMetadata'] = _DATASETMETADATA
DESCRIPTOR.message_types_by_name['SceneDataset'] = _SCENEDATASET
DESCRIPTOR.message_types_by_name['Agents'] = _AGENTS
DESCRIPTOR.message_types_by_name['AgentFiles'] = _AGENTFILES
DESCRIPTOR.enum_types_by_name['DatasetSplit'] = _DATASETSPLIT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Ontology = _reflection.GeneratedProtocolMessageType('Ontology', (_message.Message,), {

  'NameToIdEntry' : _reflection.GeneratedProtocolMessageType('NameToIdEntry', (_message.Message,), {
    'DESCRIPTOR' : _ONTOLOGY_NAMETOIDENTRY,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.Ontology.NameToIdEntry)
    })
  ,

  'IdToNameEntry' : _reflection.GeneratedProtocolMessageType('IdToNameEntry', (_message.Message,), {
    'DESCRIPTOR' : _ONTOLOGY_IDTONAMEENTRY,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.Ontology.IdToNameEntry)
    })
  ,

  'Color' : _reflection.GeneratedProtocolMessageType('Color', (_message.Message,), {
    'DESCRIPTOR' : _ONTOLOGY_COLOR,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.Ontology.Color)
    })
  ,

  'ColormapEntry' : _reflection.GeneratedProtocolMessageType('ColormapEntry', (_message.Message,), {
    'DESCRIPTOR' : _ONTOLOGY_COLORMAPENTRY,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.Ontology.ColormapEntry)
    })
  ,

  'IsthingEntry' : _reflection.GeneratedProtocolMessageType('IsthingEntry', (_message.Message,), {
    'DESCRIPTOR' : _ONTOLOGY_ISTHINGENTRY,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.Ontology.IsthingEntry)
    })
  ,

  'SupercategoryEntry' : _reflection.GeneratedProtocolMessageType('SupercategoryEntry', (_message.Message,), {
    'DESCRIPTOR' : _ONTOLOGY_SUPERCATEGORYENTRY,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.Ontology.SupercategoryEntry)
    })
  ,

  'SegmentationIdsEntry' : _reflection.GeneratedProtocolMessageType('SegmentationIdsEntry', (_message.Message,), {
    'DESCRIPTOR' : _ONTOLOGY_SEGMENTATIONIDSENTRY,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.Ontology.SegmentationIdsEntry)
    })
  ,
  'DESCRIPTOR' : _ONTOLOGY,
  '__module__' : 'dgp.proto.dataset_pb2'
  # @@protoc_insertion_point(class_scope:dgp.proto.Ontology)
  })
_sym_db.RegisterMessage(Ontology)
_sym_db.RegisterMessage(Ontology.NameToIdEntry)
_sym_db.RegisterMessage(Ontology.IdToNameEntry)
_sym_db.RegisterMessage(Ontology.Color)
_sym_db.RegisterMessage(Ontology.ColormapEntry)
_sym_db.RegisterMessage(Ontology.IsthingEntry)
_sym_db.RegisterMessage(Ontology.SupercategoryEntry)
_sym_db.RegisterMessage(Ontology.SegmentationIdsEntry)

DatasetMetadata = _reflection.GeneratedProtocolMessageType('DatasetMetadata', (_message.Message,), {
  'DESCRIPTOR' : _DATASETMETADATA,
  '__module__' : 'dgp.proto.dataset_pb2'
  # @@protoc_insertion_point(class_scope:dgp.proto.DatasetMetadata)
  })
_sym_db.RegisterMessage(DatasetMetadata)

SceneDataset = _reflection.GeneratedProtocolMessageType('SceneDataset', (_message.Message,), {

  'SceneSplitsEntry' : _reflection.GeneratedProtocolMessageType('SceneSplitsEntry', (_message.Message,), {
    'DESCRIPTOR' : _SCENEDATASET_SCENESPLITSENTRY,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.SceneDataset.SceneSplitsEntry)
    })
  ,
  'DESCRIPTOR' : _SCENEDATASET,
  '__module__' : 'dgp.proto.dataset_pb2'
  # @@protoc_insertion_point(class_scope:dgp.proto.SceneDataset)
  })
_sym_db.RegisterMessage(SceneDataset)
_sym_db.RegisterMessage(SceneDataset.SceneSplitsEntry)

Agents = _reflection.GeneratedProtocolMessageType('Agents', (_message.Message,), {

  'AgentsSplitsEntry' : _reflection.GeneratedProtocolMessageType('AgentsSplitsEntry', (_message.Message,), {
    'DESCRIPTOR' : _AGENTS_AGENTSSPLITSENTRY,
    '__module__' : 'dgp.proto.dataset_pb2'
    # @@protoc_insertion_point(class_scope:dgp.proto.Agents.AgentsSplitsEntry)
    })
  ,
  'DESCRIPTOR' : _AGENTS,
  '__module__' : 'dgp.proto.dataset_pb2'
  # @@protoc_insertion_point(class_scope:dgp.proto.Agents)
  })
_sym_db.RegisterMessage(Agents)
_sym_db.RegisterMessage(Agents.AgentsSplitsEntry)

AgentFiles = _reflection.GeneratedProtocolMessageType('AgentFiles', (_message.Message,), {
  'DESCRIPTOR' : _AGENTFILES,
  '__module__' : 'dgp.proto.dataset_pb2'
  # @@protoc_insertion_point(class_scope:dgp.proto.AgentFiles)
  })
_sym_db.RegisterMessage(AgentFiles)


_ONTOLOGY_NAMETOIDENTRY._options = None
_ONTOLOGY_IDTONAMEENTRY._options = None
_ONTOLOGY_COLORMAPENTRY._options = None
_ONTOLOGY_ISTHINGENTRY._options = None
_ONTOLOGY_SUPERCATEGORYENTRY._options = None
_ONTOLOGY_SEGMENTATIONIDSENTRY._options = None
_SCENEDATASET_SCENESPLITSENTRY._options = None
_AGENTS_AGENTSSPLITSENTRY._options = None
# @@protoc_insertion_point(module_scope)
