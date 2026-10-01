"""Derive Arrow schemas from Pydantic models, so each table layout is declared once, as a typed model.

Supported field types: ``str``, ``int``, ``float``, ``bool``, string ``Literal``s, ``list[T]``, nested models (as structs)
and ``X | None`` (nullable). Integer and float widths can be narrowed with ``Annotated[int, ArrowType(pa.uint64())]``.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Annotated, Literal, Union, get_args, get_origin

import pyarrow as pa
from pydantic import BaseModel


@dataclass(frozen=True)
class ArrowType:
    """Metadata that pins the Arrow type of an annotated field (for example ``uint64`` hashes or ``float32`` scores)."""

    type: pa.DataType


_SCALARS: dict[type, pa.DataType] = {str: pa.string(), int: pa.int64(), float: pa.float64(), bool: pa.bool_()}


def _unwrap_optional(annotation: object) -> tuple[object, bool]:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) != 1:
            raise TypeError(f"only X | None unions are supported, got {annotation!r}")
        return args[0], True
    return annotation, False


def arrow_type(annotation: object) -> tuple[pa.DataType, bool]:
    """Arrow type of one annotation, and whether it is nullable."""

    annotation, nullable = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is Annotated:
        base, *extras = get_args(annotation)
        pinned = [e.type for e in extras if isinstance(e, ArrowType)]
        if pinned:
            return pinned[0], nullable
        inner, inner_nullable = arrow_type(base)
        return inner, nullable or inner_nullable
    if origin is Literal:
        values = get_args(annotation)
        if not all(isinstance(v, str) for v in values):
            raise TypeError(f"only string literals are supported, got {annotation!r}")
        return pa.string(), nullable
    if origin is list:
        (item,) = get_args(annotation)
        item_type, _ = arrow_type(item)
        return pa.list_(item_type), nullable
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return pa.struct(list(arrow_schema(annotation))), nullable
    if annotation in _SCALARS:
        return _SCALARS[annotation], nullable
    raise TypeError(f"no Arrow mapping for {annotation!r}")


def arrow_schema(model: type[BaseModel]) -> pa.Schema:
    """Arrow schema with one field per model field, in declaration order."""

    fields = []
    for name, info in model.model_fields.items():
        annotation = info.annotation
        if info.metadata:  # pydantic moves Annotated extras into metadata
            annotation = Annotated[(annotation, *info.metadata)]
        dtype, nullable = arrow_type(annotation)
        fields.append(pa.field(name, dtype, nullable=nullable))
    return pa.schema(fields)


def table_from_models(rows: list[BaseModel], model: type[BaseModel]) -> pa.Table:
    """Arrow table of validated rows with exactly the model's schema."""

    return pa.Table.from_pylist([row.model_dump() for row in rows], schema=arrow_schema(model))


def conform(table: pa.Table, model: type[BaseModel]) -> pa.Table:
    """Select and cast a table's columns to the model's schema; raises if a required column is missing."""

    schema = arrow_schema(model)
    missing = [f.name for f in schema if f.name not in table.column_names]
    if missing:
        raise ValueError(f"table is missing columns {missing} required by {model.__name__}")
    return table.select(schema.names).cast(schema)
