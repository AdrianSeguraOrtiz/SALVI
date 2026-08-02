"""Read-only in-memory representation used by scientific components."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Self

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from salvi.domain.enums import ColumnKind
from salvi.domain.models import ColumnMetadata, Dataset

FloatMatrix = npt.NDArray[np.float64]
BooleanMatrix = npt.NDArray[np.bool_]
IntegerMatrix = npt.NDArray[np.int32]
DiscreteValue = bool | str


def _read_only(array: npt.NDArray[np.generic]) -> None:
    array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class PreparedColumnMetadata:
    """Runtime column identity, including provenance for derived columns."""

    index: int
    name: str
    kind: ColumnKind
    categories: tuple[str, ...]
    source_column_index: int
    derivation: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0 or self.source_column_index < 0:
            raise ValueError("prepared column indices must be non-negative")
        if not self.name:
            raise ValueError("prepared column names must not be empty")
        if self.kind is ColumnKind.CATEGORICAL and not self.categories:
            raise ValueError("prepared categorical columns require categories")
        if self.kind is not ColumnKind.CATEGORICAL and self.categories:
            raise ValueError("only prepared categorical columns may have categories")

    @classmethod
    def from_source(cls, column: ColumnMetadata) -> Self:
        return cls(
            index=column.index,
            name=column.name,
            kind=column.kind,
            categories=column.categories,
            source_column_index=column.index,
        )


@dataclass(frozen=True, slots=True)
class NumericColumnStatistics:
    """Global robust statistics for one prepared numeric column."""

    column_index: int
    observed_count: int
    median: float | None
    percentile_05: float | None
    percentile_95: float | None
    robust_range: float
    zero_scale: bool

    def __post_init__(self) -> None:
        if self.column_index < 0 or self.observed_count < 0:
            raise ValueError("numeric statistic indices and counts must be non-negative")
        values = (self.median, self.percentile_05, self.percentile_95)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("numeric statistics must be finite when present")
        if not math.isfinite(self.robust_range) or self.robust_range < 0.0:
            raise ValueError("robust_range must be finite and non-negative")
        if self.observed_count == 0 and any(value is not None for value in values):
            raise ValueError("all-missing columns cannot have location statistics")
        if self.observed_count > 0 and any(value is None for value in values):
            raise ValueError("observed numeric columns require complete statistics")


@dataclass(frozen=True, slots=True, eq=False)
class PreparedDataset:
    """Immutable dataset shared by every component in one run.

    ``source_observed`` records which values existed in the canonical input.
    ``available`` records which runtime values can be consumed after an explicit
    missing-value policy. Keeping both prevents imputation from inflating the
    scientific support attributed to a candidate.
    """

    metadata: Dataset
    columns: tuple[PreparedColumnMetadata, ...]
    raw_table: pa.Table
    row_identifiers: pa.Array
    source_observed: BooleanMatrix
    available: BooleanMatrix
    numeric_values: FloatMatrix
    discrete_codes: IntegerMatrix
    numeric_column_indices: tuple[int, ...]
    boolean_column_indices: tuple[int, ...]
    categorical_column_indices: tuple[int, ...]
    numeric_positions: tuple[int, ...]
    discrete_column_indices: tuple[int, ...]
    discrete_positions: tuple[int, ...]
    discrete_labels: tuple[tuple[DiscreteValue, ...], ...]
    discrete_frequencies: tuple[tuple[int, ...], ...]
    numeric_statistics: tuple[NumericColumnStatistics, ...] = ()
    standardized_numeric_values: FloatMatrix | None = None
    _support: BooleanMatrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = self.metadata.row_count
        columns = len(self.columns)
        if self.raw_table.num_rows != rows or self.raw_table.num_columns != columns:
            raise ValueError("prepared raw table dimensions do not match prepared columns")
        if tuple(column.index for column in self.columns) != tuple(range(columns)):
            raise ValueError("prepared column indices must be contiguous and zero-based")
        if tuple(column.name for column in self.columns) != tuple(self.raw_table.column_names):
            raise ValueError("prepared column names do not match the raw table")
        if len({column.name for column in self.columns}) != columns:
            raise ValueError("prepared column names must be unique")
        if any(column.source_column_index >= self.metadata.column_count for column in self.columns):
            raise ValueError("prepared column source index is outside canonical metadata")
        if len(self.row_identifiers) != rows:
            raise ValueError("row identifier count does not match metadata")
        for label, mask in (
            ("source-observed", self.source_observed),
            ("available", self.available),
        ):
            if mask.shape != (rows, columns):
                raise ValueError(f"{label} mask dimensions do not match prepared data")
            if mask.dtype != np.bool_:
                raise ValueError(f"{label} mask must use Boolean storage")
        numeric_count = len(self.numeric_column_indices)
        if self.numeric_values.shape != (rows, numeric_count):
            raise ValueError("numeric view dimensions do not match numeric columns")
        if self.numeric_values.dtype != np.float64:
            raise ValueError("numeric view must use float64 storage")
        if len(self.numeric_positions) != columns:
            raise ValueError("numeric position map does not match prepared columns")
        discrete_count = len(self.discrete_column_indices)
        if self.discrete_codes.shape != (rows, discrete_count):
            raise ValueError("discrete view dimensions do not match discrete columns")
        if self.discrete_codes.dtype != np.int32:
            raise ValueError("discrete view must use int32 storage")
        if len(self.discrete_positions) != columns:
            raise ValueError("discrete position map does not match prepared columns")
        if not (len(self.discrete_labels) == len(self.discrete_frequencies) == discrete_count):
            raise ValueError("discrete metadata must align with discrete columns")
        for labels, frequencies in zip(
            self.discrete_labels, self.discrete_frequencies, strict=True
        ):
            if not labels or len(labels) != len(frequencies):
                raise ValueError("discrete labels and frequencies must align")
            if any(frequency < 0 for frequency in frequencies):
                raise ValueError("discrete frequencies must be non-negative")
        if self.standardized_numeric_values is not None:
            if self.standardized_numeric_values.shape != self.numeric_values.shape:
                raise ValueError("standardized numeric view has an invalid shape")
            if self.standardized_numeric_values.dtype != np.float64:
                raise ValueError("standardized numeric view must use float64 storage")
        if self.numeric_statistics and len(self.numeric_statistics) != numeric_count:
            raise ValueError("numeric statistics must align with numeric columns")
        expected_kinds = {
            ColumnKind.NUMERIC: self.numeric_column_indices,
            ColumnKind.BOOLEAN: self.boolean_column_indices,
            ColumnKind.CATEGORICAL: self.categorical_column_indices,
        }
        for kind, indices in expected_kinds.items():
            actual = tuple(column.index for column in self.columns if column.kind is kind)
            if indices != actual:
                raise ValueError(f"{kind.value.lower()} column index is inconsistent with metadata")
        _read_only(self.source_observed)
        _read_only(self.available)
        _read_only(self.numeric_values)
        _read_only(self.discrete_codes)
        if self.standardized_numeric_values is not None:
            _read_only(self.standardized_numeric_values)
        support = self.source_observed
        if self.available is not self.source_observed and np.any(
            self.source_observed & ~self.available
        ):
            support = self.source_observed & self.available
        _read_only(support)
        object.__setattr__(self, "_support", support)

    @classmethod
    def from_arrow(
        cls,
        metadata: Dataset,
        table: pa.Table,
        row_identifiers: pa.Array,
    ) -> Self:
        """Build the base prepared representation without scientific transforms."""

        combined = table.combine_chunks()
        columns = tuple(PreparedColumnMetadata.from_source(column) for column in metadata.columns)
        observed = cls._arrow_availability(combined)
        return cls._build(
            metadata=metadata,
            columns=columns,
            table=combined,
            row_identifiers=row_identifiers,
            source_observed=observed,
            available=observed,
        )

    @classmethod
    def _build(
        cls,
        *,
        metadata: Dataset,
        columns: tuple[PreparedColumnMetadata, ...],
        table: pa.Table,
        row_identifiers: pa.Array,
        source_observed: BooleanMatrix,
        available: BooleanMatrix,
    ) -> Self:
        combined = table.combine_chunks()
        numeric_indices = tuple(
            column.index for column in columns if column.kind is ColumnKind.NUMERIC
        )
        numeric_values = np.full(
            (metadata.row_count, len(numeric_indices)), np.nan, dtype=np.float64
        )
        numeric_positions = [-1] * len(columns)
        for position, column_index in enumerate(numeric_indices):
            numeric_positions[column_index] = position
            values = combined.column(column_index)
            converted = np.asarray(
                pc.cast(values, pa.float64()).to_numpy(zero_copy_only=False),
                dtype=np.float64,
            )
            numeric_values[:, position] = converted
            numeric_values[~available[:, column_index], position] = np.nan
        discrete_indices = tuple(
            column.index for column in columns if column.kind is not ColumnKind.NUMERIC
        )
        discrete_positions = [-1] * len(columns)
        discrete_codes = np.full((metadata.row_count, len(discrete_indices)), -1, dtype=np.int32)
        discrete_labels: list[tuple[DiscreteValue, ...]] = []
        discrete_frequencies: list[tuple[int, ...]] = []
        for position, column_index in enumerate(discrete_indices):
            discrete_positions[column_index] = position
            column = columns[column_index]
            labels: tuple[DiscreteValue, ...] = (
                (False, True) if column.kind is ColumnKind.BOOLEAN else tuple(column.categories)
            )
            code_by_label = {label: code for code, label in enumerate(labels)}
            values = combined.column(column_index).to_pylist()
            for row_index, value in enumerate(values):
                if not available[row_index, column_index]:
                    continue
                try:
                    discrete_codes[row_index, position] = code_by_label[value]
                except KeyError as error:
                    raise ValueError(
                        f"column {column.name!r} contains undeclared value {value!r}"
                    ) from error
            observed_codes = discrete_codes[:, position]
            frequencies = np.bincount(observed_codes[observed_codes >= 0], minlength=len(labels))
            discrete_labels.append(labels)
            discrete_frequencies.append(tuple(int(value) for value in frequencies))
        return cls(
            metadata=metadata,
            columns=columns,
            raw_table=combined,
            row_identifiers=row_identifiers,
            source_observed=source_observed,
            available=available,
            numeric_values=numeric_values,
            discrete_codes=discrete_codes,
            numeric_column_indices=numeric_indices,
            boolean_column_indices=tuple(
                column.index for column in columns if column.kind is ColumnKind.BOOLEAN
            ),
            categorical_column_indices=tuple(
                column.index for column in columns if column.kind is ColumnKind.CATEGORICAL
            ),
            numeric_positions=tuple(numeric_positions),
            discrete_column_indices=discrete_indices,
            discrete_positions=tuple(discrete_positions),
            discrete_labels=tuple(discrete_labels),
            discrete_frequencies=tuple(discrete_frequencies),
        )

    @staticmethod
    def _arrow_availability(table: pa.Table) -> BooleanMatrix:
        available = np.empty((table.num_rows, table.num_columns), dtype=np.bool_)
        for index in range(table.num_columns):
            available[:, index] = np.asarray(
                pc.is_valid(table.column(index)).to_numpy(zero_copy_only=False),
                dtype=np.bool_,
            )
        return available

    @property
    def row_count(self) -> int:
        return self.metadata.row_count

    @property
    def column_count(self) -> int:
        return len(self.columns)

    @property
    def source_column_count(self) -> int:
        return self.metadata.column_count

    @property
    def missing_count(self) -> int:
        return int(self.source_observed.size - np.count_nonzero(self.source_observed))

    @property
    def unavailable_count(self) -> int:
        return int(self.available.size - np.count_nonzero(self.available))

    @property
    def imputed_count(self) -> int:
        return int(np.count_nonzero(self.available & ~self.source_observed))

    @property
    def has_robust_scaling(self) -> bool:
        return self.standardized_numeric_values is not None

    @property
    def memory_bytes(self) -> int:
        """Approximate bytes owned by the prepared scientific representation."""

        arrays = (
            self.source_observed.nbytes + self.numeric_values.nbytes + self.discrete_codes.nbytes
        )
        if self.available is not self.source_observed:
            arrays += self.available.nbytes
        if self._support is not self.source_observed:
            arrays += self._support.nbytes
        if self.standardized_numeric_values is not None:
            arrays += self.standardized_numeric_values.nbytes
        return int(self.raw_table.nbytes + self.row_identifiers.nbytes + arrays)

    def source_observed_mask(self, column_index: int) -> npt.NDArray[np.bool_]:
        self._require_column(column_index)
        return self.source_observed[:, column_index]

    def available_mask(self, column_index: int) -> npt.NDArray[np.bool_]:
        self._require_column(column_index)
        return self.available[:, column_index]

    def support_mask(self, column_index: int) -> npt.NDArray[np.bool_]:
        """Values both present in the source and available to evaluation."""

        self._require_column(column_index)
        return self._support[:, column_index]

    def numeric_column(
        self,
        column_index: int,
        *,
        standardized: bool = False,
    ) -> npt.NDArray[np.float64]:
        self._require_column(column_index)
        position = self.numeric_positions[column_index]
        if position < 0:
            raise ValueError(f"column {column_index} is not numeric")
        source = self.numeric_values
        if standardized:
            if self.standardized_numeric_values is None:
                raise ValueError("robust numeric scaling has not been applied")
            source = self.standardized_numeric_values
        return source[:, position]

    def discrete_column(self, column_index: int) -> npt.NDArray[np.int32]:
        self._require_column(column_index)
        position = self.discrete_positions[column_index]
        if position < 0:
            raise ValueError(f"column {column_index} is not Boolean or categorical")
        return self.discrete_codes[:, position]

    def support_matrix(self) -> BooleanMatrix:
        """Return the immutable prepared-column support matrix for trusted kernels."""

        return self._support

    def numeric_matrix(self, *, standardized: bool = False) -> FloatMatrix:
        """Return the immutable numeric matrix for trusted scientific kernels."""

        if not standardized:
            return self.numeric_values
        if self.standardized_numeric_values is None:
            raise ValueError("robust numeric scaling has not been applied")
        return self.standardized_numeric_values

    def discrete_matrix(self) -> IntegerMatrix:
        """Return the immutable encoded discrete matrix for trusted kernels."""

        return self.discrete_codes

    def discrete_value(self, column_index: int, code: int) -> DiscreteValue:
        position = self._discrete_position(column_index)
        labels = self.discrete_labels[position]
        if code < 0 or code >= len(labels):
            raise IndexError(f"discrete code out of range for column {column_index}: {code}")
        return labels[code]

    def discrete_code(self, column_index: int, value: DiscreteValue) -> int:
        position = self._discrete_position(column_index)
        try:
            return self.discrete_labels[position].index(value)
        except ValueError as error:
            raise ValueError(
                f"value {value!r} is not declared for discrete column {column_index}"
            ) from error

    def discrete_global_frequencies(self, column_index: int) -> tuple[int, ...]:
        return self.discrete_frequencies[self._discrete_position(column_index)]

    def discrete_observed_cardinality(self, column_index: int) -> int:
        return sum(frequency > 0 for frequency in self.discrete_global_frequencies(column_index))

    def column(self, column_index: int) -> pa.ChunkedArray:
        self._require_column(column_index)
        return self.raw_table.column(column_index)

    def column_metadata(self, column_index: int) -> PreparedColumnMetadata:
        self._require_column(column_index)
        return self.columns[column_index]

    def row_identifier(self, row_index: int) -> str:
        if row_index < 0 or row_index >= self.row_count:
            raise IndexError(f"row index out of range: {row_index}")
        return str(self.row_identifiers[row_index].as_py())

    def with_replaced_values(self, table: pa.Table) -> Self:
        if table.num_columns != self.column_count or tuple(table.column_names) != tuple(
            column.name for column in self.columns
        ):
            raise ValueError("replacement table must preserve prepared columns")
        available = self._arrow_availability(table)
        return self._build(
            metadata=self.metadata,
            columns=self.columns,
            table=table,
            row_identifiers=self.row_identifiers,
            source_observed=self.source_observed,
            available=available,
        )

    def with_appended_boolean_columns(
        self,
        additions: tuple[tuple[str, npt.NDArray[np.bool_], int, str], ...],
    ) -> Self:
        if not additions:
            return self
        table = self.raw_table
        columns = list(self.columns)
        masks_are_shared = self.available is self.source_observed
        source_masks = [self.source_observed]
        available_masks = [self.available]
        existing_names = {column.name for column in columns}
        for name, values, source_column_index, derivation in additions:
            if name in existing_names:
                raise ValueError(f"prepared column name already exists: {name}")
            if values.shape != (self.row_count,) or values.dtype != np.bool_:
                raise ValueError("derived Boolean values must match the dataset rows")
            if source_column_index < 0 or source_column_index >= self.source_column_count:
                raise ValueError("derived column source index is outside canonical metadata")
            index = len(columns)
            table = table.append_column(name, pa.array(values, type=pa.bool_()))
            columns.append(
                PreparedColumnMetadata(
                    index=index,
                    name=name,
                    kind=ColumnKind.BOOLEAN,
                    categories=(),
                    source_column_index=source_column_index,
                    derivation=derivation,
                )
            )
            source_masks.append(np.ones((self.row_count, 1), dtype=np.bool_))
            available_masks.append(np.ones((self.row_count, 1), dtype=np.bool_))
            existing_names.add(name)
        source_observed = np.concatenate(source_masks, axis=1)
        available = source_observed if masks_are_shared else np.concatenate(available_masks, axis=1)
        return self._build(
            metadata=self.metadata,
            columns=tuple(columns),
            table=table,
            row_identifiers=self.row_identifiers,
            source_observed=source_observed,
            available=available,
        )

    def select_columns(self, indices: tuple[int, ...]) -> Self:
        if not indices:
            raise ValueError("at least one prepared column must remain")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("selected prepared column indices must be sorted and unique")
        for index in indices:
            self._require_column(index)
        columns = tuple(
            replace(self.columns[old_index], index=new_index)
            for new_index, old_index in enumerate(indices)
        )
        source_observed = self.source_observed[:, indices].copy()
        available = (
            source_observed
            if self.available is self.source_observed
            else self.available[:, indices].copy()
        )
        return self._build(
            metadata=self.metadata,
            columns=columns,
            table=self.raw_table.select(indices),
            row_identifiers=self.row_identifiers,
            source_observed=source_observed,
            available=available,
        )

    def with_robust_scaling(
        self,
        statistics: tuple[NumericColumnStatistics, ...],
        standardized_values: FloatMatrix,
    ) -> Self:
        return replace(
            self,
            numeric_statistics=statistics,
            standardized_numeric_values=standardized_values,
        )

    def _require_column(self, column_index: int) -> None:
        if column_index < 0 or column_index >= self.column_count:
            raise IndexError(f"column index out of range: {column_index}")

    def _discrete_position(self, column_index: int) -> int:
        self._require_column(column_index)
        position = self.discrete_positions[column_index]
        if position < 0:
            raise ValueError(f"column {column_index} is not Boolean or categorical")
        return position


__all__ = ["NumericColumnStatistics", "PreparedColumnMetadata", "PreparedDataset"]
