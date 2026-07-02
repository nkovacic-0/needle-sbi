"""
Ingestor class for reading input files using Dask Awkward Arrays. Currently supports parquet and root
files. This class has low footprint since the dask delayed objects are not loaded at runtime, but
only when running `dask.compute()` on the Array. This makes it safe to use this class for all files
at once without running into memory issues.
"""

import reprlib
import logging
import warnings
from typing import Any, Literal, Self, Type

import dask_awkward as dak
import awkward as ak
import pydantic
import uproot

from needle.etl.array import (
    NestedArrayIndexer,
    brute_force_divisions,
    brute_force_length,
    resolve_paths,
)
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("etl", level="DEBUG")


class Ingestor:
    """Main class for reading input files. Can be extended to support other formats.

    Note:
        Currently supported formats:
            - `parquet`: Read parquet files using `dask_awkward.from_parquet`
            - `root`: Read root files using `uproot.dask`

    Attributes:
        array (dak.Array): Dask Awkward Array containing the data.
        fields (list[str]): List of fields in the array.
        num_classes (int): Number of fields in the array.
        length (int): Number of events in the array.
        SEPARATOR (str): Separator used for nested fields. Default is '.'. If it is None, then the
            fields are not nested.

    Important:
        New methods for reading other formats must implement the following:
        - Be added to the '__init__' method by being:
        - Listed as a supported format in the 'format' argument
        - [Optional] Added to the 'format == "automatic"' clause and corresponding logger
    """

    array: dak.Array  # type: ignore
    fields: list[str]
    num_classes: int
    length: int
    SEPARATOR: str = "."
    VALID_FORMATS = {"parquet", "root"}

    @pydantic.validate_call
    def __init__(
        self,
        paths: str | list[str],
        format: Literal["parquet", "root", "automatic"] = "automatic",
        columns: str | list[str] | None = None,
        reader_kwargs: dict[str, Any] | None = None,
        max_number_events: int = -1,
    ) -> None:
        """
        Read any input file and return an instance of this class.

        Args:
            paths (str | list[str]): Path(s) to the input file(s).
            format (Literal["parquet", "root", "automatic"]): Format of the input file(s).
                If `'automatic'`, the format will be inferred based on the file extension.
            columns (str | list[str] | None): List of columns to read from the input file(s).
            reader_kwargs (dict[str, Any]): Additional keyword arguments to pass to the Dask Awkward Array reader.
                - For 'parquet', these kwargs are passed to `dak.from_parquet`.
                - For 'root', they are passed to `uproot.dask`.

        Returns:
            Ingestor: An self of this class with the data loaded into a Dask Awkward Array.
        """
        paths = [paths] if isinstance(paths, str) else paths
        self.paths = paths 
        columns = [columns] if isinstance(columns, str) else columns
        format = self._resolve_format(format, paths[0])  # type: ignore
        reader_kwargs = reader_kwargs or {}

        match format:
            case "parquet":
                self.array = dak.from_parquet(paths, columns=columns, **reader_kwargs)  # type: ignore
            case "root":
                self.array = uproot.dask(paths, columns=columns, **reader_kwargs)  # type: ignore

        self.array.eager_compute_divisions()  # type: ignore
        self._inspect_array(self.array, paths)

        loaded_columns = NestedArrayIndexer.list_all_fields(self.array, separator=self.SEPARATOR, as_tuple=False)

        self._check_if_all_columns_found(columns or [], loaded_columns)
        self.fields = columns or loaded_columns
        self.num_classes = len(self.fields)

        if max_number_events > 0:
            self.array = self.array[0:max_number_events]

        self.length = self._get_length(self[self.fields[0]], paths)

        logger.info(f"Loaded {self.length} events with {self.num_classes} column(s): {reprlib.repr(self.fields)}")
        return None

    def __getitem__(self, field: str) -> dak.Array:  # type: ignore
        """Return the specified field from the array.

        Handles both flat and nested fields.

        Args:
            field (str): Field to return.

        Returns:
            dak.Array: Dask Awkward Array for the specified field.

        Raises:
            ValueError: If the specified field is not included in the list of fields.
        """
        if field not in self.fields:
            raise ValueError(f"Field '{field}' not found in array.")

        return NestedArrayIndexer.get_nested_field(self.array, field, self.SEPARATOR)

    # # TODO - test and validate this alternate padding calculation!
    # def get_padding_length(self, field: str, method: Literal["dask", "pyarrow"] = "dask") -> int:
    #     """Compute (and cache) the max per-event list length for a ragged field.
    #     Args:
    #         field (str): Field name, e.g. 'Lepton.pt'.
    #         method (Literal["dask", "pyarrow"]): Computation strategy.
    #             - "dask": lazy reduction over `self.array` via dask_awkward (default, always correct).
    #             - "pyarrow": direct per-file reads via `brute_force_max_list_length`, bypassing the
    #                 dask task graph. Opt-in; only reach for this if the "dask" path is a measured
    #                 bottleneck (e.g. many small files).
    #     Returns:
    #         int: Max list length for `field` across the full dataset.
    #     """
    #     if not hasattr(self, "_padding_lengths"):
    #         self._padding_lengths = {}
    #     if field not in self._padding_lengths:
    #         if method == "dask":
    #             column = NestedArrayIndexer.get_nested_field(self.array, field, self.SEPARATOR)
    #             # inline equivalent of PaddedDatasetBase.add_innermost_dimension
    #             # duplicated intentionally rather than shared, to avoid etl-ml crossdependancy 
    #             try:
    #                 _ = column[0][0]
    #             except IndexError:
    #                 column = ak.singletons(column, axis=0)
    #             def _get_length() -> int:
    #                 return int(ak.max(ak.ravel(ak.num(column, axis=1))))
    #             if logger.isEnabledFor(logging.DEBUG):
    #                 length = _get_length()
    #             else:
    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter("ignore")
    #                     length = _get_length()
    #         elif method == "pyarrow":
    #             length = brute_force_max_list_length(resolve_paths(self.paths), field)
    #         else:
    #             raise ValueError(f"Unknown method: {method}")
    #         self._padding_lengths[field] = length
    #     return self._padding_lengths[field]

    # alternative for the code above, should be safer to run TODO - vlaidate this get_padding_length approach!
    def get_padding_length(self, field: str, method: Literal["dask", "pyarrow"] = "dask") -> int:
        if not hasattr(self, "_padding_lengths"):
            self._padding_lengths = {}

        if field not in self._padding_lengths:
            if method == "dask":
                try:
                    length = self._get_padding_length_dask(field)
                except Exception as e:
                    logger.warning(
                        f"Dask-based padding length computation failed for field '{field}' "
                        f"({e!r}); falling back to pyarrow method."
                    )
                    length = brute_force_max_list_length(resolve_paths(self.paths), field)
            elif method == "pyarrow":
                length = brute_force_max_list_length(resolve_paths(self.paths), field)
            else:
                raise ValueError(f"Unknown method: {method}")

            self._padding_lengths[field] = length

        return self._padding_lengths[field]

    def _get_padding_length_dask(self, field: str) -> int:
        column = NestedArrayIndexer.get_nested_field(self.array, field, self.SEPARATOR)
        column = self._ensure_2d(column)

        def _get_length() -> int:
            return int(ak.max(ak.ravel(ak.num(column, axis=1))))

        if logger.isEnabledFor(logging.DEBUG):
            return _get_length()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _get_length()

    def _ensure_2d(self, column: dak.Array) -> dak.Array:
        """Determine whether `column` needs a singleton inner dimension added, using a
        cheap *eager* sample (one partition) rather than relying on lazy `dak.Array`
        indexing to raise IndexError — that behavior isn't guaranteed to match eager
        awkward semantics and could silently skip the reshape instead of erroring.
        """
        sample = column.partitions[0].compute()
        try:
            _ = sample[0][0]
            return column  # already 2D, sample confirms no reshape needed
        except IndexError:
            # to note: ak.singletons on a lazy dak.Array is fine it's only the detection step that is risky
            return ak.singletons(column, axis=0)


    def _check_if_all_columns_found(
        self,
        columns: list[str],
        loaded_fields: list[str],
    ) -> None:
        """Check that all columns were found when loading the data.

        Args:
            columns (list[str]): List of columns to check.
            loaded_fields (list[str]): List of columns that were actually loaded.

        Returns:
            None

        Raises:
            ValueError: If any of the columns in 'columns' are not found in 'loaded_fields'.
        """
        missing_columns = set(columns) - set(loaded_fields)

        if missing_columns:
            raise ValueError(f"Missing columns in ingested data: {missing_columns}")

    @staticmethod
    def _get_length(array: dak.Array, paths: str | list[str]) -> int:  # type: ignore
        """Try to determine the length of the array.

        Args:
            array (dak.Array): Input Dask Awkward Array.
            paths (str | list[str]): Paths as pattern or list of filepaths

        Returns:
            int: Length of the array

        Note:
            There are three cases to consider when reading the metadata of the input arrays.
            1. Best case, the array that was written to file implemented the __len__ method and was
                properly serialized to file. Using 'len(array)' will return the correct value.
            2. If dask_awkward is unable to determine the len() because the array is delayed, this
                usually raises an Exception. It is better to catch it and set the length to zero.
            3. Lastly, if the array did not implement __len__ at all, dask_awkward will return zero
                but not raise an Exception. This is the case if the outermost nesting is a Record and
                not an Array.
            In the last two cases, the simplest way to find the length of the array is to compute it
            file-by-file using pyarrow.parquet. This is the 'brute-force' method.
        """
        try:
            length = len(array)
        except ValueError:
            length = 0

        if length == 0:
            resolved_paths = resolve_paths(paths)

            if not resolved_paths:
                raise FileNotFoundError(f"No files could be found with pattern {paths}")

            length = brute_force_length(resolved_paths)  # NOTE Only valid for parquet
            logger.debug("Found length using 'brute force' method (pyarrow.parquet)")

        return length

    def _inspect_array(
        self,
        array: dak.Array,  # type: ignore
        paths: str | list[str],
    ) -> None:
        """Check if the input array has the required attributes.

        Args:
            array (dak.Array): Input Dask Awkward Array.
            paths (str | list[str]): Paths to the input files. Used for error messages.

        Raises:
            ValueError: If the input array does not have any fields.

        Note:
            Sideeffect: This method will try to find the divisions of the Array if they do not exist
            and assign them to `self.array.divisions`.

        NOTE: This part is a bit tricky due to nested fields. A field 'Lepton.pt' is nested because
        it must be accessed using 'array.Lepton.pt'. In this function we first list all the fields
        in our array, which is a list[str] (so completely flat). Then we need to check if the fields
        are actually nested (this happens by trying the full string, then by splitting the string into
        parts using a separator character, the actual value of which is irrelevant at this stage).
        Finally, we have to settle on one separator character if the fields are nested. To summarize, if
        the fields are flat, the separator is None. If they are nested, we return the separator so that
        afterwards we can access it more easily.
        """
        if not hasattr(array, "fields") or not array.fields:
            try:
                raise ValueError(
                    f"Input array does not have any fields: {array.fields}. "
                    "Please check the validity of the input columns. Available fields are: "
                    f"{dak.from_parquet(paths).fields}. "  # type: ignore  # NOTE Only valid for parquet
                )
            except AttributeError:
                raise ValueError("Input array does not have attribute 'fields' or is empty.")

        if not any(array.divisions):
            self.array._divisions = brute_force_divisions(resolve_paths(paths))

    @classmethod
    def _resolve_format(cls: Type[Self], fmt: str, path: str) -> str:
        """Find the correct file format based on the extension.

        Args:
            fmt: The format string. Can be "automatic" to detect format from file extension,
                 or a specific format name to validate against supported formats.
            path: The file path used for automatic format detection by extension matching.

        Returns:
            str: The resolved format name if valid.

        Raises:
            ValueError: If the format is not supported or cannot be determined from the file path.
        """
        if fmt == "automatic":
            for f in cls.VALID_FORMATS:
                if path.endswith(f".{f}"):
                    return f
            raise ValueError(
                f"Could not infer file format based on its extension: '{path}'. Currently supported "
                f"are: {cls.VALID_FORMATS}."
            )
        if fmt in cls.VALID_FORMATS:
            return fmt
        raise ValueError(f"Unsupported format: {fmt}. Currently supported are: {cls.VALID_FORMATS}.")
