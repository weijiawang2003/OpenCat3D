"""Adapters from common Re-ID dataset layouts to CatNet manifests."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from catnet.reid.data import ReIDSample, write_manifest

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def folder_layout_to_samples(
    root: str | Path,
    *,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    recursive: bool = False,
    verify_images: bool = True,
    manifest_path: str | Path | None = None,
) -> list[ReIDSample]:
    """Convert ``root/cat_id/*.jpg|png`` folders into Re-ID samples."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"folder layout root is not a directory: {root_path}")

    allowed_extensions = _normalise_extensions(extensions)
    samples: list[ReIDSample] = []
    identity_dirs = (path for path in root_path.iterdir() if path.is_dir())
    for identity_dir in sorted(identity_dirs, key=_path_key):
        candidates = identity_dir.rglob("*") if recursive else identity_dir.iterdir()
        image_paths = (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in allowed_extensions
        )
        for image_path in sorted(image_paths, key=_path_key):
            _verify_image(image_path, verify_images=verify_images)
            samples.append(ReIDSample(image_path=str(image_path), cat_id=identity_dir.name))

    _write_if_requested(manifest_path, samples)
    return samples


def folder_layout_to_manifest(
    root: str | Path,
    manifest_path: str | Path,
    **kwargs: Any,
) -> list[ReIDSample]:
    """Write a manifest for a generic folder layout and return the samples."""

    return folder_layout_to_samples(root, manifest_path=manifest_path, **kwargs)


def petface_csv_to_samples(
    csv_path: str | Path,
    *,
    image_root: str | Path | None = None,
    identity_column: str = "identity",
    image_path_column: str = "image_path",
    metadata_columns: Sequence[str] | None = None,
    verify_images: bool = True,
    manifest_path: str | Path | None = None,
) -> list[ReIDSample]:
    """Convert PetFace-like CSV records into Re-ID samples.

    Relative image paths are resolved under ``image_root``. When ``image_root``
    is omitted, paths are resolved relative to the CSV file's parent directory.
    """

    csv_file = Path(csv_path)
    root = Path(image_root) if image_root is not None else csv_file.parent
    return petface_records_to_samples(
        _read_csv_records(csv_file),
        image_root=root,
        identity_column=identity_column,
        image_path_column=image_path_column,
        metadata_columns=metadata_columns,
        verify_images=verify_images,
        manifest_path=manifest_path,
    )


def petface_records_to_samples(
    records: Iterable[Mapping[str, Any]] | Any,
    *,
    image_root: str | Path | None = None,
    identity_column: str = "identity",
    image_path_column: str = "image_path",
    metadata_columns: Sequence[str] | None = None,
    verify_images: bool = True,
    manifest_path: str | Path | None = None,
) -> list[ReIDSample]:
    """Convert iterable or dataframe-like PetFace records into Re-ID samples."""

    samples: list[ReIDSample] = []
    for index, record in enumerate(_coerce_records(records), start=1):
        cat_id = _required_text(record, identity_column, record_index=index)
        image_path = _image_path(
            _required_text(record, image_path_column, record_index=index),
            image_root=image_root,
        )
        _verify_image(image_path, verify_images=verify_images)
        metadata = _metadata(
            record,
            metadata_columns=metadata_columns,
            record_index=index,
        )
        samples.append(
            ReIDSample(
                image_path=str(image_path),
                cat_id=cat_id,
                metadata=metadata,
            )
        )

    _write_if_requested(manifest_path, samples)
    return samples


def petface_csv_to_manifest(
    csv_path: str | Path,
    manifest_path: str | Path,
    **kwargs: Any,
) -> list[ReIDSample]:
    """Write a manifest for PetFace-like CSV records and return the samples."""

    return petface_csv_to_samples(csv_path, manifest_path=manifest_path, **kwargs)


def wildlife_csv_to_samples(
    csv_path: str | Path,
    *,
    image_root: str | Path | None = None,
    identity_column: str = "identity",
    image_column: str = "image",
    date_column: str | None = "date",
    site_column: str | None = "site",
    camera_column: str | None = "camera",
    metadata_columns: Sequence[str] | None = None,
    verify_images: bool = True,
    manifest_path: str | Path | None = None,
) -> list[ReIDSample]:
    """Convert WildlifeDatasets-like CSV records into Re-ID samples."""

    csv_file = Path(csv_path)
    root = Path(image_root) if image_root is not None else csv_file.parent
    return wildlife_records_to_samples(
        _read_csv_records(csv_file),
        image_root=root,
        identity_column=identity_column,
        image_column=image_column,
        date_column=date_column,
        site_column=site_column,
        camera_column=camera_column,
        metadata_columns=metadata_columns,
        verify_images=verify_images,
        manifest_path=manifest_path,
    )


def wildlife_records_to_samples(
    records: Iterable[Mapping[str, Any]] | Any,
    *,
    image_root: str | Path | None = None,
    identity_column: str = "identity",
    image_column: str = "image",
    date_column: str | None = "date",
    site_column: str | None = "site",
    camera_column: str | None = "camera",
    metadata_columns: Sequence[str] | None = None,
    verify_images: bool = True,
    manifest_path: str | Path | None = None,
) -> list[ReIDSample]:
    """Convert iterable or dataframe-like WildlifeDatasets records."""

    samples: list[ReIDSample] = []
    for index, record in enumerate(_coerce_records(records), start=1):
        cat_id = _required_text(record, identity_column, record_index=index)
        image_path = _image_path(
            _required_text(record, image_column, record_index=index),
            image_root=image_root,
        )
        _verify_image(image_path, verify_images=verify_images)
        metadata = _metadata(
            record,
            metadata_columns=metadata_columns,
            record_index=index,
        )
        samples.append(
            ReIDSample(
                image_path=str(image_path),
                cat_id=cat_id,
                captured_at=_optional_text(record, date_column, record_index=index),
                site_id=_optional_text(record, site_column, record_index=index),
                camera_id=_optional_text(record, camera_column, record_index=index),
                metadata=metadata,
            )
        )

    _write_if_requested(manifest_path, samples)
    return samples


def wildlife_dataframe_to_samples(
    dataframe: Any,
    **kwargs: Any,
) -> list[ReIDSample]:
    """Convert a pandas-like dataframe without importing pandas."""

    return wildlife_records_to_samples(dataframe, **kwargs)


def wildlife_datasets_to_samples(
    source: str | Path | Iterable[Mapping[str, Any]] | Any,
    **kwargs: Any,
) -> list[ReIDSample]:
    """Convert a WildlifeDatasets-like CSV path, records iterable, or dataframe."""

    if isinstance(source, str | Path):
        return wildlife_csv_to_samples(source, **kwargs)
    return wildlife_records_to_samples(source, **kwargs)


def wildlife_csv_to_manifest(
    csv_path: str | Path,
    manifest_path: str | Path,
    **kwargs: Any,
) -> list[ReIDSample]:
    """Write a manifest for WildlifeDatasets-like CSV records and return samples."""

    return wildlife_csv_to_samples(csv_path, manifest_path=manifest_path, **kwargs)


def wildlife_records_to_manifest(
    records: Iterable[Mapping[str, Any]] | Any,
    manifest_path: str | Path,
    **kwargs: Any,
) -> list[ReIDSample]:
    """Write a manifest for WildlifeDatasets-like records and return samples."""

    return wildlife_records_to_samples(records, manifest_path=manifest_path, **kwargs)


generic_folder_to_samples = folder_layout_to_samples
generic_folder_to_manifest = folder_layout_to_manifest


def _normalise_extensions(extensions: Sequence[str]) -> set[str]:
    if not extensions:
        raise ValueError("extensions must not be empty")
    return {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")
        return [dict(row) for row in reader]


def _coerce_records(records: Iterable[Mapping[str, Any]] | Any) -> list[Mapping[str, Any]]:
    if isinstance(records, Mapping):
        raise TypeError("records must be an iterable of row mappings, not a single mapping")

    to_dict = getattr(records, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict(orient="records")
        except TypeError:
            records = to_dict("records")

    output: list[Mapping[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        output.append(record)
    return output


def _required_text(
    record: Mapping[str, Any],
    column: str,
    *,
    record_index: int,
) -> str:
    value = _column_value(record, column, record_index=record_index)
    text = _text_or_none(value)
    if text is None:
        raise ValueError(f"record {record_index} has an empty value for column {column!r}")
    return text


def _optional_text(
    record: Mapping[str, Any],
    column: str | None,
    *,
    record_index: int,
) -> str | None:
    if column is None:
        return None
    return _text_or_none(_column_value(record, column, record_index=record_index))


def _column_value(record: Mapping[str, Any], column: str, *, record_index: int) -> Any:
    if column not in record:
        raise KeyError(f"record {record_index} is missing column {column!r}")
    return record[column]


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    if hasattr(value, "isoformat") and not isinstance(value, str):
        value = value.isoformat()
    text = str(value).strip()
    return text or None


def _image_path(raw_path: str, *, image_root: str | Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() or image_root is None:
        return path
    return Path(image_root).expanduser() / path


def _metadata(
    record: Mapping[str, Any],
    *,
    metadata_columns: Sequence[str] | None,
    record_index: int,
) -> dict[str, Any]:
    if metadata_columns is None:
        return {}

    metadata: dict[str, Any] = {}
    for column in metadata_columns:
        value = _column_value(record, column, record_index=record_index)
        if _text_or_none(value) is None:
            continue
        metadata[column] = _json_safe(value)
    return metadata


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _verify_image(path: Path, *, verify_images: bool) -> None:
    if not verify_images:
        return
    if not path.is_file():
        raise FileNotFoundError(f"image file does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"invalid image file: {path}") from exc


def _write_if_requested(
    manifest_path: str | Path | None,
    samples: list[ReIDSample],
) -> None:
    if manifest_path is not None:
        write_manifest(Path(manifest_path), samples)


def _path_key(path: Path) -> tuple[str, str]:
    value = path.as_posix()
    return value.lower(), value


__all__ = [
    "DEFAULT_IMAGE_EXTENSIONS",
    "folder_layout_to_manifest",
    "folder_layout_to_samples",
    "generic_folder_to_manifest",
    "generic_folder_to_samples",
    "petface_csv_to_manifest",
    "petface_csv_to_samples",
    "petface_records_to_samples",
    "wildlife_csv_to_manifest",
    "wildlife_csv_to_samples",
    "wildlife_dataframe_to_samples",
    "wildlife_datasets_to_samples",
    "wildlife_records_to_manifest",
    "wildlife_records_to_samples",
]
