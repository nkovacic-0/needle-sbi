import json
from enum import Enum
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Self, Tuple

import yaml


@dataclass
class SerializableDataclass:
    """
    NOTE that fields without type annotations are not registered in the dataclass
    and therefore not included in the YAML serialization.
    """

    @classmethod
    def from_yaml(cls, yaml_file: str | Path, strict: bool = False) -> Self:
        """
        Load configuration from a YAML file.

        Args:
            yaml_file (str): Path to the YAML file.
            strict (bool):
                If True, raises an error if a key in the YAML file does not exist in the Config class.
                If False (default), adds all keys as attributes

        Returns:
            Config: Instance of this class
        """
        with open(yaml_file, "r") as f:
            new_config: Dict = yaml.safe_load(f)

        if not new_config:
            raise ValueError(f"File {yaml_file} is empty or invalid")

        for key in new_config.keys():
            if strict and key not in cls._fields():
                raise KeyError(f"Key '{key}' not found in {cls.__class__} class")

        return cls(**new_config)

    def to_yaml(self, yaml_file: str | Path) -> None:
        """
        Save the current configuration to a YAML file.
        Args:
            yaml_file (str): Path to the YAML file.
        """
        data = asdict(self)
        data = self._plain_serializable(data)
        with open(yaml_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
            # yaml.dump(asdict(self), f, default_flow_style=False)

    @staticmethod
    def _plain_serializable(obj: Any) -> Any:
        """Recursively convert Enum members to their plain value, so yaml.dump
        (and json.dump) never need a custom representer for Enum types — the
        resulting file only ever contains plain scalars/lists/dicts, safe to
        round-trip through yaml.safe_load.
        """
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: SerializableDataclass._plain_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SerializableDataclass._plain_serializable(v) for v in obj]
        return obj

    @classmethod
    def from_json(cls, json_file: str | Path, strict: bool = False) -> Self:
        """
        Load configuration from a JSON file.

        Args:
            json_file (str): Path to the JSON file.

        Returns:
            Self: Instance of this class
        """
        with open(json_file, "r") as f:
            new_config: Dict = json.load(f)

        if not new_config:
            raise ValueError(f"File {json_file} is empty or invalid")

        for key in new_config.keys():
            if strict and key not in cls._fields():
                raise KeyError(f"Key '{key}' not found in {cls.__class__} class")

        return cls(**new_config)

    def to_json(self, json_file: str | Path) -> None:
        """
        Save the current configuration to a JSON file.

        Args:
            json_file (str): Path to the JSON file.
        """
        with open(json_file, "w") as f:
            json.dump(asdict(self), f, indent=4, sort_keys=True, default=str)

    def __repr__(self) -> str:
        """Print with proper indentation."""
        return f"{self.__class__}({json.dumps(asdict(self), indent=4, sort_keys=True, default=str)})"

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def _fields(cls) -> Tuple[str, ...]:
        return tuple([f.name for f in fields(cls)])

    def fields(self) -> Tuple[str, ...]:
        return tuple([f.name for f in fields(self)])
