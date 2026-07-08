from needle.utils.config_schema import DatasetConfig
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("etl")


def validate_grouped_config(cfg: DatasetConfig) -> None:
    """Cross-field validation for the grouped particle-feature config fields.

    No-op if `feature_columns_grouped` is not set (i.e. the flat/ragged
    pipeline is being used instead). Raises ValueError on the first problem
    found, with a message naming the offending field/value.
    """
    grid = cfg.feature_columns_grouped
    if grid is None:
        return

    if not grid or any(not row for row in grid):
        err_msg = ("feature_columns_grouped must be non-empty, with non-empty rows.")
        logger.error(err_msg)
        raise ValueError(err_msg)

    n_particles = len(grid[0])
    for i, row in enumerate(grid):
        if len(row) != n_particles:
            err_msg = (
                f"feature_columns_grouped row {i} has {len(row)} entries, "
                f"expected {n_particles} (from row 0)."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    if cfg.feature_names is not None and len(cfg.feature_names) != len(grid):
        err_msg = (
            f"feature_names has {len(cfg.feature_names)} entries, "
            f"but feature_columns_grouped has {len(grid)} rows."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    if cfg.particle_names is not None and len(cfg.particle_names) != n_particles:
        err_msg = (
            f"particle_names has {len(cfg.particle_names)} entries, "
            f"but feature_columns_grouped rows have {n_particles} entries."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    all_columns = [col for row in grid for col in row]
    duplicates = {col for col in all_columns if all_columns.count(col) > 1}
    if duplicates:
        err_msg = (f"Duplicate column(s) in feature_columns_grouped: {duplicates}")
        logger.error(err_msg)
        raise ValueError(err_msg)

    missing_keys = set(cfg.replacements_missing_columns or {})
    sentinel_keys = set(cfg.replacements_sentinel_columns or {})
    both = missing_keys & sentinel_keys
    if both:
        err_msg = (
            f"Column(s) listed in both replacements_missing_columns and "
            f"replacements_sentinel_columns (mutually exclusive): {both}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    unreferenced = (missing_keys | sentinel_keys) - set(all_columns)
    if unreferenced:
        err_msg = (
            f"Column(s) in replacements_missing_columns/replacements_sentinel_columns "
            f"not found anywhere in feature_columns_grouped: {unreferenced}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    if cfg.replacements_sentinel_columns and cfg.general_sentinel_value is None:
        err_msg = (
            "replacements_sentinel_columns is set but general_sentinel_value is null — "
            "there is no sentinel value to detect, so these entries would silently "
            "have no effect. Set general_sentinel_value, or clear replacements_sentinel_columns."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)