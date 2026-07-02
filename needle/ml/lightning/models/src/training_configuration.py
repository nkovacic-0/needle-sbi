import torch
import torch.nn as nn

logger = ColorFormatter.get_logger("ml")

def custom_configure_optimizers(
    parameters,
    optimizer_configs: dict,
) -> dict:
    # figure out what optimizer the user wants, default to 'adam'
    optimizer_name = optimizer_configs.get("optimizer", "adam").lower()
    # extract learning rate, no default is given as it heavily depends on the dataset/setup
    lr = optimizer_configs["lr"]
    # set up the optimizer
    optimizer_map = {
        "adam":  (torch.optim.Adam,  {"weight_decay": 0.0}),
        "adamw": (torch.optim.AdamW, {"weight_decay": 1e-2}),
        "sgd":   (torch.optim.SGD,   {"weight_decay": 0.0, "momentum": 0.9}),
    }
    if optimizer_name not in optimizer_map:
        err_msg = (
            f"Unsupported optimizer '{optimizer_name}'. "
            f"Expected one of: {', '.join(sorted(optimizer_map))}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    optimizer_cls, optimizer_defaults = optimizer_map[optimizer_name]
    optimizer_kwargs = {
        **optimizer_defaults,
        **{
            k: v
            for k, v in optimizer_configs.items()
            if k not in ("optimizer", "lr", "scheduler")
        },
    }
    optimizer = optimizer_cls(parameters, lr=lr, **optimizer_kwargs)
    # set up the LR scheduler
    scheduler_config = optimizer_configs.get("scheduler", {})
    scheduler_name = scheduler_config.get("scheduler", "reduce_on_plateau").lower()
    scheduler_map = {
        "reduce_on_plateau": (
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            {"mode": "min", "factor": 0.5, "patience": 10},
            True,
        ),
        "cosine_annealing": (
            torch.optim.lr_scheduler.CosineAnnealingLR,
            {"T_max": 50, "eta_min": 1e-6},
            False,
        ),
        "cosine_annealing_warm_restarts": (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            {"T_0": 10, "T_mult": 2, "eta_min": 1e-6},
            False,
        ),
        "step": (
            torch.optim.lr_scheduler.StepLR,
            {"step_size": 10, "gamma": 0.5},
            False,
        ),
        "exponential": (
            torch.optim.lr_scheduler.ExponentialLR,
            {"gamma": 0.95},
            False,
        ),
        "none": (None, {}, False),
    }
    if scheduler_name not in scheduler_map:
        err_msg = (
            f"Unsupported scheduler '{scheduler_name}'. "
            f"Expected one of: {', '.join(sorted(scheduler_map))}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    scheduler_cls, scheduler_defaults, needs_monitor = scheduler_map[scheduler_name]

    if scheduler_cls is None:
        return {"optimizer": optimizer}

    scheduler_kwargs = {
        **scheduler_defaults,
        **{
            k: v
            for k, v in scheduler_config.items()
            if k != "scheduler"
        },
    }
    scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
    lr_scheduler_config = {
        "scheduler": scheduler,
        "interval": scheduler_config.get("interval", "epoch"),
        "frequency": scheduler_config.get("frequency", 1),
    }

    if needs_monitor:
        lr_scheduler_config["monitor"] = scheduler_config.get(
            "monitor", "val_loss"
        )
    return {
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler_config,
    }