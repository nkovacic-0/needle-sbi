import os
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import mplhep
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("plotting")


class PlottingWrapper:
    """Standalone base for plot-producing classes, the procedure is: subclass,
    decorate methods with @PlottingWrapper.plot(...), call generate_all_plots()
    once instance state is set.

    Each @plot method returns list[tuple[str, Figure]], this way filename (no
    extension) is paired with the figure, saved AS-IS to
    plot_save_dir/{filename}.{fmt}. One entry for a simple plot, N for
    anything parametrized (one per bin-count/quantile/feature/... variant)

    @plot itself does no wrapping and is purely for attribute tagging. All
    calling/validation/styling/saving happens centrally in
    generate_all_plots(), which:
      1. calls every @plot method once, each independently try/except-wrapped
         (one method's failure is logged and skipped; its siblings still run);
      2. checks the COMBINED filename set across every method for collisions,
         BEFORE writing anything. A collision is an authoring bug (two
         methods deriving the same filename), so this hard-fails immediately;
      3. styles (if requested) and saves every figure, once confirmed clean.
    """

    def __init__(
        self,
        plot_save_dir: str,
        rlabel: str = "FAIR Universe HiggsML",
        plotting_configs: dict | None = None,
    ) -> None:
        self.plot_save_dir = plot_save_dir
        self.rlabel = rlabel
        self.plotting_configs = dict(plotting_configs) if plotting_configs else {}
        os.makedirs(os.path.abspath(self.plot_save_dir), exist_ok=True)

    class PlottingSettings:
        formats: List[str] = ["pdf", "png"]

    def _resolve_plot_filename(self, plot_name: str) -> str:
        """Convenience helper a subclass may call while building its own
        returned filenames -- applies plot_savefile_prefix/suffix from
        plotting_configs. NOT invoked automatically; opt-in per plot method,
        not a framework-enforced naming rule.
        """
        prefix = self.plotting_configs.get("plot_savefile_prefix", "")
        suffix = self.plotting_configs.get("plot_savefile_suffix", "")
        return f"{prefix}{plot_name}{suffix}"

    def _build_title(self, distinguishing_label: str) -> str:
        prefix = self.plotting_configs.get("plot_title_prefix", "")
        suffix = self.plotting_configs.get("plot_title_suffix", "")
        base_title = self.plotting_configs.get("title", "")
        return f"{prefix}{base_title} \u2014 {distinguishing_label}{suffix}"

    def set_needle_plot_style(self, fig: Figure, axes: Axes | List[Axes] | None = None) -> Figure:
        if axes is None:
            axes = fig.axes
        if not isinstance(axes, List):
            axes = [axes]
        for ax in axes:
            mplhep.label.exp_label(loc=0, exp="NEEDLE", ax=ax, rlabel=self.rlabel)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot(*, name: str | None = None, add_needle_plot_style: bool = True) -> Callable[[Callable], Callable]:
        """Marks a method as plot-producing, for generate_all_plots() to
        discover and call. Purely for tagging it returns the method unmodified.

        The decorated method must return list[tuple[str, Figure]] 
        See class docstring for the filename/saving contract.

        Args:
            name (str, optional): bookkeeping label for this method, it is used
                in logs and the duplicate-registration check. Defaults to the
                method's own name. Has NO effect on saved filenames (those
                come entirely from the method's own return value).
            add_needle_plot_style (bool): whether generate_all_plots() should
                apply NEEDLE branding to every figure this method returns.
        """

        def decorator(func: Callable) -> Callable:
            func._plot_name = name or func.__name__
            func._add_needle_plot_style = add_needle_plot_style
            return func

        return decorator

    @property
    def _registered_plot_functions(self) -> Dict[str, Callable]:
        """bookkeeping_name -> bound method, one per @plot-decorated method.
        Walks the full MRO so a subclass correctly overriding a @plot method
        is found via normal Python attribute-lookup semantics. Seen_attr_names
        prevents a correctly-overridden method from tripping a false
        "duplicate" error against itself.
        """
        functions: Dict[str, Callable] = {}
        seen_attr_names = set()

        for klass in self.__class__.__mro__:
            for attr_name in klass.__dict__:
                if attr_name in seen_attr_names:
                    continue
                seen_attr_names.add(attr_name)

                resolved = getattr(self.__class__, attr_name, None)
                if not hasattr(resolved, "_plot_name"):
                    continue

                bookkeeping_name = resolved._plot_name
                if bookkeeping_name in functions:
                    err_msg = f"Duplicate plot names: '{bookkeeping_name}' registered by more than one method"
                    logger.error(err_msg)
                    raise ValueError(err_msg)

                functions[bookkeeping_name] = getattr(self, attr_name)

        return functions

    def generate_all_plots(self) -> Dict[str, Figure]:
        """See class docstring for the full three-step contract.

        Returns:
            Dict[str, Figure]: filename -> Figure, for every plot successfully
                produced and saved. Each Figure is already plt.close()'d so
                the return value is informational, not a set of open figures.
        """
        all_results: list[tuple[str, Figure, bool, str]] = []  # (filename, fig, add_style, bookkeeping_name)

        for bookkeeping_name, bound_method in self._registered_plot_functions.items():
            try:
                result = bound_method()
            except Exception:
                logger.exception(
                    f"[PlottingWrapper] '{bookkeeping_name}' failed for class '{self.__class__.__name__}'; skipping."
                )
                continue

            if not isinstance(result, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in result):
                logger.error(
                    f"[PlottingWrapper] '{bookkeeping_name}' must return list[tuple[str, Figure]], "
                    f"got {type(result)}; skipping."
                )
                continue

            add_style = getattr(bound_method, "_add_needle_plot_style", True)
            for filename, fig in result:
                if not isinstance(fig, Figure):
                    logger.error(
                        f"[PlottingWrapper] '{bookkeeping_name}' returned a non-Figure value for "
                        f"filename '{filename}'; skipping that entry."
                    )
                    continue
                all_results.append((filename, fig, add_style, bookkeeping_name))

        filenames_seen: dict[str, str] = {}
        for filename, _, _, bookkeeping_name in all_results:
            if filename in filenames_seen and filenames_seen[filename] != bookkeeping_name:
                err_msg = (
                    f"[PlottingWrapper] Filename collision: '{filename}' produced by both "
                    f"'{filenames_seen[filename]}' and '{bookkeeping_name}'."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            filenames_seen[filename] = bookkeeping_name

        figures: Dict[str, Figure] = {}
        for filename, fig, add_style, _ in all_results:
            fig = self.set_needle_plot_style(fig) if add_style else fig
            for fmt in self.PlottingSettings.formats:
                save_path = os.path.join(self.plot_save_dir, f"{filename}.{fmt}")
                fig.savefig(save_path)
                logger.debug(f"[PlottingWrapper] Saved plot '{filename}' to '{save_path}'.")
            plt.close(fig)
            figures[filename] = fig

        logger.info(f"[PlottingWrapper] '{self.__class__.__name__}': saved {len(figures)} plot(s).")
        return figures