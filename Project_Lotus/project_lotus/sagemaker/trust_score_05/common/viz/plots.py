"""Saving a figure together with the data behind it.

Every plot this pipeline writes goes out three times: a PNG for reading, an SVG
for putting in a document, and a CSV of the numbers that were plotted. The CSV is
the important one. A plot in an S3 prefix six months old is unarguable and
uncheckable; the same plot with its ``plot_data`` sibling can be re-drawn,
re-scaled, or shown to be wrong. A ``.plot_config.json`` alongside records the
parameters — which model, which scenario, which hyperparameters — so the numbers
can be attributed without parsing the prefix.

The one design decision worth stating: a plot that fails to save raises by
default. The notebook's version swallowed every exception, wrote a
``.plot_failed.json`` marker, and continued, with strictness controlled by a
``PLOT_STRICT_MODE`` environment variable that defaulted to off. That is the
wrong default for two reasons. The failures it hides are mostly not about
plotting — an ``ObjectStoreError`` from an expired credential fails every
subsequent write in the run too, and finding out at the end from a scatter of
marker files rather than at the first attempt wastes the whole run. And a
sweep that reports success while having written none of its diagnostics has
produced a result nobody can check.

So the flag survives, inverted: :func:`save_plot` raises, and a caller who
genuinely wants a best-effort plot passes ``strict=False`` and gets the marker
file. Which callers those are is a decision at the call site, not an environment
variable set on the cluster.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import traceback
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

from trust_score_05.common.io.s3 import join_uri, upload_file, write_json, write_pandas

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "PLOT_FORMATS",
    "PlotError",
    "configure_matplotlib",
    "safe_plot_name",
    "save_plot",
]

LOGGER = logging.getLogger(__name__)

#: Written for every plot. PNG at 160 dpi is legible in a browser and small
#: enough to open over a slow link; SVG is what survives being pasted into a
#: document and resized.
PLOT_FORMATS = ("png", "svg")

# Anything that is not a word character, a dash, or a path separator. Plot names
# are used as object keys and as local filenames, and a name arriving with a
# scenario string in it can contain a colon or a space.
_UNSAFE_NAME_CHARS = re.compile(r"[^0-9A-Za-z_\-/.]+")


class PlotError(RuntimeError):
    """Raised when a figure could not be written."""


def configure_matplotlib() -> Any:
    """Import matplotlib with a headless backend and return the ``pyplot`` module.

    ``Agg`` is selected before ``pyplot`` is imported, which is the only point at
    which the choice takes effect. Without it, matplotlib picks a backend by
    probing for a display; on a SageMaker kernel that probe succeeds and every
    ``figure()`` call then holds a window handle that is never closed, and the
    run dies several hundred figures in with "Fail to allocate bitmap".
    """
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def safe_plot_name(name: str) -> str:
    """Normalise a plot name into something safe as a key and as a filename.

    Slashes survive, because they are how plots are foldered
    (``evaluation/<scenario>/score_distribution``). ``..`` does not, because a
    name assembled from a scenario identifier should not be able to write outside
    its own prefix.

    The trailing separators are stripped *after* substitution rather than before,
    and the "normalises to nothing" check asks for a letter or a digit rather than
    for a non-empty string. Both matter because the substitution manufactures
    separators: ``"roc (test)"`` becomes ``"roc_test_"`` and a name that is
    entirely punctuation or entirely whitespace becomes ``"_"``, which is not
    empty and so passed the check. The figure then landed at ``plots/_.png`` —
    invisible in a console listing and impossible to attribute, which is the
    exact outcome the check exists to prevent.
    """
    cleaned = _UNSAFE_NAME_CHARS.sub("_", str(name))
    cleaned = cleaned.replace("..", "__")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    cleaned = cleaned.strip("_/")
    if not re.search(r"[0-9A-Za-z]", cleaned):
        raise ValueError(f"plot name {name!r} normalises to nothing")
    return cleaned


def save_plot(
    figure: Any,
    prefix: str,
    name: str,
    plot_data: Optional["pd.DataFrame"] = None,
    plot_config: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
    formats: Sequence[str] = PLOT_FORMATS,
    dpi: int = 160,
) -> Iterable[str]:
    """Write ``figure`` under ``prefix`` as ``plots/<name>.{png,svg}``.

    Args:
        figure: A matplotlib ``Figure``. Closed on the way out whether or not the
            save succeeded — an unclosed figure is a leak, and a sweep produces
            thousands.
        prefix: The run prefix. ``plots/`` and ``plot_data/`` are created under it.
        name: A slash-separated logical name, e.g.
            ``evaluation/2024-11/score_distribution``.
        plot_data: The numbers behind the figure, written to
            ``plot_data/<name>.csv``. Optional only because a few figures are
            purely schematic; anything showing data should pass it.
        plot_config: Parameters to record in ``plot_data/<name>.plot_config.json``.
        strict: Raise on failure. When ``False``, a
            ``plot_data/<name>.plot_failed.json`` marker is written instead and
            the exception is logged. See the module docstring for why ``True`` is
            the default.
        formats: Image formats to write.
        dpi: Raster resolution, applied to PNG.

    Returns:
        The URIs written.
    """
    plt = configure_matplotlib()
    safe_name = safe_plot_name(name)
    scratch = tempfile.mkdtemp(prefix="ts05_plot_")
    written: list[str] = []

    try:
        try:
            figure.tight_layout()
        except Exception as exc:  # noqa: BLE001 - a layout warning is not a failure
            # `tight_layout` raises on figures with incompatible axes — a colorbar
            # on a constrained layout, most often. The figure is still perfectly
            # saveable, so this is the one exception here that is genuinely
            # non-fatal.
            LOGGER.debug("tight_layout failed for %s: %s", safe_name, exc)

        for image_format in formats:
            local = os.path.join(scratch, f"figure.{image_format}")
            save_kwargs = {"bbox_inches": "tight"}
            if image_format == "png":
                save_kwargs["dpi"] = dpi
            figure.savefig(local, **save_kwargs)
            target = join_uri(prefix, "plots", f"{safe_name}.{image_format}")
            upload_file(local, target)
            written.append(target)

        if plot_data is not None:
            target = join_uri(prefix, "plot_data", f"{safe_name}.csv")
            write_pandas(plot_data, target)
            written.append(target)

        if plot_config is not None:
            target = join_uri(prefix, "plot_data", f"{safe_name}.plot_config.json")
            write_json(target, dict(plot_config))
            written.append(target)

    except Exception as exc:  # noqa: BLE001 - re-raised below unless asked not to
        if strict:
            raise PlotError(f"could not save plot {safe_name!r} under {prefix}: {exc}") from exc
        LOGGER.exception("plot %s failed", safe_name)
        try:
            write_json(
                join_uri(prefix, "plot_data", f"{safe_name}.plot_failed.json"),
                {
                    "plot_name": safe_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception:  # noqa: BLE001 - the store is evidently unavailable
            LOGGER.warning("could not write the failure marker for plot %s either", safe_name)
    finally:
        plt.close(figure)
        shutil.rmtree(scratch, ignore_errors=True)

    return written
