"""Local graph and chart artifacts for the MCP operations."""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path

from . import algorithm_to_dict


def write_graph(algorithm, destination: Path) -> str:
    document = algorithm_to_dict(algorithm)
    lines = ["flowchart TD", '    population["Population"]']
    for path_index, pathway in enumerate(document["pathways"], start=1):
        components = [pathway["choose"]]
        for stage in pathway["search"]:
            components.extend(
                (stage.get("choose"), stage["primary"], stage["secondary"])
            )
        components.extend((pathway["update"], *pathway["archive"]))
        previous = "population"
        for index, name in enumerate(filter(None, components), start=1):
            node = f"p{path_index}_{index}"
            label = html.escape(name).replace("|", "&#124;")
            lines.append(f'    {previous} --> {node}["{label}"]')
            previous = node
        lines.append(f"    {previous} --> population")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(destination)


def write_performance_chart(run: Path, destination: Path) -> str:
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.5, 4.2), layout="constrained")
    FigureCanvasSVG(figure)
    axes = figure.subplots()
    if (run / "ConvergenceCurve.csv").is_file():
        with (run / "ConvergenceCurve.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            points = [
                (int(row["Iteration"]), float(row["Performance"]))
                for row in csv.DictReader(stream)
            ]
        if not points:
            raise ValueError("The design run contains no convergence points.")
        steps, scores = zip(*points)
        _plot(axes, steps, scores)
        axes.set(
            title="Design convergence",
            xlabel="Search iteration",
            ylabel="Mean training performance",
        )
    elif (run / "Fitness.csv").is_file():
        with (run / "Fitness.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("The solve run contains no fitness history.")
        for row in rows:
            scores = [
                float(value)
                for key, value in row.items()
                if key and key.startswith("Iteration ") and value
            ]
            _plot(
                axes,
                range(1, len(scores) + 1),
                scores,
                label=str(row["Instance Index"]),
            )
        if len(rows) <= 8:
            axes.legend(title="Instance")
        axes.set(title="Solve history", xlabel="Iteration", ylabel="Fitness")
    else:
        raise ValueError(
            "Provide a run directory containing ConvergenceCurve.csv or Fitness.csv."
        )
    axes.grid(alpha=0.25)
    figure.savefig(destination, format="svg")
    return str(destination)


def _plot(axes, steps, scores, **options):
    axes.plot(
        steps,
        [score if math.isfinite(score) else math.nan for score in scores],
        linewidth=1.6,
        marker=".",
        **options,
    )
