"""Combine evaluation outputs into certification tables and figures."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from evaluation_core import atomic_write_json, strict_json_load, utc_now


def read_optional(name: str) -> dict[str, Any] | None:
    path = config.RESULT_DIR / name
    return strict_json_load(path) if path.is_file() else None


def percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.6f}%"


def save_count_chart(count_data: dict[str, Any], chart_dir: Path) -> str:
    splits = list(count_data["splits"])
    conf = [count_data["splits"][split]["conf_scene_count"] for split in splits]
    complete = [count_data["splits"][split]["complete_scene_count"] for split in splits]
    x = np.arange(len(splits))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - 0.18, conf, width=0.36, label="conf scenes", color="#4C78A8")
    axis.bar(x + 0.18, complete, width=0.36, label="complete scenes", color="#59A14F")
    axis.set_xticks(x, splits)
    axis.set_ylabel("file/scene count")
    axis.set_title("Dataset scene count and required-modality completeness")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    for container in axis.containers:
        axis.bar_label(container, fmt="{:,.0f}", padding=3, fontsize=8)
    figure.tight_layout()
    filename = "data_count_by_split.png"
    figure.savefig(chart_dir / filename, dpi=180)
    plt.close(figure)
    return filename


def save_platform_completeness_chart(
    count_data: dict[str, Any], chart_dir: Path
) -> str:
    splits = list(count_data["splits"])
    paths = sorted(
        {
            f"{platform['env']}/{platform['section']}/{platform['platform']}"
            for split in splits
            for platform in count_data["splits"][split]["platforms"]
        }
    )
    values = np.full((len(paths), len(splits)), np.nan, dtype=float)
    path_index = {path: index for index, path in enumerate(paths)}
    for split_index, split in enumerate(splits):
        for platform in count_data["splits"][split]["platforms"]:
            path = f"{platform['env']}/{platform['section']}/{platform['platform']}"
            values[path_index[path], split_index] = platform["complete_scene_rate"] * 100
    figure, axis = plt.subplots(figsize=(9, max(7, len(paths) * 0.36)))
    image = axis.imshow(values, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    axis.set_xticks(np.arange(len(splits)), splits)
    axis.set_yticks(np.arange(len(paths)), paths, fontsize=8)
    axis.set_title("Required-modality completeness by platform (%)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 25 < value < 85 else "white",
                )
    figure.colorbar(image, ax=axis, label="complete scenes (%)")
    figure.tight_layout()
    filename = "platform_completeness.png"
    figure.savefig(chart_dir / filename, dpi=180)
    plt.close(figure)
    return filename


def save_metric_chart(
    data: dict[str, Any],
    chart_dir: Path,
    filename: str,
    title: str,
    metric_getter,
) -> str:
    splits = list(data["splits"])
    metric_names = sorted(
        {
            name
            for split in splits
            for name in metric_getter(data["splits"][split])
        }
    )
    if not metric_names:
        return ""
    width = 0.8 / max(1, len(splits))
    x = np.arange(len(metric_names))
    figure_width = max(10, len(metric_names) * 0.85)
    figure, axis = plt.subplots(figsize=(figure_width, 6))
    for split_index, split in enumerate(splits):
        metrics = metric_getter(data["splits"][split])
        values = [
            (metrics.get(name, {}).get("accuracy") or 0.0) * 100
            for name in metric_names
        ]
        offset = (split_index - (len(splits) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=split)
    axis.set_xticks(x, metric_names, rotation=40, ha="right")
    axis.set_ylim(0, 101)
    axis.set_ylabel("accuracy (%)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(chart_dir / filename, dpi=180)
    plt.close(figure)
    return filename


def save_manual_chart(manual: dict[str, Any], chart_dir: Path) -> str:
    split_data = manual.get("by_split", {})
    splits = list(split_data)
    values = [
        (split_data[split].get("accuracy") or 0.0) * 100 for split in splits
    ]
    lowers = [
        (split_data[split].get("wilson_95_lower") or 0.0) * 100 for split in splits
    ]
    uppers = [
        (split_data[split].get("wilson_95_upper") or 0.0) * 100 for split in splits
    ]
    errors = np.array(
        [
            [max(0.0, value - lower) for value, lower in zip(values, lowers)],
            [max(0.0, upper - value) for value, upper in zip(values, uppers)],
        ]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(splits, values, color="#F28E2B", yerr=errors, capsize=5)
    axis.set_ylim(0, 101)
    axis.set_ylabel("manual accuracy (%)")
    axis.set_title("Manual instance-segmentation accuracy (Wilson 95% CI)")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f%%", padding=3)
    figure.tight_layout()
    filename = "manual_inst_seg_accuracy.png"
    figure.savefig(chart_dir / filename, dpi=180)
    plt.close(figure)
    return filename


def table_html(headers: list[str], rows: list[list[Any]]) -> str:
    heading = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>"


def build_report() -> dict[str, Any]:
    count_data = read_optional("data_count.json")
    syntax_data = read_optional("syntax_validation.json")
    semantic_data = read_optional("semantic_analysis.json")
    manual_data = read_optional("manual_inst_seg_summary.json")
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    chart_dir = config.RESULT_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    charts: dict[str, str] = {}
    if count_data:
        charts["count"] = save_count_chart(count_data, chart_dir)
        charts["platform_completeness"] = save_platform_completeness_chart(
            count_data, chart_dir
        )
    if syntax_data:
        charts["syntax"] = save_metric_chart(
            syntax_data,
            chart_dir,
            "syntax_accuracy.png",
            "Syntactic accuracy by validation unit",
            lambda split: split.get("units", {}),
        )
    if semantic_data:
        charts["semantic"] = save_metric_chart(
            semantic_data,
            chart_dir,
            "automatic_semantic_accuracy.png",
            "Automatic semantic consistency accuracy",
            lambda split: split.get("metrics", {}),
        )
    if manual_data:
        charts["manual"] = save_manual_chart(manual_data, chart_dir)

    summary: dict[str, Any] = {
        "schema_version": "certification-summary-v1",
        "created_at_utc": utc_now(),
        "result_directory": str(config.RESULT_DIR),
        "available_results": {
            "data_count": count_data is not None,
            "syntax": syntax_data is not None,
            "automatic_semantics": semantic_data is not None,
            "manual_inst_seg": manual_data is not None,
        },
        "splits": {},
        "charts": charts,
    }
    for split in config.DATASETS:
        split_summary: dict[str, Any] = {}
        if count_data:
            source = count_data["splits"][split]
            split_summary["conf_scene_count"] = source["conf_scene_count"]
            split_summary["complete_scene_count"] = source["complete_scene_count"]
            split_summary["completeness"] = source["complete_scene_rate"]
        if syntax_data:
            split_summary["syntactic_accuracy"] = syntax_data["splits"][split]["file_accuracy"]
        if semantic_data:
            split_summary["automatic_semantic_metrics"] = {
                name: metric["accuracy"]
                for name, metric in semantic_data["splits"][split]["metrics"].items()
            }
        if manual_data:
            split_summary["manual_inst_seg"] = manual_data["by_split"].get(split)
        summary["splits"][split] = split_summary
    atomic_write_json(config.RESULT_DIR / "certification_summary.json", summary)

    markdown = [
        "# Dataset 2026 certification evaluation",
        "",
        f"Generated: `{summary['created_at_utc']}`",
        "",
        "The report separates data completeness, syntactic validity, automatic semantic consistency, and human-reviewed instance-segmentation accuracy. No pass/fail threshold is assumed.",
        "",
        "## Split summary",
        "",
        "| Split | Conf scenes | Complete scenes | Completeness | Syntactic accuracy | Manual inst_seg | Manual progress |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    html_rows = []
    for split, data in summary["splits"].items():
        manual = data.get("manual_inst_seg") or {}
        row = [
            split,
            f"{data.get('conf_scene_count', 0):,}" if count_data else "N/A",
            f"{data.get('complete_scene_count', 0):,}" if count_data else "N/A",
            percentage(data.get("completeness")),
            percentage(data.get("syntactic_accuracy")),
            percentage(manual.get("accuracy")),
            f"{manual.get('decided', 0):,}/{manual.get('total', 0):,}" if manual else "N/A",
        ]
        markdown.append("| " + " | ".join(row) + " |")
        html_rows.append(row)

    detail_html: list[str] = []
    if count_data:
        count_headers = ["Split", "Modality", "Count", "Missing", "Orphan"]
        count_rows: list[list[Any]] = []
        markdown.extend(
            [
                "",
                "## Modality counts",
                "",
                "| Split | Modality | Count | Missing | Orphan |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for split, split_data in count_data["splits"].items():
            for modality, count in split_data["modality_counts"].items():
                row = [
                    split,
                    modality,
                    f"{count:,}",
                    f"{split_data['missing_counts'].get(modality, 0):,}",
                    f"{split_data['orphan_counts'].get(modality, 0):,}",
                ]
                count_rows.append(row)
                markdown.append("| " + " | ".join(row) + " |")
        detail_html.extend(
            ["<h2>Modality counts</h2>", table_html(count_headers, count_rows)]
        )

    if syntax_data:
        syntax_headers = [
            "Split",
            "Unit",
            "Checked",
            "Valid",
            "Invalid",
            "Missing",
            "Accuracy",
        ]
        syntax_rows: list[list[Any]] = []
        markdown.extend(
            [
                "",
                "## Syntactic accuracy detail",
                "",
                "| Split | Unit | Checked | Valid | Invalid | Missing | Accuracy |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for split, split_data in syntax_data["splits"].items():
            for unit, metric in split_data["units"].items():
                row = [
                    split,
                    unit,
                    f"{metric['checked']:,}",
                    f"{metric['valid']:,}",
                    f"{metric['invalid']:,}",
                    f"{metric['missing']:,}",
                    percentage(metric["accuracy"]),
                ]
                syntax_rows.append(row)
                markdown.append("| " + " | ".join(row) + " |")
        detail_html.extend(
            [
                "<h2>Syntactic accuracy detail</h2>",
                table_html(syntax_headers, syntax_rows),
            ]
        )

    if semantic_data:
        semantic_headers = [
            "Split",
            "Metric",
            "Unit",
            "Checked",
            "Passed",
            "Failed",
            "Accuracy",
        ]
        semantic_rows: list[list[Any]] = []
        markdown.extend(
            [
                "",
                "## Automatic semantic detail",
                "",
                "| Split | Metric | Unit | Checked | Passed | Failed | Accuracy |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for split, split_data in semantic_data["splits"].items():
            for metric_name, metric in split_data["metrics"].items():
                row = [
                    split,
                    metric_name,
                    metric["unit"],
                    f"{metric['checked']:,}",
                    f"{metric['passed']:,}",
                    f"{metric['failed']:,}",
                    percentage(metric["accuracy"]),
                ]
                semantic_rows.append(row)
                markdown.append("| " + " | ".join(row) + " |")
        detail_html.extend(
            [
                "<h2>Automatic semantic detail</h2>",
                table_html(semantic_headers, semantic_rows),
            ]
        )

    for chart_name, filename in charts.items():
        markdown.extend(["", f"## {chart_name.replace('_', ' ').title()}", "", f"![{chart_name}](charts/{filename})"])

    missing = [name for name, present in summary["available_results"].items() if not present]
    if missing:
        markdown.extend(["", "## Incomplete inputs", "", f"Missing result stages: `{', '.join(missing)}`."])
    (config.RESULT_DIR / "certification_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    chart_html = "".join(
        f"<h2>{html.escape(name.replace('_', ' ').title())}</h2>"
        f"<img src='charts/{html.escape(filename)}' alt='{html.escape(name)}'>"
        for name, filename in charts.items()
        if filename
    )
    missing_html = (
        f"<p class='warning'>Missing result stages: {html.escape(', '.join(missing))}</p>"
        if missing
        else ""
    )
    report_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Dataset 2026 certification evaluation</title>
<style>
body {{ font-family: sans-serif; max-width: 1500px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #bbb; padding: .45rem .6rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #e9eef5; }} img {{ max-width: 100%; border: 1px solid #ddd; }}
.warning {{ background: #fff3cd; padding: .7rem; }}
</style></head><body>
<h1>Dataset 2026 certification evaluation</h1>
<p>Generated: {html.escape(summary['created_at_utc'])}</p>
<p>Completeness, syntactic validity, automatic semantic consistency, and human-reviewed instance segmentation are reported separately.</p>
{missing_html}
<h2>Split summary</h2>
{table_html(['Split', 'Conf scenes', 'Complete scenes', 'Completeness', 'Syntactic accuracy', 'Manual inst_seg', 'Manual progress'], html_rows)}
{''.join(detail_html)}
{chart_html}
</body></html>"""
    (config.RESULT_DIR / "certification_report.html").write_text(
        report_html, encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = build_report()
    print(f"report: {config.RESULT_DIR / 'certification_report.html'}")
    for split, data in summary["splits"].items():
        print(
            f"{split:10s} completeness={percentage(data.get('completeness'))} "
            f"syntax={percentage(data.get('syntactic_accuracy'))}"
        )


if __name__ == "__main__":
    main()
