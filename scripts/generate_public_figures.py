#!/usr/bin/env python3
"""Generate the public, publication-style figures for Do ≠ See.

The script reads only the two sanitized public JSON files.  It never loads
private traces, credentials, or model responses.  SVG is the primary artifact
for papers and GitHub; PNG is emitted for quick previews and slide decks.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, RegularPolygon


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS = ROOT / "results" / "p11b_public_aggregate_v1.json"
PLAN = ROOT / "configs" / "p12_fault_conditioned_public_plan_v1.json"

BG = "#FBFAF7"
INK = "#18212B"
MUTED = "#5B6875"
NAVY = "#17324D"
TEAL = "#1B7F86"
TEAL_LIGHT = "#D9EFF0"
GOLD = "#C48A25"
GOLD_LIGHT = "#F7E8C5"
RED = "#B94A48"
RED_LIGHT = "#F6D8D4"
GREEN = "#2D7A58"
GREEN_LIGHT = "#DDEEE4"
SLATE = "#DDE3E8"
WHITE = "#FFFFFF"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#CCD2D7",
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "savefig.facecolor": BG,
            "figure.facecolor": BG,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.965, title, fontsize=18, fontweight="bold", color=NAVY, va="top")
    fig.text(0.06, 0.925, subtitle, fontsize=10.5, color=MUTED, va="top")


def rounded_box(ax, x, y, w, h, text, *, fc, ec, text_color=INK, fontsize=10, lw=1.2,
                radius=0.04, weight="normal", zorder=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.25,
        zorder=zorder + 1,
    )
    return patch


def arrow(ax, start, end, *, color=MUTED, lw=1.7, style="-|>", ls="-", label=None,
          label_xy=None, zorder=2, connectionstyle="arc3"):
    arr = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(arr)
    if label:
        x, y = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=color, bbox={"fc": BG, "ec": "none", "pad": 1.5}, zorder=zorder + 1)
    return arr


def figure_1() -> None:
    fig = plt.figure(figsize=(13.4, 7.5))
    add_title(fig, "Do ≠ See", "A local success signal can be visible while the protected target has already become false")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main causal spine.
    rounded_box(ax, 0.02, 0.57, 0.16, 0.18, "A\naction", fc=TEAL_LIGHT, ec=TEAL, fontsize=12, weight="bold")
    rounded_box(ax, 0.25, 0.57, 0.18, 0.18, "R\nreceipt", fc=GOLD_LIGHT, ec=GOLD, fontsize=12, weight="bold")
    rounded_box(ax, 0.50, 0.67, 0.20, 0.16, "S₊\nprotected target true", fc=GREEN_LIGHT, ec=GREEN, fontsize=10.5, weight="bold")
    rounded_box(ax, 0.50, 0.40, 0.20, 0.16, "S₋\nprotected target false", fc=RED_LIGHT, ec=RED, fontsize=10.5, weight="bold")
    rounded_box(ax, 0.78, 0.57, 0.18, 0.18, "D\nauthorize / hold", fc="#E7EAF0", ec=NAVY, fontsize=11, weight="bold")

    arrow(ax, (0.18, 0.66), (0.25, 0.66), color=TEAL, label="performed")
    arrow(ax, (0.43, 0.68), (0.50, 0.75), color=GREEN, label="consistent state", label_xy=(0.47, 0.78))
    arrow(ax, (0.43, 0.63), (0.50, 0.48), color=RED, lw=2.2, label="partial fault", label_xy=(0.47, 0.53))
    arrow(ax, (0.70, 0.75), (0.78, 0.68), color=GREEN)
    arrow(ax, (0.70, 0.48), (0.78, 0.64), color=RED, lw=2.2, ls="--", label="dangerous path", label_xy=(0.76, 0.47))

    # Evidence policies underneath.
    ax.text(0.02, 0.31, "What the agent is allowed to see", transform=ax.transAxes,
            fontsize=10, color=NAVY, fontweight="bold")
    rounded_box(ax, 0.04, 0.16, 0.22, 0.095, "L0  value-only\nproxy visible", fc="#F1F3F5", ec="#9AA7B3", fontsize=9)
    rounded_box(ax, 0.31, 0.16, 0.22, 0.095, "L2  causal lineage\nsource of receipt", fc=TEAL_LIGHT, ec=TEAL, fontsize=9)
    rounded_box(ax, 0.58, 0.16, 0.22, 0.095, "L4  target binding\nprotected invariant", fc=GREEN_LIGHT, ec=GREEN, fontsize=9)
    rounded_box(ax, 0.84, 0.16, 0.13, 0.095, "verified\nprobe", fc=GOLD_LIGHT, ec=GOLD, fontsize=9)
    ax.text(0.02, 0.065, "P12 tests the counterfactual: hold the action receipt fixed, change only the post-receipt state, then compare authorization.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.text(0.02, 0.015, "Conceptual model; no behavioral result is implied by this schematic.",
            transform=ax.transAxes, fontsize=8.3, color=MUTED, style="italic")
    save(fig, "fig1_do_not_equal_see")


def figure_2(aggregate: dict) -> None:
    fig = plt.figure(figsize=(13.4, 7.2))
    add_title(fig, "P11-B ordinary-prefix census", "Every prefix was active; the natural proxy–target decoupling event never appeared")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.95, bottom=0.13, top=0.84, width_ratios=[1.05, 1.15], wspace=0.27)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    c = aggregate["census"]
    f = aggregate["opportunity_funnel"]
    stages = ["trigger audits", "eligible mutations", "proxy-positive", "proxy-positive\n+ target-false"]
    vals = [f["trigger_audits"], f["eligible_mutations"], f["proxy_positive_target_true"], 0]
    colors = [NAVY, TEAL, GREEN, RED]
    y = list(range(len(stages)))[::-1]
    ax1.set_xlim(0, max(vals[:-1]) * 1.16)
    for yi, label, val, col in zip(y, stages, vals, colors):
        if val:
            ax1.barh(yi, val, color=col, height=0.48, alpha=0.95)
            ax1.text(val + max(vals[:-1]) * 0.025, yi, f"{val:,}", va="center", fontsize=11, color=INK, fontweight="bold")
        else:
            ax1.plot([0, max(vals[:-1]) * 0.025], [yi, yi], color=RED, linewidth=4, solid_capstyle="round")
            ax1.text(max(vals[:-1]) * 0.04, yi, "0", va="center", fontsize=11, color=RED, fontweight="bold")
    ax1.set_yticks(y, stages)
    ax1.set_xlabel("count")
    ax1.set_title("From observed activity to the tested event", loc="left", color=NAVY, fontweight="bold", pad=12)
    ax1.grid(axis="x", color="#E6E9EC", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.spines["bottom"].set_color("#CCD2D7")
    ax1.tick_params(axis="y", length=0)
    ax1.text(0.0, -0.24, f"{c['prefixes']} prefixes · {c['models']} models · {c['task_families']} task families",
             transform=ax1.transAxes, color=MUTED, fontsize=9.5)

    # 2x2 quadrant matrix: rows target, columns proxy.
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 2)
    ax2.set_aspect("equal")
    ax2.set_xticks([0.5, 1.5], ["proxy −", "proxy +"])
    ax2.set_yticks([0.5, 1.5], ["target −", "target +"])
    ax2.tick_params(length=0, labelsize=10)
    ax2.set_title("Proxy × protected-target quadrants", loc="left", color=NAVY, fontweight="bold", pad=12)
    cells = [
        (0, 1, f["proxy_negative_target_false"], "#EEF1F3", MUTED),
        (1, 1, f["proxy_positive_target_false"], RED_LIGHT, RED),
        (0, 0, f["proxy_negative_target_true"], "#EEF1F3", MUTED),
        (1, 0, f["proxy_positive_target_true"], GREEN_LIGHT, GREEN),
    ]
    for x, y0, val, fc, ec in cells:
        ax2.add_patch(Rectangle((x, y0), 1, 1, facecolor=fc, edgecolor="#FFFFFF", linewidth=3))
        ax2.text(x + 0.5, y0 + 0.58, f"{val:,}", ha="center", va="center", fontsize=23,
                 fontweight="bold", color=ec)
        caption = "tested dangerous cell" if (x, y0) == (1, 1) else ""
        if caption:
            ax2.text(x + 0.5, y0 + 0.23, caption, ha="center", va="center", fontsize=8.5, color=RED)
    ax2.add_patch(Rectangle((1, 1), 1, 1, fill=False, edgecolor=RED, linewidth=2.2))
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.text(0.0, -0.18, "No proxy-positive action had a false protected target in ordinary prefixes.",
             transform=ax2.transAxes, color=RED, fontsize=9.5, fontweight="bold")
    ax2.text(0.0, -0.27, "This closes the natural-opportunity route; it does not estimate false approval or agent safety.",
             transform=ax2.transAxes, color=MUTED, fontsize=9)
    save(fig, "fig2_p11b_opportunity_funnel")


def figure_3(plan: dict) -> None:
    fig = plt.figure(figsize=(13.4, 7.8))
    add_title(fig, "P12 fault-conditioned authorization design", "A matched counterfactual keeps the receipt fixed and changes the protected state after the action")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.79])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Timeline.
    steps = [
        (0.04, "1", "action\nreceipt", TEAL_LIGHT, TEAL),
        (0.26, "2", "trusted\npartial fault", RED_LIGHT, RED),
        (0.49, "3", "evidence\ncondition", GOLD_LIGHT, GOLD),
        (0.72, "4", "decision\nsubmit / hold", GREEN_LIGHT, GREEN),
    ]
    for x, num, label, fc, ec in steps:
        ax.add_patch(plt.Circle((x + 0.06, 0.72), 0.035, transform=ax.transAxes, facecolor=fc,
                                edgecolor=ec, linewidth=1.6, zorder=3))
        ax.text(x + 0.06, 0.72, num, transform=ax.transAxes, ha="center", va="center", fontsize=11,
                color=ec, fontweight="bold", zorder=4)
        ax.text(x + 0.06, 0.63, label, transform=ax.transAxes, ha="center", va="top", fontsize=10,
                color=INK, fontweight="bold", linespacing=1.2)
    for x1, x2 in [(0.14, 0.26), (0.36, 0.49), (0.59, 0.72)]:
        arrow(ax, (x1, 0.72), (x2, 0.72), color=MUTED, lw=1.8)
    ax.text(0.84, 0.77, "same prefix\n(counterfactual pair)", transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color=NAVY, bbox={"boxstyle": "round,pad=0.35", "fc": "#EEF1F8", "ec": "#B6C3D5"})

    # Fault mechanisms.
    ax.text(0.04, 0.49, "Trusted post-receipt fault families", transform=ax.transAxes, fontsize=10, color=NAVY, fontweight="bold")
    faults = [
        "release replica omits manifest",
        "migration partially rolls back entitlement",
        "rollout leaves a stale edge replica",
        "backup snapshot omits metadata",
    ]
    for i, fault in enumerate(faults):
        y = 0.40 - i * 0.065
        ax.add_patch(Rectangle((0.045, y), 0.018, 0.026, transform=ax.transAxes, facecolor=RED_LIGHT, edgecolor=RED, linewidth=1))
        ax.text(0.075, y + 0.013, fault, transform=ax.transAxes, va="center", fontsize=9.2, color=INK)

    # Six arms, with explicit pairs.
    ax.text(0.46, 0.49, "Six matched evidence arms", transform=ax.transAxes, fontsize=10, color=NAVY, fontweight="bold")
    arms = [
        ("L0", "echo", "value-only", "#F1F3F5", "#9AA7B3"),
        ("L0", "erased", "value-only", "#F1F3F5", "#9AA7B3"),
        ("L2", "echo", "causal lineage", TEAL_LIGHT, TEAL),
        ("L2", "erased", "causal lineage", TEAL_LIGHT, TEAL),
        ("L4", "echo", "target binding", GREEN_LIGHT, GREEN),
        ("L4", "verified", "target binding", GOLD_LIGHT, GOLD),
    ]
    for i, (level, arm, desc, fc, ec) in enumerate(arms):
        x = 0.46 + (i % 2) * 0.23
        y = 0.40 - (i // 2) * 0.09
        rounded_box(ax, x, y, 0.19, 0.055, f"{level} · {arm}\n{desc}", fc=fc, ec=ec, fontsize=8.2, lw=1.0)

    # Scale and boundary.
    ax.add_patch(FancyBboxPatch((0.04, 0.045), 0.91, 0.105, transform=ax.transAxes,
                                boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#F1F3F5",
                                edgecolor="#C8D0D8", linewidth=1.0))
    pilot = plan["pilot"]
    conf = plan["confirmatory_census"]
    scale = (f"Pilot: {pilot['prefixes']} prefixes · {pilot['model']} · seed {pilot['provider_seed']} · excluded from confirmation     "
             f"Confirmatory: {conf['prefixes']} prefixes · {len(conf['models'])} models × {conf['task_families']} task families × {len(conf['provider_seeds'])} seeds")
    ax.text(0.06, 0.106, scale, transform=ax.transAxes, fontsize=8.9, color=INK, va="center")
    ax.text(0.06, 0.071, "Primary estimand: echo_L2 submit − erased_L2 submit, conditional on a trusted post-receipt fault.",
            transform=ax.transAxes, fontsize=9.1, color=NAVY, fontweight="bold")
    ax.text(0.06, 0.033, "Design-frozen and provider-free at release; no live P12 behavioral result is claimed.",
            transform=ax.transAxes, fontsize=8.7, color=MUTED, style="italic")
    save(fig, "fig3_p12_fault_conditioned_design")


def drawio_file(name: str, cells: list[str], width: int = 1400, height: int = 800) -> None:
    """Write an editable draw.io source for the two conceptual diagrams."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>\n<mxfile host="drawio" version="26.0.0"><diagram name="Page-1"><mxGraphModel page="1" pageWidth="%d" pageHeight="%d"><root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram></mxfile>\n""" % (width, height, "".join(cells))
    (FIG_DIR / name).write_text(xml, encoding="utf-8")


def box_cell(cid, value, x, y, w, h, fill, stroke, parent="1"):
    return (f'<mxCell id="{cid}" value="{escape(value)}" style="rounded=1;whiteSpace=wrap;html=1;'
            f'fillColor={fill};strokeColor={stroke};fontFamily=Arial;fontSize=16;fontColor={INK};" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge_cell(cid, source, target, color=MUTED, dashed=False, label=""):
    dash = "dashed=1;" if dashed else ""
    return (f'<mxCell id="{cid}" value="{escape(label)}" style="edgeStyle=orthogonalEdgeStyle;rounded=1;'
            f'orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};endArrow=block;{dash}" edge="1" parent="1" source="{source}" target="{target}">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>')


def write_drawio_sources() -> None:
    fig1 = [
        box_cell("a", "A — action", 70, 260, 190, 90, TEAL_LIGHT, TEAL),
        box_cell("r", "R — local success receipt", 360, 260, 230, 90, GOLD_LIGHT, GOLD),
        box_cell("sp", "S+ — target true", 720, 160, 220, 90, GREEN_LIGHT, GREEN),
        box_cell("sm", "S− — target false", 720, 360, 220, 90, RED_LIGHT, RED),
        box_cell("d", "D — authorize / hold", 1050, 260, 230, 90, "#E7EAF0", NAVY),
        edge_cell("e1", "a", "r", TEAL),
        edge_cell("e2", "r", "sp", GREEN, label="consistent state"),
        edge_cell("e3", "r", "sm", RED, label="partial fault"),
        edge_cell("e4", "sp", "d", GREEN),
        edge_cell("e5", "sm", "d", RED, dashed=True, label="dangerous path"),
        box_cell("l0", "L0 — value-only", 130, 560, 220, 75, "#F1F3F5", "#9AA7B3"),
        box_cell("l2", "L2 — causal lineage", 440, 560, 220, 75, TEAL_LIGHT, TEAL),
        box_cell("l4", "L4 — target binding", 750, 560, 220, 75, GREEN_LIGHT, GREEN),
    ]
    drawio_file("fig1_do_not_equal_see.drawio", fig1)
    fig3 = [
        box_cell("s1", "1. action receipt", 80, 120, 220, 90, TEAL_LIGHT, TEAL),
        box_cell("s2", "2. trusted partial fault", 380, 120, 240, 90, RED_LIGHT, RED),
        box_cell("s3", "3. evidence condition", 700, 120, 230, 90, GOLD_LIGHT, GOLD),
        box_cell("s4", "4. submit / hold", 1010, 120, 220, 90, GREEN_LIGHT, GREEN),
        edge_cell("p1", "s1", "s2", MUTED), edge_cell("p2", "s2", "s3", MUTED), edge_cell("p3", "s3", "s4", MUTED),
        box_cell("a1", "L0 echo", 380, 330, 200, 70, "#F1F3F5", "#9AA7B3"),
        box_cell("a2", "L0 erased", 610, 330, 200, 70, "#F1F3F5", "#9AA7B3"),
        box_cell("a3", "L2 echo", 380, 450, 200, 70, TEAL_LIGHT, TEAL),
        box_cell("a4", "L2 erased", 610, 450, 200, 70, TEAL_LIGHT, TEAL),
        box_cell("a5", "L4 echo", 380, 570, 200, 70, GREEN_LIGHT, GREEN),
        box_cell("a6", "L4 verified", 610, 570, 200, 70, GOLD_LIGHT, GOLD),
    ]
    drawio_file("fig3_p12_fault_conditioned_design.drawio", fig3)


def manifest() -> None:
    files = {}
    for path in sorted(FIG_DIR.glob("fig*.svg")):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "format": "dosee.public-figure-manifest.v1",
        "source_files": [str(RESULTS.relative_to(ROOT)), str(PLAN.relative_to(ROOT))],
        "figures": files,
        "generator": "scripts/generate_public_figures.py",
        "claim_boundary": "Figures use public aggregate/design metadata only; no raw behavioral content is rendered.",
    }
    (FIG_DIR / "figure_manifest_v1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    style()
    aggregate = json.loads(RESULTS.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    figure_1()
    figure_2(aggregate)
    figure_3(plan)
    write_drawio_sources()
    manifest()
    print("Generated public figures in", FIG_DIR)


if __name__ == "__main__":
    main()
