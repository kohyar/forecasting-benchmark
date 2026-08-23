"""The paper's figures."""
import matplotlib.pyplot as plt
import numpy as np

from tsbench.report.style import (
    FAMILIES,
    GRID,
    INK,
    INK_MUTED,
    apply_paper_style,
    family_style,
)


def pareto_frontier(cost, error) -> np.ndarray:
    """Mask of the non-dominated points, minimising both axes.

    A point drops out only when another is at least as good on both and
    strictly better on one, so ties on an axis keep the cheaper point.
    """
    cost = np.asarray(cost, dtype=float)
    error = np.asarray(error, dtype=float)

    dominated = ((cost[:, None] <= cost[None, :])
                 & (error[:, None] <= error[None, :])
                 & ((cost[:, None] < cost[None, :])
                    | (error[:, None] < error[None, :]))).any(axis=0)
    return ~dominated


def _family_handles(families, patch=False):
    handles = []
    for family in FAMILIES:
        if family not in families:
            continue
        style = family_style(family)
        if patch:
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=style["color"],
                                         alpha=0.75, edgecolor="white", label=family))
        else:
            handles.append(plt.Line2D([], [], linestyle="none", label=family,
                                      marker=style["marker"], markersize=6,
                                      markerfacecolor=style["color"],
                                      markeredgecolor="white", markeredgewidth=0.5))
    return handles


def _place_labels(ax, names, x, y, show):
    """Offset each label to whichever corner sits furthest from other markers.

    With seventeen models several sit close enough that a fixed offset drops a
    label straight on top of a neighbour.
    """
    ax.figure.canvas.draw()
    points = ax.transData.transform(np.column_stack([x, y]))
    left_edge = ax.transAxes.transform((0, 0))[0]
    right_edge = ax.transAxes.transform((1, 0))[0]

    for index, name in enumerate(names):
        if not show[index]:
            continue
        # a label pushed past either edge lands on the axis furniture
        outward = [(9, 4), (9, -12)]
        inward = [(-9, 4), (-9, -12)]
        if points[index][0] > right_edge - 70:
            candidates = inward
        elif points[index][0] < left_edge + 70:
            candidates = outward
        else:
            candidates = outward + inward
        best, best_clearance = candidates[0], -np.inf
        for dx, dy in candidates:
            anchor = points[index] + np.array([dx * 3, dy * 1.5])
            distances = np.hypot(*(points - anchor).T)
            distances[index] = np.inf
            if distances.min() > best_clearance:
                best_clearance, best = distances.min(), (dx, dy)

        ax.annotate(name, (x[index], y[index]), textcoords="offset points",
                    xytext=best, ha="left" if best[0] > 0 else "right",
                    fontsize=7.5, color=INK, zorder=4)


def accuracy_vs_cost(cost, accuracy, horizon: int, highlight=(),
                     accuracy_column=None, accuracy_label="median MASE"):
    """F1: where a model sits on the accuracy/compute trade-off.

    Cost is seconds to run the model once over 1k series, on a log axis because
    the roster spans three and a half orders of magnitude. Only the frontier and
    the models called out in `highlight` are labelled - the full roster is in
    the accompanying table rather than crowded onto the plot.
    """
    apply_paper_style()
    score = accuracy_column or f"MASE_h{horizon}"
    cost_column = f"per1k_s_h{horizon}"
    # both tables label the family, so take only the score off the accuracy side
    joined = cost.join(accuracy[[score]], how="inner").dropna(
        subset=[cost_column, score])
    x = joined[cost_column].to_numpy()
    y = joined[score].to_numpy()
    on_front = pareto_frontier(x, y)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for family, group in joined.groupby("family"):
        style = family_style(family)
        ax.scatter(group[cost_column], group[score],
                   s=58, marker=style["marker"], color=style["color"],
                   edgecolor="white", linewidth=0.6, zorder=3, label=family)

    order = np.argsort(x[on_front])
    frontier = ax.step(x[on_front][order], y[on_front][order], where="post",
                       color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=2)[0]
    frontier.set_label("Pareto frontier")

    ax.set_xscale("log")
    # room for the labels on the outermost points, which otherwise run off
    ax.set_xlim(x.min() * 0.55, x.max() * 2.6)
    span = y.max() - y.min()
    ax.set_ylim(y.min() - span * 0.10, y.max() + span * 0.10)

    _place_labels(ax, list(joined.index), x, y,
                  [front or name in highlight
                   for name, front in zip(joined.index, on_front)])

    ax.set_xlabel("compute seconds per 1,000 series (one fit + one predict, log scale)")
    ax.set_ylabel(f"{accuracy_label}, h={horizon}  (lower is better)")
    ax.set_title(f"Accuracy against compute, horizon {horizon}", loc="left")
    ax.grid(axis="both", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=_family_handles(set(joined["family"])) + [frontier],
              loc="upper right", fontsize=8)
    return fig


def error_distribution(metrics, horizons, metric="MASE", families=None):
    """F8: the spread behind each median, one panel per horizon.

    Medians hide that some models are wrong in a much wider band than others,
    which is what the box plot is here to show.
    """
    apply_paper_style()
    scope = metrics[metrics["metric"] == metric]
    order = (scope[scope["horizon"] == horizons[0]]
             .groupby("model")["value"].median().sort_values().index.tolist())

    # sharex too: different scales per panel would make the spreads look
    # comparable across horizons when they are not
    fig, axes = plt.subplots(1, len(horizons), figsize=(7.4, 5.2),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        per_model = [scope[(scope["model"] == m) & (scope["horizon"] == horizon)]["value"]
                     .dropna().to_numpy() for m in order]
        box = ax.boxplot(per_model, vert=False, widths=0.6, showfliers=False,
                         patch_artist=True, medianprops={"color": INK, "linewidth": 1.2})
        for patch, model in zip(box["boxes"], order):
            family = families.get(model, "") if families else ""
            patch.set_facecolor(family_style(family)["color"])
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
            patch.set_linewidth(0.6)
        for whisker in box["whiskers"] + box["caps"]:
            whisker.set_color(INK_MUTED)
            whisker.set_linewidth(0.8)
        ax.axvline(1.0, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=1)
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_yticklabels(order)
        ax.set_xlabel(f"{metric}, h={horizon}")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    if families:
        axes[-1].legend(handles=_family_handles(set(families.values()), patch=True),
                        loc="lower right", fontsize=8)
    fig.suptitle(f"{metric} across series; dotted line marks parity with the "
                 "scaling baseline", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def coverage_calibration(metrics, horizons, target=0.8, families=None):
    """Achieved coverage of the nominal interval, per model.

    A model far below the target is claiming more certainty than it has; far
    above means intervals so wide they say nothing.
    """
    apply_paper_style()
    scope = metrics[metrics["metric"] == "coverage_80"]
    order = (scope[scope["horizon"] == horizons[0]]
             .groupby("model")["value"].median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    positions = np.arange(len(order))
    ax.axvline(target, color=INK, linewidth=1.0, zorder=1)
    ax.annotate(f"nominal {target:.0%}", (target, len(order) - 0.4),
                textcoords="offset points", xytext=(6, 0), fontsize=8, color=INK)

    for offset, horizon, filled in zip((-0.16, 0.16), horizons, (True, False)):
        values = (scope[scope["horizon"] == horizon].groupby("model")["value"]
                  .median().reindex(order))
        for pos, model, value in zip(positions, order, values):
            style = family_style(families.get(model, "") if families else "")
            ax.plot(value, pos + offset, marker=style["marker"], markersize=6,
                    color=style["color"] if filled else "white",
                    markeredgecolor=style["color"], markeredgewidth=1.2, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels(order)
    ax.set_xlabel("achieved coverage of the nominal 80% interval")
    ax.set_title("Interval calibration (filled: h=%d, open: h=%d)" % tuple(horizons),
                 loc="left")
    if families:
        missing = sorted(set(families) - set(order))
        if missing:
            ax.set_xlabel(ax.get_xlabel()
                          + f"\nno intervals produced by: {', '.join(missing)}")
        ax.legend(handles=_family_handles(set(families.values())),
                  loc="lower right", fontsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
