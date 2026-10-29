"""Write the Question 2 report tables (LaTeX) straight from the saved results, so no number is retyped.

Outputs report_tables/table{1..6}.tex and report_tables/figure1_stats.tex.
"""
import json
from pathlib import Path

import numpy as np
import torch

from data import load_splits, window
from flow import flow_directions_and_speeds
from metrics import fmt
from transforms import mirror, rotate

RESULTS = Path("results")
OUT = Path("report_tables")


def tex_number(value: float) -> str:
    """fmt() rules (3 s.f., scientific below 1e-3) rendered for LaTeX."""
    text = fmt(value)
    if "e" in text:
        mantissa, exponent = text.split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return text


def tex_cell(values: list[float], digits: str | None = None) -> str:
    render = (lambda v: format(v, digits)) if digits else tex_number
    if len(values) == 1:
        return render(values[0])
    mean = sum(values) / len(values)
    return f"\\makecell[l]{{{render(mean)}\\\\\\footnotesize ({render(values[0])}, {render(values[1])})}}"


def transposed_table(rows: dict, columns: list[tuple[str, str]], metrics: list[tuple[str, str]],
                     caption: str, label: str, extra_rows: list[tuple[str, list[str]]] = (),
                     size: str = r"\small") -> str:
    """Metrics as table rows, models as table columns."""
    spec = "@{}l" + "l" * len(columns) + "@{}"
    lines = [r"\begin{table}[htbp]", r"\centering" + size, f"\\caption{{{caption}}}\\label{{{label}}}",
             f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & " + " & ".join(f"\\textbf{{{name}}}" for _, name in columns) + r" \\", r"\midrule"]
    for key, name in metrics:
        cells = [tex_cell(rows[col][key]) if key in rows[col] else "--" for col, _ in columns]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    for name, cells in extra_rows:
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table1() -> str:
    def max_diff(a, b):
        return f"{(a - b).abs().max().item():g}, \\texttt{{torch.equal}} = {torch.equal(a, b)}"

    def bright(v):
        f, _, i, j = torch.nonzero(v).squeeze(0).tolist()
        return f"({f}, {i}, {j})"

    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1
    numbered = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).clone()

    def frames(k):
        w = window(numbered, start=1, speed=k)
        ins = ", ".join(str(int(v)) for v in w.inputs[:, 0, 0, 0])
        tgt = ", ".join(str(int(v)) for v in w.targets[w.target_mask, 0, 0, 0])
        return f"{ins} / {tgt}"

    rows = [
        ("(a)", r"$\max|M(M(x)) - x|$", "0", max_diff(mirror(mirror(x)), x)),
        ("(b)", r"$\max|R^4(x) - x|$", "0", max_diff(rotate(x, 4), x)),
        ("(c)", r"$\max|MRM(x) - R^{-1}(x)|$", "0", max_diff(mirror(rotate(mirror(x))), rotate(x, -1))),
        ("(d)", "bright pixel of $M(p)$ (frame, row, column)", "(7, 10, 43)", bright(mirror(p))),
        ("(e)", "bright pixel of $R(p)$ (frame, row, column)", "(7, 43, 10)", bright(rotate(p))),
        ("(f)", r"1$\times$ input / target frames", "1--5 / 6--10", frames(1)),
        ("(g)", r"2$\times$ input / target frames", "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", frames(2)),
        ("(h)", r"3$\times$ input / target frames", "1, 4, 7, 10, 13 / 16, 19", frames(3)),
    ]
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Checks of the transformation and window functions that all experiments use "
             r"(synthetic tensors).}\label{tab:q2-1}",
             r"\begin{tabular}{@{}lp{0.3\textwidth}p{0.24\textwidth}p{0.3\textwidth}@{}}", r"\toprule",
             r"& \textbf{Check} & \textbf{Expected} & \textbf{Yours} \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def figure1_stats() -> str:
    directions, speeds = flow_directions_and_speeds(load_splits()["train"][:50])
    counts, _ = np.histogram(directions, bins=np.arange(0, 361, 20))
    fractions = counts / len(directions)
    return (f"\\newcommand{{\\FlowKept}}{{{len(directions):,}}}\n"
            f"\\newcommand{{\\FlowMedian}}{{{np.median(speeds):.3g}}}\n"
            f"\\newcommand{{\\FlowMinBin}}{{{fractions.min():.3f}}}\n"
            f"\\newcommand{{\\FlowMaxBin}}{{{fractions.max():.3f}}}\n"
            f"\\newcommand{{\\FlowPninety}}{{{np.percentile(speeds, 90):.3g}}}\n")


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "table1.tex").write_text(table1())
    (OUT / "figure1_stats.tex").write_text(figure1_stats())

    t2 = load("table2")
    (OUT / "table2.tex").write_text(transposed_table(
        t2, [("Model A", "Model A"), ("Blank", "Blank"), ("Copy-last", "Copy-last")],
        [(f"MSE_{h}", f"MSE$_{h}$") for h in range(1, 6)] + [("E5", "$E^{(5)}$")],
        r"Model A and the two reference predictors (test split, 1$\times$, $t=1$). "
        r"Entries for model A show mean (seed 0, seed 1).", "tab:q2-2",
        extra_rows=[("Trainable parameters", ["480,784", "0", "0"])]))

    t3 = load("table3")
    (OUT / "table3.tex").write_text(transposed_table(
        t3, [("Model A", "Model A"), ("Blank", "Blank"), ("Copy-last", "Copy-last")],
        [("E5", "$E^{(5)}$"), ("CM", "$C_M$"), ("EM", "$E_M$"), ("CR", "$C_R$"), ("ER", "$E_R$")],
        r"Mirror and rotation tests (test split, 1$\times$, $t=1$). Entries show mean (seed 0, seed 1).", "tab:q2-3"))

    t4 = load("table4")
    (OUT / "table4.tex").write_text(transposed_table(
        t4, [("Model A", "Model A"), ("Copy-last", "Copy-last")],
        [("E2@1x", r"$E^{(2)}$ at 1$\times$"), ("E2@2x", r"$E^{(2)}$ at 2$\times$"), ("E2@3x", r"$E^{(2)}$ at 3$\times$"),
         ("E5@1x", r"$E^{(5)}$ at 1$\times$"), ("E5@2x", r"$E^{(5)}$ at 2$\times$")],
        r"Faster motion (test split, $t=1$). Entries show mean (seed 0, seed 1).", "tab:q2-4"))

    metrics_5 = [("E5", r"$E^{(5)}$ at 1$\times$"), ("CM", "$C_M$"), ("EM", "$E_M$"), ("CR", "$C_R$"), ("ER", "$E_R$"),
                 ("E2@1x", r"$E^{(2)}$ at 1$\times$"), ("E2@2x", r"$E^{(2)}$ at 2$\times$"),
                 ("E2@3x", r"$E^{(2)}$ at 3$\times$")]

    def timing_rows(rows, names):
        return [("Trainable parameters", ["480,784"] * len(names)),
                (r"\makecell[l]{Training time\\per epoch (s)}", [tex_cell(rows[n]["train_s_per_epoch"], ".2f") for n in names]),
                (r"\makecell[l]{Inference time,\\1,000 videos (s)}", [tex_cell(rows[n]["inference_s"], ".3f") for n in names])]

    t5 = load("table5")
    (OUT / "table5.tex").write_text(transposed_table(
        t5, [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")], metrics_5,
        r"Models A--D (test split, $t=1$). Entries show mean (seed 0, seed 1).", "tab:q2-5",
        extra_rows=timing_rows(t5, ["A", "B", "C", "D"]), size=r"\footnotesize"))

    t6 = load("table6")
    (OUT / "table6.tex").write_text(transposed_table(
        t6, [("B", "Symmetrised B"), ("D", "Symmetrised D (final)")], metrics_5,
        r"Symmetrised forecasters (test split, $t=1$). Entries show mean (seed 0, seed 1).", "tab:q2-6",
        extra_rows=timing_rows(t6, ["B", "D"])))
    print("wrote", sorted(p.name for p in OUT.iterdir()))
