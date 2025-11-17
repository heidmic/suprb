import numpy as np
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime

def visualize_pareto_front(self, pareto_front: List, save_path: str = "Objective_Space_Distribution") -> None:
    """
    Generates PDF of the objective space distribution of the input list of rules.
    """
    if not pareto_front:
        print("No Pareto front provided for visualization.")
        return

    objs = self._fitness_objs_runtime()
    labels = self._fitness_labels_runtime()

    if len(objs) != 2:
        print(f"Expected exactly 2 objectives, found {len(objs)}. Skipping plot.")
        return

    obj_matrix = np.array([[objs[0](r), objs[1](r)] for r in pareto_front])

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    plt.figure(figsize=(6, 4))
    plt.scatter(
        obj_matrix[:, 0],
        obj_matrix[:, 1],
        s=40,
        c="royalblue",
        edgecolors="black",
        alpha=0.8
    )
    plt.xlabel(labels[0], fontsize=11)
    plt.ylabel(labels[1], fontsize=11)
    plt.title("Objective Space Distribution", fontsize=12, pad=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"{save_path}_{timestamp}.pdf"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Pareto front visualization saved as PDF: '{pdf_path}'")