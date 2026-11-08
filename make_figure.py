"""Regenerate the vector figure from state and effect matrices."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from qaf_core import I, X, Y


def main():
    angles = np.linspace(0, 2*np.pi, 281)
    effects = [(I+np.cos(a)*X+np.sin(a)*Y)/2 for a in angles]
    fig, ax = plt.subplots(figsize=(7.0, 2.85), layout="constrained")
    for rho, name, color, style in [
            ((I+X)/2, r"$|+\rangle$", "#066b74", "-"),
            ((I+Y)/2, r"$(|0\rangle+i|1\rangle)/\sqrt{2}$", "#a04d22", "--"),
            (I/2, "Bell marginal", "#3f3f52", ":")]:
        ax.plot(angles, [np.trace(rho@e).real for e in effects],
                color=color, ls=style, lw=2.25, label=name)
    ax.set(xlabel=r"Probe azimuth $\theta$", ylabel="Acceptance probability", xlim=(0,2*np.pi),
           ylim=(-.04,1.04), yticks=[0,.25,.5,.75,1])
    ax.set_xticks(np.arange(5)*np.pi/2, ["0",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=.17)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,1.22), ncol=3, frameon=False)
    fig.savefig(Path(__file__).with_name("responses.pdf"), bbox_inches="tight")


if __name__=="__main__":
    main()
