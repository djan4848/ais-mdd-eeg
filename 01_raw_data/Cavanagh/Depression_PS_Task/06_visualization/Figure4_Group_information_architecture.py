import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dds_base.io.paths import DERIV_ROOT

# =========================================================
# PATHS
# =========================================================
BASE = DERIV_ROOT / "te_n450" / "group_architecture"

TE_PATH = BASE / "te_to_frontal_by_group.csv"
PID_PATH = BASE / "pid_by_group.csv"
ROLE_PATH = BASE / "integration_index_by_group.csv"

OUTDIR = DERIV_ROOT / "figures_paper"
OUTDIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# LOAD
# =========================================================
te = pd.read_csv(TE_PATH)
pid = pd.read_csv(PID_PATH)
role = pd.read_csv(ROLE_PATH)

group_order = ["CTL", "DEP"]
cond_order = ["INIT", "INHIB"]
cond_labels = {"INIT": "Initiation", "INHIB": "Inhibition"}

for df in [te, pid, role]:
    df["cond"] = df["cond"].replace(cond_labels)

# =========================================================
# COLOR PALETTE (aligned with final Figure 3 style)
# =========================================================
COL_INIT = "#bdc3c7"      # light grey
COL_INHIB = "#2c3e50"     # dark blue-grey

COL_UNIQUE_RH = "#9b59b6" # muted purple
COL_UNIQUE_LH = "#52b788" # muted green
COL_REDUND = "#5b8db8"    # muted blue
COL_SYNERGY = "#d36b6b"   # muted red

PID_COLORS = [COL_UNIQUE_RH, COL_UNIQUE_LH, COL_REDUND, COL_SYNERGY]

# =========================================================
# STYLE
# =========================================================
sns.set_context("paper", font_scale=1.30)
sns.set_style("ticks")

fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))
fig.suptitle(
    "Group Differences in the Information Architecture of the N450",
    fontsize=19,
    fontweight="bold",
    y=0.98
)

# =========================================================
# PANEL A — TE to frontal
# =========================================================
ax = axes[0]

sns.barplot(
    data=te,
    x="group",
    y="mean_te_bits",
    hue="cond",
    order=group_order,
    hue_order=["Initiation", "Inhibition"],
    palette=[COL_INIT, COL_INHIB],
    ax=ax,
    errorbar=None
)

# manual SD bars
for i, grp in enumerate(group_order):
    for j, cond in enumerate(["Initiation", "Inhibition"]):
        row = te[(te["group"] == grp) & (te["cond"] == cond)]
        if len(row) == 0:
            continue
        mean = row["mean_te_bits"].values[0]
        sd = row["std_te_bits"].values[0]
        xpos = i + (-0.2 if j == 0 else 0.2)
        ax.errorbar(
            x=xpos, y=mean, yerr=sd,
            fmt="none", ecolor="black", capsize=4, lw=1.1
        )

ax.set_title("A. TE to Frontal", fontweight="bold", pad=14)
ax.set_xlabel("")
ax.set_ylabel("TE (bits)")
ax.legend_.remove()
sns.despine(ax=ax)

# =========================================================
# PANEL B — PID stacked horizontal
# =========================================================
ax = axes[1]

pid_inhib = pid[pid["cond"] == "Inhibition"].copy()
pid_inhib = pid_inhib.set_index("group").loc[group_order].reset_index()

components = ["mean_unique_rh", "mean_unique_lh", "mean_redundancy", "mean_synergy"]
labels = ["Unique RH", "Unique LH", "Redundancy", "Synergy"]

left = np.zeros(len(pid_inhib))
ypos = np.arange(len(pid_inhib))

for comp, lab, col in zip(components, labels, PID_COLORS):
    vals = pid_inhib[comp].values
    ax.barh(ypos, vals, left=left, color=col, label=lab, height=0.55)
    left += vals

ax.set_yticks(ypos)
ax.set_yticklabels(group_order)
ax.invert_yaxis()  # CTL arriba
ax.set_xlabel("PID components (bits)")
ax.set_title("B. PID in Inhibition", fontweight="bold", pad=14)
sns.despine(ax=ax, left=False, bottom=False)

# =========================================================
# PANEL C — Integration index
# =========================================================
ax = axes[2]

sns.barplot(
    data=role,
    x="group",
    y="integration_index",
    hue="cond",
    order=group_order,
    hue_order=["Initiation", "Inhibition"],
    palette=[COL_INIT, COL_INHIB],
    ax=ax,
    errorbar=None
)

ax.set_title("C. Integration Index", fontweight="bold", pad=14)
ax.set_xlabel("")
ax.set_ylabel(r"(TE$_{in}$ + Synergy) / AIS")
ax.legend_.remove()
sns.despine(ax=ax)

# =========================================================
# GLOBAL LEGEND
# =========================================================
handles_cond = [
    plt.Rectangle((0, 0), 1, 1, color=COL_INIT),
    plt.Rectangle((0, 0), 1, 1, color=COL_INHIB),
]
labels_cond = ["Initiation", "Inhibition"]

handles_pid = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PID_COLORS]
labels_pid = labels

fig.legend(
    handles_cond + handles_pid,
    labels_cond + labels_pid,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.93),
    ncol=6,
    frameon=False,
    fontsize=10.5
)

# =========================================================
# FOOTNOTE
# =========================================================
n_ctl = int(te[te["group"] == "CTL"]["n_subjects"].max())
n_dep = int(te[te["group"] == "DEP"]["n_subjects"].max())

fig.text(
    0.5, 0.02,
    f"Sample: CTL = {n_ctl}, DEP = {n_dep} | N450 window: ±200 ms around frontal peak | TE lag = 16 ms | PID bins = 4",
    ha="center",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
)

plt.tight_layout(rect=[0, 0.06, 1, 0.88])
plt.savefig(OUTDIR / "Figure_4_Group_Information_Architecture_v3.png", dpi=300, bbox_inches="tight")
plt.show()
