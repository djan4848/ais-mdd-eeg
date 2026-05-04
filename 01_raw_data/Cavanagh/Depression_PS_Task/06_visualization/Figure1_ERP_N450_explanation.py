import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dds_base.io.paths import DERIV_ROOT, hayling_epo_files, ROIS

# --- SETTINGS & CONFIG ---
OUTDIR = DERIV_ROOT / "figures_paper"
OUTDIR.mkdir(exist_ok=True, parents=True)

ROIS_DICT = ROIS
WINDOW = (0.390, 0.524)

all_init, all_inhib, roi_data = [], [], []
total_trials_counted = 0

print("-> Processing Figure 1 (Final Layout)...")
for f in hayling_epo_files():
    if any(p in f.name for p in ["P4", "P5", "P19"]): continue 
    try:
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        if len(epochs['ASOC']) > 0 and len(epochs['NOASOC']) > 0:
            total_trials_counted += len(epochs['ASOC']) + len(epochs['NOASOC'])
            all_init.append(epochs['ASOC'].average())
            all_inhib.append(epochs['NOASOC'].average())
            
            # ROI extraction
            ev_init, ev_inhib = all_init[-1], all_inhib[-1]
            for roi_name, channels in ROIS_DICT.items():
                existing = [c for c in channels if c in ev_init.ch_names]
                t_mask = (ev_init.times >= WINDOW[0]) & (ev_init.times <= WINDOW[1])
                a_init = ev_init.copy().pick(existing).data[:, t_mask].mean() * 1e6
                a_inhib = ev_inhib.copy().pick(existing).data[:, t_mask].mean() * 1e6
                roi_data.append({'ROI': roi_name, 'Condition': 'Initiation', 'Amplitude': a_init})
                roi_data.append({'ROI': roi_name, 'Condition': 'Inhibition', 'Amplitude': a_inhib})
    except Exception: continue

ga_init = mne.grand_average(all_init)
ga_inhib = mne.grand_average(all_inhib)
df_roi = pd.DataFrame(roi_data)

# --- PLOTTING ---
fig = plt.figure(figsize=(22, 8))
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")

# Panel A: ERP
ax_erp = plt.subplot2grid((1, 5), (0, 0), colspan=2)
combined = ROIS_DICT['frontal'] + ROIS_DICT['cACC']
mne.viz.plot_compare_evokeds(
    {'Initiation': ga_init, 'Inhibition': ga_inhib},
    picks=[c for c in combined if c in ga_init.ch_names], combine='mean', axes=ax_erp,
    colors={'Initiation': '#7f8c8d', 'Inhibition': '#2c3e50'}, linestyles={'Initiation': '--', 'Inhibition': '-'},
    show=False, title=""
)
ax_erp.set_title("A. Grand Average ERP (Frontal-cACC)", fontweight='bold', pad=25)
ax_erp.axvspan(WINDOW[0], WINDOW[1], color='orange', alpha=0.15, label='N450 Window')
ax_erp.legend(frameon=False, loc='upper right')

# Panel B: Topomap con Colorbar "Pegada"
ax_topo = plt.subplot2grid((1, 5), (0, 2))
diff = mne.combine_evoked([ga_inhib, ga_init], weights=[1, -1])
d_pico = diff.data[:, diff.time_as_index(0.450)[0]]

im, _ = mne.viz.plot_topomap(d_pico, diff.info, axes=ax_topo, show=False, cmap='RdBu_r', contours=0)

# AJUSTE QUIRÚRGICO DE COLORBAR
divider = make_axes_locatable(ax_topo)
cax = divider.append_axes("right", size="5%", pad=0.1) # La "pega" a la derecha del mapa
plt.colorbar(im, cax=cax)

ax_topo.set_title("B. N450 Topography\n(Inhibition - Initiation)", fontweight='bold', pad=25)

# Panel C: ROI Comparison
ax_bar = plt.subplot2grid((1, 5), (0, 3), colspan=2)
sns.barplot(data=df_roi, x='ROI', y='Amplitude', hue='Condition', 
            ax=ax_bar, palette=['#bdc3c7', '#2c3e50'], capsize=.05, errwidth=1.5)
ax_bar.set_title("C. Regional Mean Amplitude\n(N450 Window: 390-524 ms)", fontweight='bold', pad=25)
ax_bar.set_ylabel("Amplitude (µV)")
ax_bar.legend(frameon=False, loc='lower right', fontsize='small')
sns.despine(ax=ax_bar)

# Nota de Muestra
fig.text(0.5, 0.02, f"Total ERP sample: N = {len(all_init)} subjects | n = {total_trials_counted} trials", 
         ha='center', fontsize=13, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(OUTDIR / "Figure_1_Phenomenon_Final_v4_EN.png", dpi=300)
print(f"[OK] Figure 1 updated (v4). Check: {OUTDIR}")
