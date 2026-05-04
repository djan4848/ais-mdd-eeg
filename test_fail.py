import logging; import traceback; from pathlib import Path; import sys; import mne
from mne_icalabel import label_components
f=list(Path("Cavanagh/Depression_PS_Task/derivatives/epochs").glob("*.fif"))[0]
epochs = mne.read_epochs(f, preload=True, verbose=False)
ica = mne.preprocessing.ICA(n_components=15, random_state=42, method='fastica', max_iter=200)
ica.fit(epochs, verbose=False)
try:
    ic_labels = label_components(epochs, ica, method='iclabel')
    labels = ic_labels['labels']
    probs = ic_labels['y_pred_proba']
    brain_ics = [i for i, (lbl, prb) in enumerate(zip(labels, probs)) if lbl == 'brain' and prb > 0.70]
    print("Brain ICs length:", len(brain_ics))
except Exception as e:
    print("Error:", e)
