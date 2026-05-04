import scipy.io as sio
import numpy as np

mat = sio.loadmat('/media/neuraldyn/PortableSSD/Cavanagh_DEP/Depression_PS_Task/Data/507.mat', squeeze_me=True, struct_as_record=False)
print("Keys in mat:", list(mat.keys()))
for key in ['EEG', 'data', 'FS', 'chanlocs', 'event', 'srate', 'times']:
    if key in mat:
        val = mat[key]
        print(f"--- {key} ---")
        if isinstance(val, sio.matlab.mio5_params.mat_struct):
            print("Fields:", val._fieldnames)
            if hasattr(val, 'data'):
                print("data shape:", np.shape(val.data))
            if hasattr(val, 'srate'):
                print("srate:", val.srate)
            if hasattr(val, 'chanlocs'):
                chans = val.chanlocs
                if hasattr(chans, '__iter__'):
                    labels = [c.labels for c in chans if hasattr(c, 'labels')]
                    print("chan labels (first 10):", labels[:10])
            if hasattr(val, 'event'):
                events = val.event
                print("event type:", type(events), "len:", len(events) if hasattr(events, '__len__') else "unknown")
            if hasattr(val, 'ref'):
                print("ref:", val.ref)
        elif isinstance(val, np.ndarray):
            print(f"Shape: {val.shape}")
            if key == 'chanlocs' and len(val) > 0 and hasattr(val[0], 'labels'):
                print("chan labels (first 10):", [c.labels for c in val[:10]])
        else:
            print(f"Type: {type(val)}")
            print(f"Value: {val}")

# If EEG is in mat, check its contents specifically
if 'EEG' in mat:
    EEG = mat['EEG']
    if hasattr(EEG, 'data'):
        print("EEG.data shape (channels x times x trials) or (channels x times):", np.shape(EEG.data))
    if hasattr(EEG, 'xmin') and hasattr(EEG, 'xmax'):
        print(f"Epoch time: {EEG.xmin} to {EEG.xmax}")
    if hasattr(EEG, 'event'):
        for i, e in enumerate(EEG.event[:5]):
            type_str = getattr(e, 'type', None)
            print(f"Event {i} type:", type_str)
