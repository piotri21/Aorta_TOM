import tensorflow as tf
import pandas as pd
from pathlib import Path


data_dir = Path('../DATA')
raw_files = sorted([f for f in data_dir.glob('**/*.nrrd') if 'seg' not in f.name])
seg_files = sorted([f for f in data_dir.glob('**/*seg*.nrrd')])

def get_base_name(path):
    return path.stem.replace('.seg', '').replace('_seg', '')

raw_dict = {get_base_name(f): f for f in raw_files}
seg_dict = {get_base_name(f): f for f in seg_files}

common_keys = set(raw_dict.keys()) & set(seg_dict.keys())
df = pd.DataFrame({
    'raw_path': [str(raw_dict[k]) for k in common_keys],
    'seg_path': [str(seg_dict[k]) for k in common_keys]
})

print(df.head())