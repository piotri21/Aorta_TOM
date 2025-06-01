import tensorflow as tf
import pandas as pd
from pathlib import Path
import nrrd
import skimage.morphology
import numpy as np

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


class NrrdDataset(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=1, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_raw = []
        batch_skel = []
        for i in batch_indexes:
            raw, _ = nrrd.read(self.df.loc[i, 'raw_path'])
            skel, _ = nrrd.read(self.df.loc[i, 'seg_path'])
            skel = skimage.morphology.skeletonize(skel)

            batch_raw.append(raw.astype(np.float32)[..., np.newaxis])
            batch_skel.append(skel.astype(np.float32)[..., np.newaxis])
        return np.stack(batch_raw), np.stack(batch_skel)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

train_dataset = NrrdDataset(df, batch_size=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(1, 3),
    tf.keras.layers.Conv2D(1, 3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(train_dataset, epochs=5)
