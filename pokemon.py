# %%
from datasets.load import load_dataset
import numpy as np

# %%

hfds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
# %%
# resize images to 64x64
hfds = hfds.map(
    lambda sample: {"image": sample["image"].resize((64, 64))},
    remove_columns=["text"],
    batch_size=96,
)
# %%
X = np.stack(hfds["image"])

# %%
X.shape
