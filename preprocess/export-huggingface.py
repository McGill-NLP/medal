# %%
import torch
from models.electra import Electra
import transformers as tfm


# %%
net = Electra()


# %%
url = "https://github.com/BruceWen120/medal/releases/download/data0.0.1/electra.pt"


# %%
state_dict = torch.hub.load_state_dict_from_url(url)


# %%
for key, value in state_dict.items():
    new_key = key[len('module.'): ] if key.startswith('module.') else key
    if new_key not in net.state_dict():
        print(new_key, 'not expected')
        continue
    try:
        net.state_dict()[new_key].copy_(value)
    except:
        print(new_key, 'not loaded')
        continue

# %% [markdown]
# ## Save pytorch

# %%
net.electra.save_pretrained('models/electra-medal')


# %%
electra = tfm.ElectraModel.from_pretrained('models/electra-medal')

# %% [markdown]
# ## Convert to tensorflow and save

# %%
tf_electra = tfm.TFElectraModel.from_pretrained('models/electra-medal', from_pt=True)


# %%
tf_electra.save_pretrained('models/electra-medal')

# %% [markdown]
# ## Save tokenizer

# %%
tok = tfm.ElectraTokenizer.from_pretrained('google/electra-small-discriminator')


# %%
tok.save_pretrained('models/electra-medal')


