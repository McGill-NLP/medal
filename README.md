# MeDAL dataset

![](./figures/rs_illustration.svg)

Repository for **Me**dical **D**ataset for **A**bbreviation Disambiguation for Natural **L**anguage Understanding (MeDAL), a large medical text dataset curated for abbreviation disambiguation, designed for natural language understanding pre-training in the medical domain. It was published at the ClinicalNLP workshop at EMNLP.

📜 [Paper](https://www.aclweb.org/anthology/2020.clinicalnlp-1.15/)\
💻 [Code](https://github.com/BruceWen120/medal)\
💾 [Dataset (Kaggle)](https://www.kaggle.com/xhlulu/medal-emnlp)\
💽 [Dataset (Zenodo)](https://zenodo.org/record/4265633)

<!-- 🤗 [Pre-trained ELECTRA Small (Hugging Face)]()

🔥 [Pre-trained LSTM (Torch Hub)]() -->

## Quickstart

<!-- COMING SOON
### Using Torch Hub

You can directly load LSTM and LSTM-SA with `torch.hub`:
```python
import torch

lstm = torch.hub.load("BruceWen120/medal, "lstm")
lstm_sa = torch.hub.load("BruceWen120/medal, "lstm_sa")
```

If you want to use the Electra model, you need to first install transformers:
```
pip install transformers
```
Then, you can load it with `torch.hub`:
```python
import torch
electra = torch.hub.load("BruceWen120/medal, "electra")
```

-->

### Using Huggingface `transformers`

If you are only interested in the pre-trained ELECTRA weights (without the disambiguation head), you can load it directly from the Hugging Face Repository:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("xhlulu/electra-medal")
tokenizer = AutoTokenizer.from_pretrained("xhlulu/electra-medal")
```


## Running the code

Coming soon!


## Citation

Download the `bibtex` [here](https://www.aclweb.org/anthology/2020.clinicalnlp-1.15.bib), or copy the text below:
```
@inproceedings{wen-etal-2020-medal,
    title = "{M}e{DAL}: Medical Abbreviation Disambiguation Dataset for Natural Language Understanding Pretraining",
    author = "Wen, Zhi and Lu, Xing Han and Reddy, Siva",
    booktitle = "Proceedings of the 3rd Clinical Natural Language Processing Workshop",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.clinicalnlp-1.15",
    pages = "130--135",
}
```



## License, Terms and Conditions

The ELECTRA model is licensed under [Apache 2.0](https://github.com/google-research/electra/blob/master/LICENSE). The license for the libraries used in this project (`transformers`, `pytorch`, etc.) can be found in their respective GitHub repository. Our model is released under a MIT license.


The original dataset was retrieved and modified from the [NLM website](https://www.nlm.nih.gov/databases/download/pubmed_medline.html). By using this dataset, you are bound by the [terms and conditions](https://www.nlm.nih.gov/databases/download/terms_and_conditions_pubmed.html) specified by NLM:

> INTRODUCTION
> 
> Downloading data from the National Library of Medicine FTP servers indicates your acceptance of the following Terms and Conditions: No charges, usage fees or royalties are paid to NLM for this data.
> 
> MEDLINE/PUBMED SPECIFIC TERMS
> 
> NLM freely provides PubMed/MEDLINE data. Please note some PubMed/MEDLINE abstracts may be protected by copyright.  
> 
> GENERAL TERMS AND CONDITIONS
> 
>    * Users of the data agree to:
>        * acknowledge NLM as the source of the data by including the phrase "Courtesy of the U.S. National Library of Medicine" in a clear and conspicuous manner,
>        * properly use registration and/or trademark symbols when referring to NLM products, and
>        * not indicate or imply that NLM has endorsed its products/services/applications. 
>
>    * Users who republish or redistribute the data (services, products or raw data) agree to:
>        * maintain the most current version of all distributed data, or
>        * make known in a clear and conspicuous manner that the products/services/applications do not reflect the most current/accurate data available from NLM.
>
>    * These data are produced with a reasonable standard of care, but NLM makes no warranties express or implied, including no warranty of merchantability or fitness for particular purpose, regarding the accuracy or completeness of the data. Users agree to hold NLM and the U.S. Government harmless from any liability resulting from errors in the data. NLM disclaims any liability for any consequences due to use, misuse, or interpretation of information contained or not contained in the data.
>
>    * NLM does not provide legal advice regarding copyright, fair use, or other aspects of intellectual property rights. See the NLM Copyright page.
>
>    * NLM reserves the right to change the type and format of its machine-readable data. NLM will take reasonable steps to inform users of any changes to the format of the data before the data are distributed via the announcement section or subscription to email and RSS updates.
