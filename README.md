# Scoring-Assisted Generative Exploration for Amine (SAGE-Amine)
**Graphical Abstract**
![Graphical Abstract](https://github.com/user-attachments/assets/b1d1bf83-ef0c-4144-af6e-cdb592cd2045)

# What is SAGE-Amine?
**Scoring-Assisted Generative Exploration for Amine (SAGE-Amine)** is an advanced extension of [Scoring-Assisted Generative Exploration](https://github.com/hclim0213/SAGE/tree/main), specifically tailored for Amine.
It employs natural language processing techniques, including LSTM, Transformer, Transformer Decoder, and modified Transformer Decoder (X-Transformer Decoder), to generate Amines. 
These generated amines are then evaluated using multiple QSPR models to assess key physicochemical properties relevant to CO2 absorption.

# Notes
This is additional supplementary data in "SAGE-Amine: Generative Amine Design with Multi-Property Optimization for Efficient CO2 Capture"

# Prerequisites
* LINUX/UNIX Cluster Machines (Ubuntu 20.04)
* Python 3.7
* conda install -c conda-forge openbabel=3.1.0
* pip3 install numpy==1.19.5 scipy scikit-learn xgboost==1.0.2 nltk networkx pyyaml==5.4.1 nbformat lightgbm optuna pandas==1.3.0 soltrannet git+https://github.com/hcji/PyFingerprint.git PyTDC mordred gensim==3.8.3 git+https://github.com/samoturk/mol2vec PubChemPy torch==1.8.1 matplotlib selfies seaborn jupyter neptune-client==0.16.6 tqdm rdkit-pypi==2021.9.5.1 tensorflow==2.5.0 deepchem autograd==1.2 tensorflow_addons==0.13 tensorflow_probability==0.13 umap-learn plotly holoviews==1.14.9 Flask==2.2.2 Jinja2==3.0 bokeh==2.3.3 panel==0.12.1 swagger-spec-validator==2.7.4 guacamol==0.5.5 requests==2.24.0 transformers==4.6.1 sentencepiece==0.1.95 catboost==1.2.5 omegaconf skorch pydot einops==0.6.1 omegaconf==2.3.0 accelerate==0.20.3 x-transformers==1.22.24 scikit-learn-intelex py3Dmol aiohttp==3.8.6 attrs==23.2.0 jsonschema==3.2.0 charset_normalizer==2.0.4 protobuf==3.20.3
* pip3 install torch==1.8.1 torch-geometric==2.0.4 torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torchvision==0.9.1 dirsync==2.2.5 docrep==0.3.2 docstring-parser==0.12 more-itertools==8.8.0 pytorch-lightning==1.3.8 torchmetrics==0.6.2

# Extenstion
* SAGE (https://github.com/hclim0213/SAGE)
* PyFingerprint (https://github.com/hcji/PyFingerprint)
* mol2vec (https://github.com/samoturk/mol2vec)
* guacamol (https://github.com/BenevolentAI/guacamol)
* RAscore (https://github.com/reymond-group/RAscore)
* SolTranNet (https://github.com/gnina/SolTranNet)
* CoPriNet (https://github.com/oxpig/CoPriNet)

# Contact
* Dr. Hocheol Lim (ihc0213@yonsei.ac.kr)

# Acknowledegments
This research was supported by Quantum Advantage challenge research based on 
Quantum Computing through the National Research Foundation of Korea (NRF) 
funded by the Ministry of Science and ICT (RS-2023-00257288).

# How to Cite
Lim, Hocheol, Hyein Cho, Jeonghoon Kim, and Kyoung Tai No., "SAGE-Amine: Generative Amine Design with Multi-Property Optimization for Efficient CO2 Capture" arXiv preprint arXiv:2503.02534 (2025).
Accepted to publish in Carbon Capture Science & Technology.
