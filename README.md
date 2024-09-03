# LLaMA_cosmoChat

LLaMA_cosmoChat is a python module to generate SQL queries to CosmoHub with LLaMA generative AI open source models. 

## Installation
(Last update Sept 3 2024)

- Create a new conda environment. It is usually better to follow python version one or two behind, we recommend 3.11.

```
conda create -n chatcosmohub -c conda-forge python=3.11 pip=24.0
conda activate chatcosmohub
```

- Clone the repo into your machine and perform an *editable* installation:

```
git clone https://github.com/lauracabayol/LLaMA_cosmoChat.git 
cd LLaMA_cosmoChat
pip install -e .
```

- To access the [LLaMA models]( https://ai.meta.com/blog/meta-llama-3-1/), one needs to request it to [Huggingface](https://huggingface.co/docs/transformers/main/en/model_doc/llama). To access the model, one needs to declare the access token in Huggingface as an environment variable called 'KERNEL_HIGGINFACE'.

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel` and `jupytext`:

```
pip install ipykernel
python -m ipykernel install --user --name chatcosmohub --display-name chatcosmohub
```

#### Tutorials:

In the `notebooks` folder, there is a  tutorial.
These are .py scripts, in order to pair them to .ipynb, please run:

```
jupytext your_script --to ipynb
```