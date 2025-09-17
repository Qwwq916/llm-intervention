
# llm-intervention
Multi-Layer Intervention for Controllable and Interpretable LLMs

We propose a **weighted multi-layer intervention framework** that adjusts semantic control vectors across selected layers to enhance refusal control while mitigating semantic degradation in LLMs. The code supports reproducing all experiments reported in the paper.

## Features
-  Extraction of **refusal semantic vectors** from hidden states.
-  **Single-layer** and **multi-layer** intervention experiments.
-  Automated **Bayesian optimization** for optimal weight search.
-  Benchmark evaluation on LLaMA, Gemma, Mistral, and Qwen models.

## Datasets
The dataset used in this project is specifically designed for behavioral intervention studies involving large language models (LLMs). 

-  Anchor dataset: Used to calculate "rejection vectors" for model intervention by providing behavioral benchmarks. File path: ./data/benigntest(harmful).txt

-  Test dataset: Used to evaluate the effectiveness of interventions, its content is context-specific but differs from baseline content. File path: ./data/harmful(harmless).csv

## Usage Guidelines
-  Maintain scenario relevance but content independence when expanding
-  Harmful samples for research only, following ethical guidelines
-  Directly loadable via load_dataset function in utils.py

## Install Dependencies
1. Create a new virtual environment with conda:
   
  ```python
  conda create -n YOUR_ENVIRONMENT_NAME python=3.11
```

2. Switch to it:

 ```python
 conda activate YOUR_ENVIRONMENT_NAME
```

3. Install pip requirements:

 ```python
  pip install -r requirements.txt
```

4. Run single-layer intervention:
 
 ```python
python Intervention_sub(add)_perlayer.py
```

5. Run muti-layer equal-weight:

```python
python Intervention_equal_sub.py
```

6. Run Bayesian optimization for weight tuning:

```python
python optimize_intervention_weights.py
```
