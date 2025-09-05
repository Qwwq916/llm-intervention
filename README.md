# llm-intervention
Multi-Layer Intervention for Controllable and Interpretable LLMs

We propose a **weighted multi-layer intervention framework** that adjusts semantic control vectors across selected layers to enhance refusal control while mitigating semantic degradation in LLMs. The code supports reproducing all experiments reported in the paper.

## Features
-  Extraction of **refusal semantic vectors** from hidden states.
-  **Single-layer** and **multi-layer** intervention experiments.
-  Automated **Bayesian optimization** for optimal weight search.
-  Benchmark evaluation on LLaMA, Gemma, Mistral, and Qwen models.

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
