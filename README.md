# MAFALDA_NAACL

<img src="assets/logo.png" width="400" height="400">

## Abstract
We introduce MAFALDA, a benchmark for fallacy classification that unites previous datasets. It comes with a taxonomy of fallacies that aligns, refines, and unifies previous classifications. We further provide a manual annotation of the dataset together with manual explanations for each annotation. We propose a new annotation scheme tailored for subjective NLP tasks, and a new evaluation method designed to handle subjectivity.<br/>
We then evaluate several language models under a zero-shot learning setting and human performances on MAFALDA to assess their fallacy detection and classification capability. 

## Installation
```bash
git clone https://github.com/username/MAFALDA_NAACL.git
cd  MAFALDA_NAACL
pip install -r requirements.txt
```

## Run Experiment with Dummy Models
```bash
./run_dummy.sh
```

## Run Experiments with Local Models

### with GPU
```bash
./run_with_gpu.sh
```

### with CPU
```bash
./run_with_cpu.sh
```
## Run Experiments with OpenAI (GPT 3.5)
```bash
./run_with_openai.sh
```

## Run Evaluation
```bash
./run_eval.sh
```

N.B: Code tested with Python 3.9.12

We acknowledge the use of a code writing assistant.