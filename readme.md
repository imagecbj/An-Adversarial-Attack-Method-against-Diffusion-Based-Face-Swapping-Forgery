# An Adversarial Attack Method against Diffusion-Based Face Swapping Forgery

Implementation of "An Adversarial Attack Method against Diffusion-Based Face Swapping Forgery".

---

## 1. Environment Setup

This project requires Python 3.8.20. You can set up the environment using the provided `environment.yml` file via Conda.

---

## 2. Adversarial Perturbation Training
To train and generate adversarial perturbations against diffusion-based face swapping systems, execute the main entry script:
```
python main.py
```

## 3. Evaluation
To evaluate the effectiveness and performance of the generated adversarial examples, use the following scripts:
```
python evaluate.py
python evaluate-imp.py
```

## Acknowledgments
We highly appreciate the foundational work done by the following repository, which contributed significantly to this project: faceshield.