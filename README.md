# SUBoost: A Novel Boosting-Based Selective Undersampling for Handling Imbalanced Data

[![Paper](https://img.shields.io/badge/IEEE-Paper-blue.svg)](https://ieeexplore.ieee.org/document/11273834)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-yellow.svg)](https://www.python.org/)

**Official implementation of the paper:** *“[SUBoost: A Novel Boosting-Based Selective Undersampling for Handling Imbalanced Data](https://ieeexplore.ieee.org/document/11273834)”*  
Published in **IEEE, 10 December 2025**

---

## 📌 Overview

**SUBoost (Selective Undersampling Boosting)** is a boosting-based ensemble algorithm designed to tackle **imbalanced classification problems** more effectively than traditional methods.

Unlike conventional approaches such as AdaBoost and RUSBoost that rely on random undersampling, SUBoost adopts a **selective undersampling strategy**: at each boosting iteration, majority-class samples that are confidently and correctly classified by the current weak learner are removed, while difficult and informative samples are preserved. This allows subsequent learners to focus on harder decision regions without losing critical information.

SUBoost also uses an **adaptive weight initialization** based on class sizes to ensure balanced learning from the start. Extensive experiments on **27 imbalanced datasets** from the KEEL repository show that SUBoost consistently outperforms state-of-the-art ensemble methods, achieving top performance in terms of **AUC** and **G-Mean**, especially in highly imbalanced scenarios.

---

## 🚀 Key Features

* **Selective Undersampling:** Removes only correctly classified majority-class samples, rather than random removal, forcing learners to focus on harder instances.

* **Adaptive Weight Initialization:** Sample weights are initialized as `w_i = 1 / n_minor` for minority samples and `w_i = 1 / n_major` for majority samples, ensuring equal class importance.

* **Information Preservation:** Retains complex majority-class samples and avoids the loss of informative data common in random undersampling techniques.

* **Robust Performance:** Ranked **1st** in both **AUC** and **G-Mean** across extensive experimental benchmarks.

---

## 📊 Experimental Results

SUBoost was evaluated on **27 imbalanced datasets** from the KEEL repository using **AUC** and **G-Mean** metrics.

| Metric     | Rank | Description                                   |
|------------|:----:|-----------------------------------------------|
| **AUC**    | 1st  | Superior class discrimination                 |
| **G-Mean** | 1st  | Best balance between minority and majority accuracy |

Results confirm that SUBoost consistently outperforms competing ensemble methods, particularly in highly imbalanced scenarios.

---

## 💻 Prerequisites

* Python 3.8+
* NumPy
* Pandas
* Scikit-learn

---

## 🔗 Citation

If you use SUBoost in your research, please cite the following paper:

```bibtex
@INPROCEEDINGS{11273834,
  author={Baghmishe, Nima Rasi and Tanha, Jafar and Roshan, Ehsan},
  title={SUBoost: A Novel Boosting-Based Selective Undersampling for Handling Imbalanced Data},
  booktitle={2025 IEEE International Conference},
  year={2025},
  pages={1--7},
  doi={10.1109/Conference.2025.11273834}
}
