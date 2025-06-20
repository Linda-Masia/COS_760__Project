# Leveraging Transfer Learning and Data Augmentation for Multilabel Emotion Classification in Bantu Languages

**Contributors:**  
- Sharika Kanti Narsing  
- Linda Masia  
- Simphiwe Nonabe  

---

## Overview  
This project explores transfer learning and data augmentation techniques to improve multilabel emotion classification in low-resource African (Bantu) languages using the BRIGHTER dataset. All experiments, data processing, model training, and evaluation are contained within a single Jupyter notebook.

---

## Contents of the Zip File  
- `Brighter_Final.ipynb` — Main Jupyter notebook containing all project code and experiments  
- `README.md` — This file  

---

## Setup Instructions  

1. Download and install [Anaconda](https://www.anaconda.com/products/distribution) or ensure you have Python 3.8+ and Jupyter installed.

2. Create and activate a virtual environment (optional but recommended):  
   ```bash
   conda create -n brighter python=3.8 -y
   conda activate brighter
   ```

3. Install required Python packages:  
   ```bash
   pip install -r requirements.txt
   ```  
   *(If `requirements.txt` is not included, install dependencies manually: `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, etc.)*

---

## Running the Code  

1. Open the notebook:  
   ```bash
   jupyter notebook Brighter_Final.ipynb
   ```  
   or launch Jupyter from your IDE.

2. Follow the notebook cells sequentially to reproduce data loading, preprocessing, model training, and evaluation steps.

---

## Data Information  

- The notebook uses the **BRIGHTER Emotion Dataset**, which is loaded dynamically using the `datasets` library from Hugging Face.  
- The dataset is **not included** in the zip file due to its size.  
- To download, run the following code inside the notebook:  
  ```python
  from datasets import load_dataset
  dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories")
  ```  
- Ensure you have internet access when running the notebook to load the dataset.

---

## Notes  

- The notebook is self-contained; all code, explanations, and results are combined in one place for ease of use and reproducibility.  
- For best results, run on a machine with GPU support or use Google Colab.  
- [Optional] You can upload the notebook to Google Colab and run it interactively:  
  [▶️ Open in Colab](https://colab.research.google.com)

---

## References  
1. [BRIGHTER Dataset on Hugging Face](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories)  
2. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.  
3. Pfeiffer, J. et al. (2021). AdapterFusion: Non-Destructive Task Composition for Transfer Learning.
