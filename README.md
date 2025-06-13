# Leveraging Transfer Learning and Data Augmentation for Multilabel Emotion Classification in Bantu Languages

**Contributors:**  
- Sharika Kanti Narsing 
- Linda Masia  
- Simphiwe Nonabe 

---

## 📌 Research Question
How can different transfer learning approaches, such as cross-lingual models and adapter-based methods, be leveraged to improve multilabel emotion analysis in low-resource African languages?

---

## 🧠 Problem Statement
Although Africa is home to over 2,000 languages, natural language processing (NLP) research remains heavily focused on high-resource languages. Emotion analysis, an important area in NLP with applications in mental health, education, customer support, and policymaking, is underexplored in African languages due to a lack of annotated data and tools.  

The BRIGHTER dataset contributes to addressing this gap by offering multilabel and intensity-based emotion annotations across 28 African languages. However, building generalizable models remains difficult due to limited data and the linguistic diversity of African languages.  

This project investigates how transfer learning and data augmentation can enhance multilabel emotion classification, focusing on English samples from the BRIGHTER dataset as a base for improving performance in low-resource Bantu languages.

---

## 📂 Dataset
**BRIGHTER Emotion Dataset**  
📍 [Hugging Face Dataset Link](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories)  

- **Languages**: 28 African languages  
- **Total Records**: 139,595  
- **Attributes**:  
  `id`, `text` (utterance),  
  `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise` (binary labels),  
  `emotions` (summary of multilabels)  
- **Focus Language**: English (7,290 entries)

---

## 🧪 Methodology
### 🔄 Preprocessing
- Clean text (normalisation, deduplication)  
- Handle inconsistent or missing labels  
- Represent multilabels as binary vectors (e.g., `[1, 0, 1, 0, 0, 1]` for three active emotions)  

### 📊 Baseline
- Fine-tune Multilingual BERT (mBERT) on English samples as a foundation for cross-lingual transfer  

### 🚀 Transfer Learning Approaches
1. **Cross-lingual Fine-tuning**  
   - Fine-tune mBERT on English + low-resource African language subsets  
   - Leverages shared semantic space across languages  

2. **Adapter-based Learning**  
   - Introduce lightweight adapter modules into mBERT  
   - Train adapters on emotion-labelled samples while keeping base model frozen  
   - Ideal for limited data/compute environments  

### 🔁 Data Augmentation
- Apply back-translation on English samples to generate additional training instances  
- Improves generalisation for small African language datasets  

### 📈 Evaluation Strategy
- **Baseline**: mBERT fine-tuned on English BRIGHTER data  
- **Metrics**:  
  Precision, Recall, F1-score (macro & samples)  
  Hamming Loss, Jaccard Similarity, AUC-ROC  
- **Statistical Tests**: Paired t-tests  
- **Error Analysis**: Confusion matrices and case-wise inspection  

---

## � Expected Outputs & Contributions
- Robust mBERT performance on English multilabel classification  
- **12–18% F1-score improvement** on low-resource African languages via transfer learning  
- Enhanced generalisability through data augmentation  
- **Public release of**:  
  - Emotion classification pipeline  
  - Language-specific adapter modules  
  - Preprocessing/evaluation scripts  
- Applications in mental health monitoring, education analytics, and cultural insight tools  

---

## 🛠️ How to Run
```bash
# Clone repository
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>

# (Optional) Set up virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
# For Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Evaluate model
python evaluate.py
```
🧪 Run in Google Colab

```bash

# Step 1: Clone repository
!git clone https://github.com/<your-org>/<your-repo>.git
%cd <your-repo>

# Step 2: Install dependencies
!pip install -r requirements.txt

# Step 3: Load BRIGHTER dataset
from datasets import load_dataset
dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories")
print(dataset["train"][0])  # View sample

# Step 4: Fine-tune model
!python train.py

# Step 5: Evaluate
!python evaluate.py

```
[▶️ Open in Colab](https://colab.research.google.com)

---

## 📚 References
1. [BRIGHTER Dataset on Hugging Face](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories)  
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805)  
3. Pfeiffer, J., Kamath, A., Rücklé, A., Cho, K., & Gurevych, I. (2021). [**AdapterFusion: Non-Destructive Task Composition for Transfer Learning**](https://arxiv.org/abs/2005.00247)
