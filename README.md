1. Overview
Hematologic diseases such as anemia, leukemia, and thrombocytopenia are major contributors of morbidity worldwide and are regularly diagnosed through Complete Blood Count (CBC) tests. Nonetheless, CBC interpretation is subjective and susceptible to inter-observer variation, especially in resource-constrained facilities without a specialist hematologist.
This work presents a cohesive deep learning (DL) and explainable artificial intelligence (XAI) framework for automated multi-class classification of fifteen hematological disorders using common numerical CBC parameters. Sourced from a clinically validated dataset of 7,196 patient records with 16 numeric features, the proposed framework bridges the gap between black-box deep learning models and clinical transparency through multi-tiered XAI methods (Gradient Saliency, Permutation Importance, SHAP, and LIME).
2. Methodology & Pipeline
• Data Preprocessing:  Evaluated on 7,196 patient records featuring 16 numeric blood parameters and 15 target classes. Steps include deduplication, standard Z-score scaling, and stratified 80:20 training-to-testing partitioning. Features were reshaped into 3D sequential representations (samples × features × 1) for 1D convolution and recurrent sequence modeling.
• Deep Learning Architectures:  Comparative evaluation of four standalone deep learning baselines (ANN, 1D-CNN, SimpleRNN, LSTM) alongside three sequential hybrid architectures (CNN + BiLSTM, RNN + BiLSTM, and CNN + RNN + BiLSTM).
• Explainable AI (XAI) Framework:  Integration of built-in gradient saliency, model-agnostic permutation feature importance, SHAP (GradientExplainer) for global and per-class attribution, and LIME for patient-specific linear surrogate explanations.
[Raw CBC Data (16 Numeric Parameters)]
                 │
                 ▼
[Preprocessing, Standardization & Feature Reshaping]
                 │
                 ▼
[Deep Learning & Hybrid Models: 1D-CNN / CNN-RNN-BiLSTM]
                 │
                 ▼
[15-Class Hematological Diagnosis Output]
                 │
                 ▼
[Explainable AI Validation: SHAP, LIME, Gradient Saliency]
                 │
                 ▼
[Transparent Clinical Decision Support]
3. Key Findings & Experimental Results
The table below summarizes the quantitative evaluation across all seven evaluated deep learning and hybrid architectures on the 20% held-out test set:
Model Architecture	Accuracy (%)	Precision (%)	Recall (%)	F1-Score (%)
1D-CNN (Best Standalone)	94.44%	94.47%	94.44%	94.42%
CNN + RNN + BiLSTM (Best Hybrid)	93.75%	94.36%	93.75%	93.90%
CNN + BiLSTM	92.43%	92.79%	92.43%	92.49%
ANN (Fully Connected)	89.44%	90.52%	89.44%	89.66%
SimpleRNN	82.42%	86.94%	82.42%	83.02%
RNN + BiLSTM	82.00%	87.10%	82.00%	83.10%
LSTM	75.54%	81.74%	75.54%	76.65%

• Dominant Features:  Across permutation importance, gradient saliency, and SHAP analyses, the primary predictive biomarkers were Red Cell Distribution Width (RDW), Hemoglobin (Hb), White Blood Cell count (WBC), Lymphocyte percentage (LYMP), and Platelet count.
• Clinical Interpretability:  In Chronic Leukemias, high WBC values combined with low Hb pushed predictions toward leukemic leukocytosis with coexisting anemia. In Anemia of Chronic Disease, low Hb and reduced MCV/RBC prevailed, aligning with established medical hematological literature.
4. Dataset Information
Sourced from the Clinically Validated CBC Dataset of 7,196 Samples for Hematological Disorder Analysis (Mendeley Data / NLM Dataset Catalogue):
• Features (16 Attributes):  Gender, Age, Hb, RBC, WBC, PLATELETS, LYMP, MONO, HCT, MCV, MCH, MCHC, RDW, PDW, MPV, PCT.
• Target Diagnoses (15 Classes):  Anemia of Chronic Disease (19.44%), Normocytic Anemia (17.57%), Bacterial Infections (16.12%), Thrombocytosis (8.63%), Infectious Mononucleosis (6.56%), Beta-Thalassemia Minor (6.00%), Normal (5.77%), Viral Infections (4.49%), Leukopenia (3.24%), Microcytic Anemia (3.22%), Iron Deficiency (3.20%), Leukocytosis (2.38%), Thrombocytopenia (1.61%), Chronic Leukemias (1.38%), Polycythemia Vera (0.40%).
5. How to Run (Step-by-Step Execution Guide)
This project is fully executable and reproducible via Google Colab or local Jupyter Notebook environments. Follow the steps below:
Step 1: Environment Setup & Dependencies
Prepare a Python 3.10+ environment and install the required machine learning and explainability libraries:
# Clone the repository
git clone https://github.com/AsmanaYasmin/hematological-disorder-classification-xai
cd hematological-disorder-classification-xai

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow shap lime

Step 2: Running via Jupyter Notebook / Google Colab
Open the interactive Jupyter Notebook `BDCBC7196_Analysis.ipynb` in Google Colab or JupyterLab:
1. Mount your Google Drive or upload the dataset file (`BDCBC7196_Hematology_Dataset.csv`) to the working directory.
2. Execute the notebook cells sequentially from top to bottom:
   • Section 1 (Imports): Loads scientific computing, modeling, and XAI packages.
   • Section 2 (Data Loading & EDA): Inspects distributions, class shares, and correlation heatmaps.
   • Section 3 (Preprocessing): Executes deduplication, label encoding, and stratified 80:20 train-test split.
   • Section 4 (Model Training): Trains 1D-CNN and hybrid architectures with Adam optimizer and early stopping.
   • Section 5 (Model Evaluation): Evaluates confusion matrices, weighted precision, recall, and F1-score.
   • Section 6 (Explainable AI): Generates gradient saliency, permutation importance, SHAP summary beeswarm, and LIME local linear explanations.
Step 3: Verification & Reproducibility
The training scripts automatically validate performance against the held-out test fold (1,439 test samples) and generate saved visual outputs for confusion matrices, SHAP waterfall plots, and LIME patient attributions in the `results/` folder.
