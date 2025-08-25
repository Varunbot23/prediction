# README – Student Dropout Risk Prediction Project

## Index of Files/Directories
1. **app.py**  
   - Streamlit-based web application for predicting student dropout risk using a Random Forest model.  
   - Provides interactive inputs, probability-based risk scoring, personalised review suggestions, and batch prediction via CSV.

2. **random_forest_model.pkl**  
   - Serialized Random Forest model trained on the dataset.  
   - Used by `app.py` for predictions.

3. **requirements.txt**  
   - List of Python dependencies needed to run the project:
     ```
     streamlit==1.48.1
     numpy==1.26.4
     pandas==2.2.2
     scikit-learn==1.2.0
     joblib==1.4.2
     ```

4. **Students Performance Dataset (1).xls**  
   - Dataset of 5,000 anonymised student records with 24 academic, behavioural, and demographic features.  
   - Used for training, EDA, and model evaluation.

5. **Team RED.ipynb**  
   - Jupyter Notebook containing data preprocessing, feature engineering, model training, evaluation, and explainability.

6. **Group_Final_Report_Team(RED).pdf**  
   - Final MSc project report with methodology, results, discussion, and deployment notes.

---

## Dataset Notes
- The dataset is anonymised and publicly available for educational purposes.  
- No personally identifiable information (PII) is included.  
- Demographic/socio-economic features (e.g., parental education, family income) may risk bias; redistribution may be restricted for ethical reasons.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
