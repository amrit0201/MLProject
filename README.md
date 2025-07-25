#  Mental Health in Tech Survey — Treatment Prediction

This project uses data from the **Mental Health in Tech Survey** to predict whether individuals have sought treatment for mental health issues based on demographic and workplace-related factors. The final model is deployed using **Streamlit**.

---

##  Dataset Overview

The dataset includes:
- Demographic details (age, gender, country)
- Workplace attributes (company size, benefits, wellness programs, remote work)
- Mental health-related variables (family history, interference with work, etc.)

Data source: [Mental Health in Tech Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)

---

##  Data Cleaning & Preprocessing

- **Dropped Columns:** `comments`, `Timestamp`, `state`
- **Age:** Filtered outliers (e.g., age < 18 or > 100), filled missing values with median
- **Gender:** Recoded to standard categories
- **Self-Employed:** Filled missing values with mode
- **Work Interfere:** Filled missing values with "Unknown"
- **Encoding:**
  - Ordinal variables: Label encoded
  - Categorical variables: One-hot encoded (e.g., gender, country)
  - Age: Scaled using MinMaxScaler

---

##  Exploratory Data Analysis (EDA)

###  Target Variable
- **Balanced classes** of people who sought treatment vs. those who didn’t

###  Key EDA Insights (Visual + Statistical):

#### Numerical (Age):
- Mann-Whitney U test shows age distribution is significantly different across treatment groups *(p = 0.0125)*.

#### Categorical (Countplots):
- **Gender:** Females more likely to seek treatment.
- **Country:** US respondents most likely to seek treatment.
- **Family History:** Strong positive association.
- **Work Interfere:** More interference → higher treatment rates.
- **Company Size:** 6–25 & 500–1000 employee ranges show higher treatment.
- **Remote Work:** Slightly more treatment seekers.
- **Mental Health Benefits & Programs:** Strong positive correlation.

#### Chi-Square Tests:
Significant associations with treatment found in:
- Gender, Family History, Work Interfere
- Mental Health Consequence, Anonymity, Leave
- Care Options, Benefits, Observed Consequence
- Country, Mental vs Physical health treatment parity
- Wellness Programs, Seeking Help, Coworker Support

*No significant relationship*: Number of employees, Remote work, Supervisor support, Self-employment.

---

##  Machine Learning Modeling

###  Preprocessing
- Dropped statistically insignificant variables
- Performed train-test split
- Encoded categorical variables appropriately
- Scaled numerical features

###  Models Trained
- Logistic Regression
- XGBoost
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine

###  Model Evaluation
- Cross-validation was used for performance comparison
- **Hyperparameter tuning** done via GridSearchCV
- **Random Forest** outperformed others with an accuracy of **84.7%**

---

##  Deployment

The final model was deployed using **Streamlit**, offering a user-friendly interface for treatment prediction based on user input.

---

##  How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/amrit0201/MLProject.git
cd MLProject

### 2. Install the required packages

```bash
pip install -r requirements.txt

### 3.Run the Streamlit App 

```bash
streamlit run app.py

