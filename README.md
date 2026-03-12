# AI_Dependency_Awareness

This project investigates the concept of **AI Dependency**, exploring how reliance on AI affects user performance, self-confidence, and awareness across different professional and academic roles.

## 📌 Project Overview
As AI becomes a standard tool in creative and technical workflows, understanding the balance between human skills and machine assistance is crucial. This study tracks 500 participants—including Students, Professionals, and Freelancers—to measure task completion scores and confidence levels with and without AI assistance.

## 📊 Dataset Description
The dataset contains 500 entries with several key features:
* **User Demographics:** Age, Role (Student, Freelancer, Professional).
* **Behavioral Data:** Primary task type (Coding, Writing, Research, etc.) and daily AI usage frequency.
* **Performance Metrics:** Task completion scores (0-10) both with and without AI.
* **Psychological Metrics:** Confidence levels (0-10) with and without AI, and self-perceived dependency levels.

## 🛠️ Methodology & Preprocessing
To ensure high-quality analysis, the following steps were performed:
1.  **Data Cleaning:** Renamed long descriptive columns for easier coding (e.g., `score_no_ai`, `score_with_ai`, `awareness_ai`).
2.  **Ordinal Mapping:** Categorized `awareness_of_ai_dependency` into a logical order (Low < Medium < High).
3.  **Feature Engineering:**
    * `ai_impact`: Calculated the direct performance improvement (With AI Score - Without AI Score).
    * `confidence_boost`: Measured the increase in user confidence attributable to AI usage.

## 📈 Key Insights
* **Performance Shift:** AI assistance increased average task completion scores significantly, highlighting its role as a productivity multiplier.
* **User Distribution:** Students were found to have the highest daily AI usage frequency.
* **Confidence Trends:** A measurable "Confidence Boost" was observed across all roles, though the correlation between confidence and actual score improvement varies by task type.

## 🤖 Predictive Modeling
The project utilizes **Linear Regression** to predict final task scores based on user behavioral patterns and demographics.
* **Model Goal:** Predict continuous performance scores with AI.
* **Evaluation Metrics:** Mean Absolute Error (MAE) and $R^2$ Score.

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## 🚀 How to Use
1. Ensure the dataset `ai_dependency_awareness_dataset.csv` is in the project directory.
2. Run the notebook `AI_Dependency_Awareness.ipynb` to execute the data cleaning, Exploratory Data Analysis (EDA), and model training.
