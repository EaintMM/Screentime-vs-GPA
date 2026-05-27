# 📊 Screen Time vs. Academic Performance
### A Data Science Study on Student Habits at Siam University

> **Does spending more time on your phone actually hurt your grades?**  
> The answer is more nuanced — and more interesting — than you'd expect.

---

## 🔍 Project Overview

This end-to-end data science project investigates the relationship between students' screen time habits and their academic performance (CGPA). Rather than simply confirming the assumption that "more screen time = worse grades," the analysis uncovers that **how** you use your screen matters far more than **how long**.

Built as a **128-352 Data Science Term Project** at Siam University (Team No. 4), the project covers the full pipeline: survey design → data collection → cleaning → exploratory analysis → diagnostic analysis → machine learning → interactive dashboard.

---

## 🚀 Live Demo

> 🔗 **[Launch the Interactive Dashboard](#)** 

The dashboard is built with Streamlit and lets you explore every finding interactively — filter by demographic, drill into charts, and follow the story section by section.

---

## 💡 Key Findings

| Question | Finding |
|---|---|
| Does high screen time hurt GPA? | Weak correlation of **-0.13** — raw volume is a poor predictor |
| Which activity is worst? | **Gaming & Streaming** users average 0.29 GPA points below Study users |
| Is late-night use worse? | **Yes** — it doubles the risk of sleep deprivation, which drops avg GPA to 3.32 |
| Who is most at risk? | Students with **high in-class distraction** and low compensatory study hours |

---

## 🗂️ Project Structure

```
📁 Project/
├── README.md              
├── project.py              # Streamlit dashboard (main app)
├── SyntheticDataset.xlsx   # Synthetic dataset (mirrors real data distributions)
└── requirements.txt
```

> **Note on Data:** The original survey data was collected from 129 Siam University students under an anonymity guarantee. To respect participant privacy and comply with Thailand's PDPA, this repository uses a **synthetic dataset** generated with [SDV (Synthetic Data Vault)](https://sdv.dev/) that preserves all statistical properties and relationships of the original data. The real findings are documented in the presentation slides.

---

## 🧱 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

- **Data Collection:** Google Forms survey with optional screen time screenshot upload
- **Data Cleaning:** Pandas + manual validation
  - Removed rows with missing CGPA or Screen Time (key variables)
  - Imputed remaining missing values: mode for categorical, mean for numerical
  - Standardized `Major` field — fixed abbreviations, capitalization inconsistencies, and grouped into uniform categories
  - Cross-validated self-reported screen time values against participants' actual device screenshots for accuracy
  - Converted all screen time entries to a consistent unit (hours)
- **EDA & Visualization:** Plotly Express, Plotly Graph Objects
- **Machine Learning:** Scikit-learn Decision Tree Classifier
- **Dashboard:** Streamlit (multi-section sidebar navigation)
- **Synthetic Data:** SDV GaussianCopulaSynthesizer

---

## 📖 Dashboard Sections

1. **Data Overview** — Dataset pulse: key metrics at a glance
2. **The Baseline (EDA)** — Scatter plots and correlation heatmap busting the "screen time = bad grades" myth
3. **The Plot Twist** — Purpose over volume: what you do on your screen matters more than how long
4. **Demographics** — The Sleep Penalty, The Gender Efficiency Gap, and The All-Nighter Effect
5. **Actionable Insights** — The Classroom Immunity Paradox and The Freshman Illusion
6. **Predictive Modeling** — Decision Tree Classifier attempting to predict high performers (CGPA ≥ 3.5)

---

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone 
cd Screentime-vs-GPA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run ScreenGpa.py
```

### requirements.txt
```
streamlit
pandas
plotly
scikit-learn
matplotlib
openpyxl
```

---

## 🤔 Limitations & Learnings

- **Sample size (n=129):** The Decision Tree model achieved 45.5% accuracy — below the ~57% baseline of always predicting the majority class. This is an honest result: with this sample size and self-reported survey data, behavioral habits alone don't carry enough signal to reliably predict GPA. More data and richer features would be needed. Recognizing *why* a model underperforms is as valuable as building one that works.
- **Self-reported data:** Screen time values were cross-validated against uploaded device screenshots where provided, but self-reporting bias remains a factor.
- **Single university:** Findings reflect Siam University students and may not generalize broadly.

---

## 👥 Team

**Team No. 4** — Data Science, Siam University

---

## 📄 License

This project is for academic and portfolio purposes. The synthetic dataset is freely usable for educational use.
