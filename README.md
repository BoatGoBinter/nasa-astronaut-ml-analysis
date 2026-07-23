# NASA Astronaut Selection & Career Trends

This is a data science project analyzing how NASA astronaut demographics have shifted over time, and whether educational or military background predicts career outcomes like mission count and time spent in space. 

## Research Questions

1. **Selection trends** — How do birthplace, age, gender, and other demographic characteristics relate to the likelihood of being selected into the astronaut program, and how have these patterns shifted across eras of NASA history?
2. **Career outcomes** — Can an astronaut's background (undergraduate major, military branch) predict measurable career outcomes like total mission count or time spent in space?

## Data

Two publicly available datasets were merged into a single analytical dataset:

- **International Astronaut Database** (CSIS Aerospace Security Project) — 570 astronauts globally, including country, gender, flights, and total flight hours.
https://aerospace.csis.org/data/international-astronaut-database/
- **NASA Astronaut Database** (via Kaggle, sourced from NASA's public records) — 357 NASA-associated astronauts, including birth year, alma mater, undergraduate/graduate major, and military background.
https://www.kaggle.com/datasets/nasa/astronaut-yearbook

After merging, deduplicating, and cleaning, the final analytical dataset covers **357 NASA astronauts**.

## Methods

- Data cleaning and merging across inconsistent name formats and mission-count definitions
- Feature engineering: simplified military branch categories, binary STEM/non-STEM major indicator, grouped birthplace categories
- Descriptive statistics and visualization (share of women by decade, birthplace distribution, major/branch breakdowns)
- Logistic regression and logistic Lasso (with cross-validation) to model pre- vs. post-1990 selection
- Linear regression, Lasso regression, PCA, and bootstrap resampling to test whether background predicts flight hours or number of missions

## Key Findings

- **Selection has shifted over time.** The share of women among selected astronauts rose from 0% in the 1950s–60s to ~21–22% in the 1990s–2000s (logistic regression: odds of selection being female increase ~1.75x per decade). Average age at selection has also increased (~1.1 years older per decade). A logistic Lasso model using birth year, gender, and birthplace achieved an AUC of 0.96 predicting pre- vs. post-1990 selection — driven overwhelmingly by birth year, with birthplace playing a secondary role and gender contributing little once cohort is accounted for.
- **Background weakly predicts career outcomes.** While descriptive stats show Army astronauts logging more flight hours on average and STEM majors dominating overall, linear regression, bootstrap resampling, and Lasso regression all show these effects are weak, unstable, and often indistinguishable from zero. Neither undergraduate major nor military branch reliably predicts total mission count or time spent in space.

## Repository Structure
```
.
├── merge_astronaut_datasets.py     # Merges the two source datasets into one
├── merged_astronauts_data.csv      # Cleaned, merged dataset used for analysis
├── nasa_astronaut_model_testing.ipynb  # Feature engineering, modeling, and visualization
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

## Limitations & Future Work

This analysis only observes astronauts who were *selected*, so absolute selection probabilities can't be estimated — only how the composition of selected astronauts has shifted. Future work could incorporate data on the broader applicant pool, training performance, and mission-assignment criteria, and explore more flexible models like random forests or gradient boosting.
