# Income Classification and Customer Segmentation Report

Income Classification and Customer Segmentation Report
JPMC Take-Home Project
## 1. Executive Summary
This project uses Census-style demographic, employment, household, and financial-profile data to predict whether an individual earns more than $50,000. The dataset is highly imbalanced: only about 6.3% of records belong to the high-income class. Because of this imbalance, accuracy alone is not a reliable model evaluation metric.
After exploratory data analysis, feature engineering, and model training, the final XGBoost model achieved strong ranking performance:
| Metric | Result |
| --- | --- |
| Test ROC-AUC | 0.948 |
| Test PR-AUC | 0.655 |
| Test Precision for > $50K | 0.528 |
| Test Recall for > $50K | 0.670 |
| Test F1 for > $50K | 0.591 |
| Selected Threshold | 0.81 |

The final model was trained with both Census sampling weights and class-imbalance weights. The probability threshold was tuned on the validation set using positive-class F1 rather than relying on the default 0.5 threshold.
In addition to classification, customer segmentation was performed using engineered customer-profile features. Four clustering methods were compared: K-means, Gaussian Mixture Model, BIRCH, and Agglomerative Clustering. BIRCH achieved the highest silhouette score on the sampled data, while K-means was selected as the final segmentation model due to its scalability and interpretability.
## 2. Dataset Overview
The original dataset contained 199,523 rows and 42 columns. After removing exact duplicate rows, the final cleaned dataset contained 196,294 rows.
The target variable was converted into a binary label:
0: income <= $50,000
1: income > $50,000
| Income Class | Count | Percentage |
| --- | --- | --- |
| <= $50,000 | 183,912 | 93.69% |
| > $50,000 | 12,382 | 6.31% |

This shows a strong class imbalance, so model evaluation focused on PR-AUC, precision, recall, and positive-class F1 rather than accuracy alone.
## 3. Data Cleaning
Details can be find in notebooks/EDA.ipynb
Duplicate removal: Exact duplicate rows were removed from the dataset.
Missing value handling: The only column with missing values was hispanic_origin. Missing values were filled with "?" to match the dataset’s existing unknown-value convention.
Target encoding: The original income label was converted into a binary target variable.
Sampling weight separation: The Census weight column was removed from the feature set and stored separately as a sample weight. This avoids treating survey sampling weight as a predictive customer attribute.
## 4. Exploratory Data Analysis Findings
Details can be find in notebooks/EDA.ipynb
### 4.1 Class Imbalance
The high-income class is rare, representing only about 6.3% of the data. A model could achieve high accuracy by predicting most records as low income while still performing poorly on the business objective: identifying high-income individuals.
### 4.2 Age
Age shows a nonlinear relationship with income. Younger individuals and children are much less likely to be in the high-income group, while working-age adults are more represented among high-income records. Therefore, age was converted into age buckets instead of relying only on the raw numeric age.
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/7be139b6-eff4-4d35-abb4-6c8d6f6c46b5" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/72f8c2eb-70e3-4c20-80fc-2bdf1f624d4f" />
Fig.1 Age distribution (upper) and boxplot (lower) by income group. 
### 4.3 Work Intensity
weeks_worked_in_year is one of the most important features. The low-income group contains many records with 0 weeks worked, while the high-income group is heavily concentrated around 52 weeks worked. This motivated the creation of work-intensity features such as worked_none_year, worked_full_year, and bucketed weeks-worked indicators.
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/566f89c5-47a5-4ab2-8f99-9f0150fd95f2" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/056e859f-279c-4b5d-91db-7fc553825374" />
Fig.2 Weeks worked in year distribution (upper) and boxplot (lower) by income group. 
### 4.4 Money and Investment Features
Money-related columns such as wage_per_hour, capital_gains, capital_losses, and dividends_from_stocks are highly zero-inflated and right-skewed. Both binary indicators and log-transformed versions were created so the model can capture whether a person has an income/investment source and the magnitude of that value.
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/5a5c89d0-c97a-47b8-a615-f5720c4dd0f7" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/6b9d65a7-967b-41c1-a74d-59cb934f4a92" />
Fig.3 Money and Investment example:  wage per hour distribution (upper) and boxplot (lower) by income group. 
### 4.5 Categorical Features
Several categorical features showed strong separation between income groups, including education, marital_stat, class_of_worker, major_industry_code, major_occupation_code, full_or_part_time_employment_stat, tax_filer_stat, and detailed_household_summary_in_household. Other features were excluded because they were dominated by “Not in universe” / “?”, overly sparse, or redundant with stronger features.
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/6e895725-601b-4b07-aaa9-f141ac9cb1fd" />
Fig.4 Categorical feature example:  veterans Benefits classes by income group. 
## 5. Sensitive Features
To reduce fairness and compliance risk, the following sensitive or sensitive-adjacent variables were excluded from final model training:
race
sex
hispanic_origin
country_of_birth_self
These variables may contain predictive signal, but using them directly in an income prediction model could introduce fairness concerns, especially in a financial-services context. They should be retained only for exploratory analysis or post-model fairness checks.
A stricter production version should also consider excluding or carefully auditing citizenship, country_of_birth_father, and country_of_birth_mother.
## 6. Feature Engineering
The final feature set contained 122 engineered features.
### 6.1 Age Features
The raw age variable was converted into one-hot encoded age buckets: 0-17, 18-24, 25-34, 35-44, 45-54, 55-64, and 65+. This captures nonlinear income patterns across life stages.
### 6.2 Money and Investment Features
For each major money-related feature, two engineered features were created: a binary indicator showing whether the value is positive and a log-transformed feature to reduce the impact of extreme values. Examples include has_dividends_from_stocks and log_dividends_from_stocks.
### 6.3 Weeks Worked Features
From weeks_worked_in_year, features were created for worked_none_year, worked_full_year, and bucketed work ranges. The original weeks_worked_in_year was also standardized and retained because it was highly predictive.
### 6.4 Industry and Occupation
The detailed industry and occupation recode features were excluded to reduce redundancy and sparsity. Instead, the broader major industry and occupation categories were one-hot encoded. Rare categories such as Armed Forces were removed or grouped due to extremely low frequency.
### 6.5 Grouped Categorical Features
Several categorical variables were grouped into broader categories before one-hot encoding. This reduced sparsity and improved interpretability. Examples include class_of_worker_grouped, marital_stat_grouped, education_grouped, full_or_part_time_employment_stat_grouped, family_members_under_18_grouped, and detailed_household_summary_in_household_grouped.
## 7. Features Excluded from Final Model
| Feature | Reason |
| --- | --- |
| detailed_industry_recode | redundant with major_industry_code |
| detailed_occupation_recode | redundant with major_occupation_code |
| enroll_in_edu_inst_last_wk | mostly “Not in universe”; overlaps with age and education |
| reason_for_unemployment | overwhelmingly “Not in universe” |
| region_of_previous_residence | sparse and redundant with migration features |
| state_of_previous_residence | sparse and dominated by “Not in universe” |
| detailed_household_and_family_stat | overly granular; overlaps with household summary features |
| migration_code_change_in_msa | redundant with broader migration features |
| migration_code_move_within_reg | redundant migration signal |
| migration_prev_res_in_sunbelt | mostly unknown or not applicable |
| country_of_birth_father | high-cardinality and sensitive-adjacent |
| country_of_birth_mother | high-cardinality and sensitive-adjacent |
| fill_inc_questionnaire_for_veteran's_admin | overwhelmingly “Not in universe” |

## 8. Model Training Approach
Details can be find in notebooks/feature_engineering_and_modeling.ipynb
The data was split into train, validation, and test sets. The Census sampling weight was used as sample_weight rather than as a model feature. To address class imbalance, the training weight combined Census sampling weight and class-balanced weight.
| Split | Rows | Percentage |
| --- | --- | --- |
| Train | 125,628 | 64% |
| Validation | 31,407 | 16% |
| Test | 39,259 | 20% |

## 9. XGBoost Model
Details can be find in notebooks/feature_engineering_and_modeling.ipynb
XGBoost was selected as the final classification model because it is well suited for this project’s data structure and modeling objective. The dataset contains a mix of numerical, categorical, sparse, and highly nonlinear features, including age, work intensity, education, occupation, tax filing status, and investment-related variables. XGBoost can capture nonlinear relationships and feature interactions more effectively than a linear model such as Logistic Regression.

This was important because EDA showed several nonlinear patterns. For example, high-income individuals were concentrated in certain age ranges, were more likely to work a full year, and were more likely to have capital gains or dividend income. These relationships are not always linear, and tree-based boosting models can learn threshold-based patterns such as `weeks_worked_in_year = 52`, `capital_gains > 0`, or specific education and occupation groups.

XGBoost also performs well on imbalanced classification problems when combined with appropriate sample weighting and threshold tuning. In this project, the high-income class represented only about 6.3% of the dataset, so I optimized the model using PR-AUC and tuned the decision threshold using positive-class F1 instead of relying on accuracy or the default 0.5 threshold.

Compared with Logistic Regression, XGBoost provided stronger predictive performance while still remaining interpretable through SHAP analysis. SHAP values were used to identify the most important feature drivers, such as weeks worked, age bucket, tax filer status, education, occupation, and investment-income features. Therefore, XGBoost offered a strong balance between predictive power, flexibility, and interpretability for this income classification task.
### 9.1 Baseline XGBoost
A baseline weighted XGBoost model was trained with n_estimators = 800, learning_rate = 0.03, max_depth = 4, and eval_metric = aucpr.
| Metric | Value |
| --- | --- |
| Validation ROC-AUC | 0.949 |
| Validation PR-AUC | 0.650 |

### 9.2 Hyperparameter Tuning
Optuna was used to tune XGBoost hyperparameters using validation PR-AUC as the objective. The best validation PR-AUC was 0.653.
| Parameter | Best Value |
| --- | --- |
| learning_rate | 0.0308 |
| max_depth | 5 |
| min_child_weight | 10 |
| subsample | 0.8876 |
| colsample_bytree | 0.8472 |
| reg_lambda | 1.6330 |
| reg_alpha | 2.9770 |

## 10. Threshold Tuning
Details can be find in notebooks/feature_engineering_and_modeling.ipynb
Because the high-income class is rare, the default 0.5 threshold is not necessarily optimal. The validation set was used to choose the probability threshold that maximized positive-class F1.
| Metric | Value |
| --- | --- |
| Threshold | 0.81 |
| Validation Precision | 0.544 |
| Validation Recall | 0.676 |
| Validation F1 | 0.603 |

A top-decile targeting analysis was also performed. When targeting the top 10% highest-probability records, the model reached a probability cutoff of 0.742, Precision@Top10% of 0.465, and Recall@Top10% of 0.741.
## 11. Final Test Performance
Details can be find in notebooks/feature_engineering_and_modeling.ipynb
The final tuned XGBoost model was evaluated once on the held-out test set using the selected validation threshold.
| Metric | Test Result |
| --- | --- |
| ROC-AUC | 0.948 |
| PR-AUC | 0.655 |
| Precision for > $50K | 0.528 |
| Recall for > $50K | 0.670 |
| F1 for > $50K | 0.591 |

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| <= $50K | 0.98 | 0.96 | 0.97 |
| > $50K | 0.53 | 0.67 | 0.59 |

The model performs very well in ranking individuals by high-income likelihood, as shown by the high ROC-AUC and strong PR-AUC for an imbalanced dataset. The tuned threshold provides a reasonable tradeoff between precision and recall for the high-income class.
## 12. Model Interpretation with SHAP
Details can be find in notebooks/feature_engineering_and_modeling.ipynb
SHAP was used to interpret the final XGBoost model. The most important individual features included:
scaled_weeks_worked_in_year
age_bucket_0_17
tax_filer_stat_nonfiler
detailed_household_summary_in_household_grouped
scaled_num_persons_worked_for_employer
scaled_dividends_from_stocks
education_grouped_bachelors
education_grouped_less_than_high_school
education_grouped_graduate_degree
scaled_capital_gains
worked_full_year
marital_stat_grouped_married
major_occupation_code_adm_support_including_clerical
<img width="792" height="590" alt="image" src="https://github.com/user-attachments/assets/1ce11448-9206-4084-9793-5f05355597b3" />
Fig.5 Feature importance ranking indicated by SHAP. 
| Feature Group | Interpretation |
| --- | --- |
| Age bucket | Life stage is highly predictive |
| Weeks worked | Work intensity is one of the strongest signals |
| Tax filer status | Filing status strongly separates income groups |
| Education | Higher education levels are associated with higher income |
| Major occupation | Occupation type strongly affects income likelihood |
| Household summary | Household role/status contributes predictive signal |
| Major industry | Industry group adds employment context |
| Marital status | Marital/family structure is associated with income differences |
| Dividends / capital gains | Investment income is informative for high-income prediction |

Overall, the model relies most heavily on work intensity, age/life stage, tax filing status, education, occupation, household structure, and investment-related features.
## 13. Customer Segmentation
Details can be find in notebooks/feature_engineering_and_modeling.ipynb
Customer segmentation was performed using the engineered feature matrix. The income label was not used to create the clusters, but it was used afterward to profile the segments. A 20,000-row sample was used for visualization and method comparison. The high-dimensional feature matrix was projected into lower-dimensional space using TruncatedSVD.
| Method | Silhouette Score | Number of Clusters |
| --- | --- | --- |
| BIRCH | 0.430 | 4 |
| K-means | 0.315 | 4 |
| Agglomerative Clustering | 0.298 | 4 |
| Gaussian Mixture Model | 0.269 | 4 |
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/f4d14cbc-1284-4b67-9543-f013d4c748e5" />
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/e84745a7-c8b2-4661-8df9-55dde872fa53" />
Fig.6 K-means (upper) and BIRCH (lower) customer segmentation visualized in reduced feature space 

Although BIRCH had the highest silhouette score, K-means was selected as the final segmentation model because it is scalable, stable, and easier to explain in a business setting.
## 14. Segment Profiles
Details can be find in notebooks/feature_engineering_and_modeling.ipynb
The final K-means model produced 4 customer segments. Segment names are business-friendly interpretations based on the segment profile statistics, not labels generated directly by the algorithm.
### Segment 0: Older Non-Working / Low-Income Segment
| Profile Attribute | Value |
| --- | --- |
| Average age | 54.9 |
| High-income rate | 1.4% |
| Average weeks worked | 1.4 |
| Worked no weeks | 88.8% |
| Top education | High school graduate |
| Top class of worker | Not in universe |
| Top marital status | Married-civilian spouse present |

Interpretation: This segment appears to represent older individuals largely outside the active labor force. The high-income rate is very low, and work intensity is minimal.
### Segment 1: Active Working Mainstream Segment
| Profile Attribute | Value |
| --- | --- |
| Average age | 38.8 |
| High-income rate | 11.3% |
| Average weeks worked | 46.8 |
| Worked full year | 72.2% |
| Top class of worker | Private |
| Top industry | Retail trade |
| Top occupation | Administrative support / clerical |
| Top education | High school graduate |

Interpretation: This segment represents active workers, mostly in private-sector roles. The high-income rate is above the overall dataset average but still moderate.
### Segment 2: Children / Young Non-Working Segment
| Profile Attribute | Value |
| --- | --- |
| Average age | 8.1 |
| High-income rate | 0.0% |
| Average weeks worked | 0.3 |
| Worked no weeks | 97.6% |
| Top education | Children |
| Top class of worker | Not in universe |
| Top marital status | Never married |

Interpretation: This segment clearly represents children or very young individuals with no labor-force participation. It has essentially no high-income potential under the current income definition.
### Segment 3: Investment / Capital-Loss High-Income Segment
| Profile Attribute | Value |
| --- | --- |
| Average age | 44.0 |
| High-income rate | 30.1% |
| Average weeks worked | 41.6 |
| Worked full year | 67.2% |
| Has capital losses | 100% |
| Has dividends from stocks | 25.2% |
| Segment-level high-income rate | Highest among the four segments |

Interpretation: This segment appears to capture financially active individuals with investment-related activity, especially capital losses and dividends. It has the highest high-income rate and may represent a strong target group for financial products.
## 15. Business Recommendations
### Use Model Scores for Targeted Outreach
The model should be used as a ranking tool to prioritize individuals with high predicted probability of earning more than $50,000. For limited campaign budgets, the business can target the top 10% highest-scored records.
### Prioritize High-Value Segments
Segment 3 has the highest high-income rate and strong investment-related characteristics. This group may be appropriate for wealth management, investment advisory, premium financial services, brokerage, or retirement planning campaigns.
### Avoid Low-Value Targeting Groups
Segments 0 and 2 have very low high-income rates. These groups should receive lower priority for high-income-focused campaigns, though they may still be relevant for different products or life-stage strategies.
### Use Sensitive Features Only for Fairness Checks
Sensitive variables should not be used for model prediction. However, they can be used after modeling to evaluate whether model performance varies meaningfully across demographic groups.
## 16. Limitations
Income threshold is binary: The model predicts whether income is above or below $50,000, not actual income amount.
Dataset is Census-style, not transaction-level: The dataset does not include balances, transaction history, product ownership, or behavioral engagement data.
Class imbalance remains challenging: The high-income class is rare. Even with strong ROC-AUC, precision for the high-income class remains moderate.
Some features may act as proxies: Even after excluding direct sensitive variables, other features may partially proxy for demographic differences. A fairness audit is recommended before production use.
Segmentation is profile-based: The segments should be validated with downstream campaign outcomes before operational use.
## 17. Conclusion
The final XGBoost model provides strong predictive ranking performance for identifying high-income individuals in an imbalanced Census-style dataset. The model achieved a test ROC-AUC of 0.948 and PR-AUC of 0.655, with a tuned threshold that balanced precision and recall for the rare high-income class.
SHAP analysis showed that the strongest predictors were work intensity, age, tax filer status, education, occupation, household structure, and investment-related features. Customer segmentation further identified meaningful customer groups, including an active working segment and a high-value investment-related segment with the highest high-income rate.
Overall, the project demonstrates a complete workflow from EDA and feature engineering to imbalance-aware modeling, threshold tuning, model interpretation, and customer segmentation.
