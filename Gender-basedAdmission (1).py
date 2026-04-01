#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"> $Investigating$  $Gender$  $Bias$  $in$  $Graduate$  $Admissions$  $at$  $UC$  $Berkeley$  $(1973)$  <h3>

# 
# <h1 align="center">
#  $Mukaila$ $Rafiu$
#     </h1>
# 

# <h1 align = "center">
#     $Northeastern$ $University$
#     <h1>

# <b>Description:</b> 
# 
# The dataset contains admission outcomes for six majors during the Fall 1973 admission cycle.  
# This dataset is widely used in statistics and econometrics as a classic example of Simpson’s paradox, where an aggregate pattern reverses when the data are disaggregated into subgroups.
# 
# <b>Research Question:</b>  
# Did gender causally affect admission outcomes at UC Berkeley in 1973?
# 
# <br>
# 
# 
# <b>Methods Employed:</b>  
# OLS, Fixed Effects, Logistic Regression, Causal Inference, and Data Visualization.
# 
# <br>
# 
# <b>Key Finding:</b>  
# The analysis finds no evidence of gender-based discrimination in the admissions process.  
# The aggregate gender gap is instead driven by application patterns, as women disproportionately applied to more competitive programs compared to men.
# 
# <br>
# 
# <b>Tools:</b>  
# Python, Pandas, Statsmodels, PyFixest, Scikit-learn, Matplotlib, Seaborn.
# 
# <br>
# 
# 
# <b>Data Source:</b>  
# https://vincentarelbundock.github.io/Rdatasets/datasets.html
# 
# 
# <b>Dataset:</b>  
# Gender bias among graduate school admissions to UC Berkeley (1973)
# </h3>

# In[80]:


#Load Packages
import pandas as pd
import numpy as np
import datetime as dt
import pyfixest as pf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import statsmodels.api as sm


# In[81]:


#load data
df = pd.read_csv(r"C:\Users\Mukaila Rafiu\Downloads\admissions.csv")
df


# In[82]:


df.shape


# In[83]:


df.duplicated()


# In[84]:


df.dtypes


# In[85]:


#Correlation heatmap
df1 = df.select_dtypes(include='number').copy()
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()


# In[86]:


#Enhanced Correlation heatmap
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix,
            annot=True,
            mask=mask,
            cmap='coolwarm',
           vmin=-1, vmax=1,
           center=0,
           square=True,
           linewidths= .5,
           fmt='.2f',
           cbar_kws={'shrink': .8, 'label': 'correlation coefficient'})

plt.title("Enhanced Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.show()


# In[87]:


# Calculate number admitted and rejected
df['admitted_count'] = (df['admitted'] / 100 * df['applicants']).round().astype(int)
df['rejected_count'] = df['applicants'] - df['admitted_count']
df


# In[88]:


#Correlation heatmap
df2 = df.select_dtypes(include='number').copy()
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()


# In[89]:


#Enhanced Correlation heatmap
correlation_matrix2 = df2.corr()
plt.figure(figsize=(9,6))
mask = np.triu(correlation_matrix2)
sns.heatmap(correlation_matrix2,
            annot=True,
            mask=mask,
            cmap='coolwarm',
           vmin=-1, vmax=1,
           center=0,
           square=True,
           linewidths= .5,
           fmt='.2f',
           cbar_kws={'shrink': .8, 'label': 'correlation coefficient'})
plt.title("Enhanced Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()


# In[90]:


#Overall rates ignoring major
overall_rates = df.groupby('gender').agg({'admitted_count': 'sum', 'applicants': 'sum'})
overall_rates['admission_rate'] = (overall_rates['admitted_count'] / overall_rates['applicants'] * 100).round(2)
print(overall_rates[['applicants', 'admitted_count', 'admission_rate']])


# In[91]:


#View 
overall_rates['admission_rate'].plot(kind='bar', color=['#4169E1', '#FF69B4'], title='Overall Admission Rates', ylabel='Admission Rate (%)', figsize=(10, 6))
for i, v in enumerate(overall_rates['admission_rate']): plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
plt.xticks(rotation=0)
plt.tight_layout();
plt.show()


#  **Women appear to be lower admission rate of 34.5% to men of 44.5 person**

# In[92]:


#Within-major rates

by_major = df.pivot_table(values='admitted', index='major', columns='gender', aggfunc='first')
print(by_major.round(2))


# In[93]:


#View
by_major.plot(kind='bar', color=['#4169E1', '#FF69B4'], title='Admission Rates by Major', ylabel='Admission Rate (%)', figsize=(10, 6))
for container in plt.gca().containers: plt.bar_label(container, fmt='%.0f%%', fontweight='bold')
plt.xticks(rotation=0); plt.legend(title='Gender')
plt.tight_layout();
plt.show()


# **Women have higher admission rates in 4 majors out of 6 majors; the differences in C and E are small**

# In[94]:


#Percentage of applicants by gender

applicant_dist = df.pivot_table(values='applicants', index='major', columns='gender', aggfunc='first')
applicant_pct = applicant_dist.div(applicant_dist.sum()) * 100
print("Percentage of applicants by major:")
print(applicant_pct.round(2))


# In[95]:


#
applicant_pct.plot(kind='bar', color=['#4169E1', '#FF69B4'], title='Percentage of Applicants by Major', ylabel='% of Total Applicants', figsize=(10, 5))
for container in plt.gca().containers: plt.bar_label(container, fmt='%.1f%%', fontweight='bold', fontsize=9)
plt.xticks(rotation=0); plt.legend(title='Gender')
plt.tight_layout();
plt.show()


# In[96]:


#Major by difficulty
major_difficulty = df.groupby('major')['admitted'].mean().sort_values()
for major, rate in major_difficulty.items():
    print(f"  Major {major}: {rate:.1f}% admission rate")


# In[97]:


#
ax = major_difficulty.plot(kind='barh', color='#FF6B6B', title='Major Difficulty (by admission rate)', xlabel='Admission Rate (%)', figsize=(10, 5))
ax.bar_label(ax.containers[0], fmt='%.1f%%', fontweight='bold', padding=3)
ax.invert_yaxis()
plt.tight_layout(); plt.savefig('simple_major_difficulty.png', dpi=300, bbox_inches='tight'); plt.show()


# **Women concentrated in harder majors (C, E, F) which are harder to get while Men concentrated in easier majors (A, B)**

# In[98]:


# Female concentration in difficulty major
major_stats = df.groupby('major').agg({'admitted': 'mean', 'applicants': 'sum'})
female_pct = (df[df['gender'] == 'women'].set_index('major')['applicants'] / major_stats['applicants'] * 100)
plt.figure(figsize=(8, 5))
plt.scatter(major_stats['admitted'], female_pct, s=400, alpha=0.6, c=['red', 'orange', 'yellow', 'lightgreen', 'lightblue', 'purple'], edgecolor='black', linewidth=1.5)
for major in major_stats.index: plt.annotate(major, (major_stats.loc[major, 'admitted'], female_pct[major]), ha='center', va='center', fontweight='bold', fontsize=11)
plt.xlabel('Major Selectivity (% admitted)', fontsize=11, fontweight='bold'); plt.ylabel('% Female Applicants', fontsize=11, fontweight='bold'); 
plt.title('Women apply to harder majors', fontweight='bold', fontsize=13); 
plt.grid(True, alpha=0.3); plt.tight_layout(); 
plt.show()


# **Women are disproportionately concentrated in the most competitive majors, especially E, C, and D**

# In[99]:


#Regression with Fixed Effect
#Using Gender effect controlling major
# data preparation
reg = df.copy()
reg['is_female'] = (reg['gender'] == 'women').astype(int)
reg['major_effect_B'] = (reg['major'] == 'B').astype(int)
reg['major_effect_C'] = (reg['major'] == 'C').astype(int)
reg['major_effect_D'] = (reg['major'] == 'D').astype(int)
reg['major_effect_E'] = (reg['major'] == 'E').astype(int)
reg['major_effect_F'] = (reg['major'] == 'F').astype(int)

X = reg[['is_female', 'major_effect_B', 'major_effect_C', 'major_effect_D', 'major_effect_E', 'major_effect_F']]
X = sm.add_constant(X)
y = reg['admitted']

model = sm.OLS(y, X).fit()
print(model.summary())


# **After controlling for major, women have 3.50% higher admission rate than men**

# In[100]:


#T-TEST
#Testing within-major differences to see if significant
men_rates = df[df['gender'] == 'men']['admitted'].values
women_rates = df[df['gender'] == 'women']['admitted'].values
t_stat, p_value = stats.ttest_ind(men_rates, women_rates)
print(f"\nT-test (men vs women admission rates across majors):")
print(f"  Men average: {men_rates.mean():.2f}%")
print(f"  Women average: {women_rates.mean():.2f}%")
print(f"  Difference: {women_rates.mean() - men_rates.mean():.2f} percentage points")
print(f"  T-statistic: {t_stat:.3f}")
print(f"  P-value: {p_value:.4f}")


# In[101]:


#
means = [men_rates.mean(), women_rates.mean()]
sems = [men_rates.std() / np.sqrt(len(men_rates)), 
        women_rates.std() / np.sqrt(len(women_rates))]

plt.figure(figsize=(10, 5))
bars = plt.bar(['Men', 'Women'], means, yerr=sems, capsize=15, 
               color=['#4169E1', '#FF69B4'], alpha=0.7, edgecolor='black', 
               linewidth=2, error_kw={'linewidth': 2})

for bar, mean in zip(bars, means): 
    plt.text(bar.get_x() + bar.get_width()/2, mean + 5, 
             f'{mean:.2f}%', ha='center', fontweight='bold', fontsize=11)

plt.ylabel('Admission Rate (%)', fontsize=11, fontweight='bold')
plt.title(f'T-Test: Men vs Women Admission Rates\nP-value: {p_value:.4f} - NOT significant', 
          fontweight='bold', fontsize=13)
plt.ylim(0, 60)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# **Although women have a 3.5 percentage point higher admission rate across majors, and the model explains 97% of the variation, the result is not statistically significant. This is largely due to the small sample size n = 12.
# With fewer than 20 observations, the test lacks sufficient statistical power, making it difficult to obtain a small p-value even if a real difference truly exists. The T-test also confirm the 3.5 pp of women ahead of mean in admission rate when we subtract 38.17% from 41.67% we get 3.5%**  

# **At first glance, the aggregate data suggest the presence of gender bias: 44.5% of men were admitted compared to 34.5% of women.
# However, this apparent gap disappears once we control for major choice. Within the same majors, women are admitted at rates about 3.5 percentage points higher than their male counterparts, and in 4 of 6 majors, women are admitted at equal or higher rates than men. The overall disparity is instead driven by selection into majors. Women tend to apply disproportionately to more competitive majors (C, D, E, and F), which have an average admission rate of 25.5%, while men are more concentrated in less competitive majors (A and B), where the average admission rate is 68.75%. .
# In short, the observed gender gap in overall admissions reflects selection bias, not discrimination.**

# ## **ROBUSTNESS CHECKS**

# **- Weighted OLS by number of applicants**
# 
# **- Logistic Regression** 
# 
# **- Aggregate level of admission rates ignoring major**

# In[102]:


#Weighted OLS by number of applicants
robustness_results = {}
weighted = sm.WLS(y, X, weights=reg['applicants']).fit()
print(weighted.summary())
robustness_results['Weighted OLS'] = {
    'coef': weighted.params['is_female'],
    'se': weighted.bse['is_female'],
    'pval': weighted.pvalues['is_female']
}
print()


# **While weighted by the number of applicant, it still statistically not significant as the rest due to that sample size.  The point estimate of 2.24 is consistent with the baseline OLS of 3.50, with 98% variation, confirming the robustness of the finding that women have equal or slightly higher admission rates within majors.** 
# 
# **This contradicts the aggregate data, illustrating Simpson's Paradox: the apparent gender disadvantage 
# at the aggregate level (women 34.5% vs men 44.5%) disappears when controlling for major selectivity, indicating that differential major selection rather than discrimination explains  the aggregate disparity.**

# In[103]:


# Logistic Regression 
# Create binary indicator for gender (1 = female, 0 = male)
df['is_female'] = (df['gender'] == 'women').astype(int)
# Create dummy variables for major
major_dummies = pd.get_dummies(df['major'], prefix='major', drop_first=True)
df_prepared = pd.concat([df[['is_female', 'admitted_count', 'rejected_count']], major_dummies], axis=1)
data_list = []
for idx, row in df_prepared.iterrows():
    for _ in range(int(row['admitted_count'])):
        record = {'admitted': 1, 'is_female': row['is_female']}
        for col in major_dummies.columns:
            record[col] = row[col]
        data_list.append(record)
    for _ in range(int(row['rejected_count'])):
        record = {'admitted': 0, 'is_female': row['is_female']}
        for col in major_dummies.columns:
            record[col] = row[col]
        data_list.append(record)
df_expanded = pd.DataFrame(data_list)
feature_cols = ['is_female'] + list(major_dummies.columns)
X = df_expanded[feature_cols].values
y = df_expanded['admitted'].values


# In[104]:


# Fit logistic regression
logit_model = LogisticRegression(fit_intercept=True, max_iter=1000)
logit_model.fit(X, y)
female_idx = 0
female_coef = logit_model.coef_[0][female_idx]
odds_ratio = np.exp(female_coef)
print(f"\nSample size: {len(df_expanded)} individuals")
print(f"Admitted: {y.sum()}, Rejected: {(1-y).sum()}")
print(f"\nLogistic Regression Results:")
print(f"Female Coefficient (log-odds): {female_coef:.4f}")
print(f"Odds Ratio: {odds_ratio:.4f}")


# **Women are 10% more likely to be admitted than the men, matching the findings from OLS +3.5%, all confirming Simpson's Paradox**

# In[105]:


# Aggregate level of admission rates ignoring major
men_admitted = df[df['gender'] == 'men']['admitted_count'].sum()
men_total = df[df['gender'] == 'men']['applicants'].sum()
women_admitted = df[df['gender'] == 'women']['admitted_count'].sum()
women_total = df[df['gender'] == 'women']['applicants'].sum()
men_rate = (men_admitted / men_total) * 100
women_rate = (women_admitted / women_total) * 100
aggregate_diff = women_rate - men_rate
print(f"\nMen:     {men_admitted:4d} admitted / {men_total:4d} applicants = {men_rate:6.2f}%")
print(f"Women:   {women_admitted:4d} admitted / {women_total:4d} applicants = {women_rate:6.2f}%")
print(f"Difference: {aggregate_diff:+.2f} percentage points")


# **The unadjusted aggregate gap is −10.05 pp. After controlling for major, this reverses to +3.5 pp, illustrating Simpson's Paradox**

# ## **Final remarks:**
# **The robustness check confirms the initial findings and reinforces the conclusion that there is no evidence of gender bias in the admission process.**
