#!/usr/bin/env python
# coding: utf-8

# ## <a id='b'></a>Executive Summary
# 
# This analysis seeks two objectives:
# - (1)   discover an insured company's most important features for determining its loss cost, given that a loss occurs. 
# - (2)   classify a news article’s category based on its headline.
# 
# **<font color = blueviolet > First, </font>** this analysis concludes that the following insured company features matter most in determining loss cost:
# - number of employees
# - sector
# - risk zone
# - industry 
# 
# The winning model was XGBoost Regression, with an **R^2 of 0.8346** and a Mean Absolute Error of **77.79 units of loss cost.**
#  - *Credit is given to [Tianqi Chen](https://github.com/tqchen) for creating XGBoost.*
# 
# **<font color = blueviolet>Second, </font>** this analysis correctly picked the category of the news headline **98.26%** of the time. 
# 
# The winning model was a Multinomial Naïve Bayes Classifier enhanced with a TF-IDF vectorizer. 
# - *Credit is given to [Jake VanderPlas](https://github.com/jakevdp) for posting the working pipeline on his Github.*
# 
# Credit to God, my Mother, family and friends.
# 
# All errors are my own.
# 
# Best, <br>
# <font color = green> George John Jordan Thomas Aquinas Hayward, Optimist </font>

# ### <a id='a'></a> Table of Contents
# 
# - [Executive Summary](#b)
# - [Selected Data Visualizations](#c)
# - [Business Overview](#d)
# - [Key Assumptions](#e)
#   - [Step 0 – Getting Ready](#f)<br>
# - [Problem 1: Exploratory Data Analysis (EDA)](#g)
#   - [Step 1 – Read in the Data, Combine DataFrames, Create Independent Variable(s), Export to SQL](#h)
#   - [Step 2 – Begin to Explore the Data via SQL and Visualizations](#i)
#   - [Step 3 – Continue to Explore the Data with Correlation Matrix and Pair Plot](#j)
#   - [Step 4 – Wrangle the Data, More Feature Engineering, Get Dataset Ready for Modeling](#k)
# - [Problem 2: Modeling & Statistical Learning & Problem 3: Model Evaluation](#l)
#   - [Step 5 – Model the Data, and Evaluate the Models](#m)
#      - [Model 1: XGBoost](#n)
#      - [Model 2: Standard Linear Regression](#o)
#      - [Model 3: Support Vector Regression](#p)
#      - [Model 4: Lasso Regression](#q)
#   - [Step 6 – Side Project - Loss Classification](#r)
#      - [Model 5: Logistic Regression Classification](#s)
#      - [Model 6: XGBoost Classification](#t)
# - [Problem 4: Text Mining](#u)
#    - [Step 6 – Text Classification of News Articles](#v)
# - [Thank you!](#w)

# [Back to Top](#a)

# ## <a id='c'></a>Selected Data Visualizations
# 
# #### Loss vs. No Loss Breakdown
# 
# <a href="https://ibb.co/txypCgZ"><img src="https://i.ibb.co/JyXjkVx/exhbit003-loss-to-no-loss-breakdown.png" alt="exhbit003-loss-to-no-loss-breakdown" border="0"></a>
# 
# #### Feature Correlation Matrix
# 
# <a href="https://ibb.co/VHNMrSn"><img src="https://i.ibb.co/jV53BM0/exhibit-0a-correlation-matrix.png" alt="exhibit-0a-correlation-matrix" border="0"></a>
# 
# #### Feature Pair Plots
# 
# <a href="https://ibb.co/BrG30nj"><img src="https://i.ibb.co/z7Fxq6b/exhibit-0b-pair-plot-lower-res.png" alt="exhibit-0b-pair-plot-lower-res" border="0"></a>
# 
# #### XGBoost Regression Results
# 
# <a href="https://ibb.co/xDBFR2F"><img src="https://i.ibb.co/KXfK35K/exhibit-b-xgb-regression.png" alt="exhibit-b-xgb-regression" border="0"></a>
# 
# #### XGBoost Feature Importance
# 
# <a href="https://ibb.co/RHDCCgf"><img src="https://i.ibb.co/BsCggfb/exhibit-c-xgbfeatures.png" alt="exhibit-c-xgbfeatures" border="0"></a>
# 
# #### XGB Classification Confusion Matrix
# 
# <a href="https://ibb.co/zxyvykV"><img src="https://i.ibb.co/dMdYdh4/exhibit-h-xgboost-regression-confusion-matrix.png" alt="exhibit-h-xgboost-regression-confusion-matrix" border="0"></a>
# 
# #### Multinomial Naive Bayes Text Classification Confusion Matrix
# 
# <a href="https://ibb.co/DKHmSkC"><img src="https://i.ibb.co/0JR1dZM/exhibit-x9-text-classification-confusion-matrix.png" alt="exhibit-x9-text-classification-confusion-matrix" border="0"></a>

# [Back to Top](#a)

# ###  <font color = 'blue'> <a id='d'></a> Business Overview </font>
# 
# <font color = 'blue'> The insurance product is mainly focused on two metrics: losses and premiums. Losses is what the insurance company pays out to policyholders for claims, and premiums is the price charged to policyholders for purchasing an insurance policy. Because the premium is set at the inception of the policy, insurance companies must do their best to calculate the appropriate premium amount, which fundamentally involves computing expected future loss from the policy.
# 
# Two ratios are often used to assess the profitability of an insurance company:
# <br></br>
# $$
# Loss\ Ratio = \frac{Total\ Losses}{Premiums}
# $$
# <br></br>
# $$
# Loss\ Cost = \frac{Total\ Losses}{Exposure\ Units}
# $$

# [Back to Top](#a)

# ## <a id='e'></a> Key Assumptions
# - I focused the models on the data in the Companies CSV. I assume that we want to focus on *customer* insights. I did, however, consider the month when the claim was filed, which came from the Claims CSV. I also used exposure units from the Policy CSV to help compute the Loss Cost independent variable.
# - I focused the models on regression, and not classification. Therefore, my chief data objective was to answer a problem containted in a smaller universe: looking at the losses we know occurred, can we predict how much the loss will be based on the insured company's features?
#   - As a [brief side-project](#r), I included two classification models to begin tackling the larger question of how accurately we can predict *whether* a loss will occur in the first place.
#   - Ultimately, we would want to connect the two models. **If** the loss-predict-classifier model predicted a 'Yes,' then we would apply the los-cost-predict-regression model. 
# - As detailed below, I focused the model exclusively on the United States.
# - As detailed below, I excluded revenue and cost of revenue from the data because I believe it was based off of premium, which we must exclude here because "in reality premium would only be available after a predicted loss cost is calculated."
# - As detailed below, I opted to leave outliers **in** the dataset. Although it's often a good idea to remove the outliers from models, because insurance loss is so impacted by outliers, I did not feel comfortable taking them out without more input from the team.
#   - I did, however, in addition to the full model results graph, provide a zoomed-in version focused on the non-outlier points.
# - This notebook is largely set up for the user to tinker with parameters and assumptions. In practice, we could employ pipelines to use fewer lines of code.

# [Back to Top](#a)

# ## <a id='f'></a> Step 0 - Getting Ready
# - Load in the dependencies to be used throughout the notebook.
# - There will be some redundancies because I'm combining several notebooks into one here.
# 

# In[100]:


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import mysql.connector as mysql
from sqlalchemy import create_engine
import altair as alt
alt.renderers.enable('notebook')
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, explained_variance_score, confusion_matrix, accuracy_score, precision_score, recall_score
import statsmodels.formula.api as sm
import missingno as msno
import xgboost as xgb
from sklearn.svm import SVR
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
mnb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
import warnings #just for the final version
warnings.filterwarnings("ignore")


# [Back to Top](#a)

# ### <font color="blue"> <a id='g'></a> Problem 1: Exploratory Data Analysis (EDA) </font>
# 
# <font color="blue">You are provided with 3 datasets containing information related the 2016 policy year for a particular line of business:
# 
# **Policies_Data.csv**: A dataset containing details about the company's policies written in 2016.  
# **Claims_Data.csv**: A dataset containing details about the company's claims paid (also called losses) for the 2016 policy year.  
# **Companies_Data.csv**: A dataset containing detailed information regarding the insureds.  
# 
# Note that Claims_Data can be joined with Policies_Data using `policy_id`, and Policies_Data can be joined with Companies_Data using `company_id`.
# 
# For this first task, read in the 3 datasets and explore the data a bit. Create some summary tables and visualizations to drawing meaningful conclusions which can help us later in the modeling or decision making process. Some suggestions to get you started (note that this isn't an exhaustive list):
# 
# - Generate some basic information about the datasets (shape, cardinality, missing values, etc.).
# - Explore potential correlations among variables.
# - Calculate some useful ratios (eg: average loss ratios, claim counts by category, etc.).
# - Explain what variables you would use as initial features for a predictive model, and why.
# 

# [Back to Top](#a)

# ## <a id='h'></a> Step 1 - Read in the Data, Combine DataFrames, Create Independent Variable(s), Export to SQL
# - Read in the CSV files.
# - Join the files on the correct keys.
# - Create the two variables we will need for the rest of the presentation:
#   - Was There A Loss? (1 or 0) 
#   - Loss Cost (Claim Amount   /   Exposure Units)
# - Ensure times are in datetime format.
# - Peek at data.
# - Export the data to a SQL database, which I've set up locally in MySQL Workbench.
# 

# In[2]:


policies = pd.read_csv("Policies_Data.csv")
claims = pd.read_csv("Claims_Data.csv")
companies = pd.read_csv("Companies_Data.csv")


# In[3]:


companies_policies = companies.merge(policies, left_on='company_id', right_on='company_id',                                     how='left',                                    suffixes = ('_company','_policy'))


# In[4]:


combined = companies_policies.merge(claims, left_on='policy_id', right_on='policy_id',                                    how='left',                                  suffixes = ('_policy','_claims'))


# In[5]:


combined.head(3)


# In[6]:


loss_cost = []
was_there_a_loss = []
for i in range(len(combined.company_id_policy)):
    if pd.isnull(combined.claim_amount[i]) == True:
        loss_cost.append(0)
        was_there_a_loss.append(0)
    else:
        numerator = combined.claim_amount[i]
        denominator = combined.exposure_units[i]
        loss_cost_for_given_row = numerator / denominator
        loss_cost.append(loss_cost_for_given_row)
        was_there_a_loss.append(1)


# In[29]:


combined['loss_cost'] = loss_cost
combined['was_there_a_loss'] = was_there_a_loss


# #### 👆🏽Thoughts about this? 👆🏽
# - This will be very helpful for solve another related issue: classification of when a loss will occur.

# In[8]:


combined['inception_date_policy'] = pd.to_datetime(combined['inception_date_policy'],                                             format='%Y-%m-%d')
combined['expiration_date_policy'] = pd.to_datetime(combined['expiration_date_policy'],                                             format='%Y-%m-%d')
combined['claim_date'] = pd.to_datetime(combined['claim_date'],                                             format='%Y-%m-%d')
combined['inception_date_claims'] = pd.to_datetime(combined['inception_date_claims'],                                             format='%Y-%m-%d')
combined['expiration_date_claims'] = pd.to_datetime(combined['expiration_date_claims'],                                             format='%Y-%m-%d')


# #### 👆🏽Thoughts about this? 👆🏽
# - Time info can be very helpful. 
# - In this project, I'm going to focus on data in the claims CSV file. 
#   - I'll use, for instance, month of claim date to see if the time of year has any insight into loss cost.
#   - **BUT** because I'm focusing on just the claim data, I'm not going to pull the policy length information (which could also be computed here).

# In[9]:


combined.head(3)


# In[10]:


engine = create_engine('mysql+mysqlconnector://newuser:data@localhost:3306/sys', echo=False)
combined.to_sql(name='insurance2', con=engine, if_exists = 'append', index=False)
#this is called 'insurance2' for display purposes...the rest of the data will queries from 'insurance'


# [Back to Top](#a)

# ## <a id='i'></a>Step 2 - Begin to Explore the Data via SQL and Visualizations
# - Using SQL's powerful aggregation capabilities, we can begin to see trends and leverage points in the data.
#   - This is very similiar to using Pivot tables in Excel. 
#   - This can be done in Pandas, but I prefer SQL and the functionality of MySQL Workbench.
# - This is done to prepare to 'wrangle' the data.
#   - Are there categoricals that should be binned?
#   - Are ther ordinals that should be binned?
#   - When does it make sense to form an 'Other' category?
# - I am only showing a select few of the queries here. In practice, there are many more views, as I investigate the data.
# 

# In[11]:


#connect to the MySQL database
db = mysql.connect(
    host = "localhost",
    user = "newuser",
    passwd = "data",
    auth_plugin='mysql_native_password',
    database = 'sys')


# In[12]:


loss_cost_and_country_by_count = pd.read_sql("""
select
country,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim,
round((count(*) / (select count(*) from sys.insurance where was_there_a_loss = 1)),2) as prct_of_all_losses
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 4 desc;
""", con=db)
loss_cost_and_country_by_count.head(10)


# #### 👆🏽Thoughts about this? 👆🏽
# - USA dominates the dataset at about 70% (of losses).
#   - Maybe we should think about doing a USA vs. 'Other' breakdown.

# In[13]:


loss_cost_and_country_by_amount = pd.read_sql("""
select
country,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim,
round((count(*) / (select count(*) from sys.insurance where was_there_a_loss = 1)),2) as prct_of_all_losses
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
loss_cost_and_country_by_amount.head(10)


# #### 👆🏽Thoughts about this? 👆🏽
# - We see some very high numbers that seem to differ by county.
# - This is where 'domain expertise' comes in.
# - Because different countries have such different laws and regulations, not to mention other particularizations (geography, climate, risk for certain natural diasters), I am going to make a BIG decision:
#   - I am going to limit the entire model to **only** the United States.
#   - In practice, I think there should be many models for many different niches.
#    - I have big concerns with combining such segmented data. Instead, I think multiple models should be used.

# In[14]:


loss_cost_and_state_by_count = pd.read_sql("""
select
state,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim,
round((count(*)/(select count(*) from sys.insurance where country='USA' and was_there_a_loss=1)),2) as prct_usa_losses
from sys.insurance
where was_there_a_loss = 1
and state is not null
group by 1
order by 4 desc;
""", con=db)
loss_cost_and_state_by_count.head(10)


# #### 👆🏽Thoughts about this? 👆🏽
# - The state data look a lot more evenly distributed. I feel good about focusing on the USA, and making the states dummy variables.
#   - CA has the most claims, and, with the largest population, this makes sense.

# In[15]:


company_type_break_down = pd.read_sql("""
select
company_type,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim,
round((count(*) / (select count(*) from sys.insurance where was_there_a_loss = 1)),2) as prct_of_all_losses
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
company_type_break_down


# #### 👆🏽Thoughts about this? 👆🏽
# - To simplify the dummy variable process, it will make sense to divide company_type into "1", "2", and "Other."

# In[16]:


industry_code_breakdown = pd.read_sql("""
select
industry_code,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
industry_code_breakdown.head(10)


# #### 👆🏽Thoughts about this? 👆🏽
# - Since the data here seems so evenly distributed, this will make a good dummy variable without binning.

# In[17]:


sector_code_breakdown = pd.read_sql("""
select
sector_code,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc
limit 10;
""", con=db)
sector_code_breakdown


# #### 👆🏽Thoughts about this? 👆🏽
# - Similarly, since the data here seems so evenly distributed, this will make a good dummy variable without binning.

# In[18]:


location_count_breakdown = pd.read_sql("""
select
locations,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
location_count_breakdown


# #### 👆🏽Thoughts about this? 👆🏽
# - Looks like there's a definite trend here with location count. 
# - Seems like we see a real trend here of the fewer the locations, the more the loss cost. It could be that the loss is getting spread out, but this definitely seems important (or at least correlated or even colinear).
# - Let's graph it. 

# In[19]:


scatter_plot = alt.Chart(location_count_breakdown) .mark_circle(size=200) .encode(
    x= alt.X('locations',scale=alt.Scale(domain=[0, 30]),\
            axis=alt.Axis(title="Number of Locations")),
    y= alt.Y('loss_per_company',scale=alt.Scale(domain=[0, 400]),\
            axis=alt.Axis(title='Average Loss Cost')))\
 .properties(
    width=300,
    height=300  
)#.interactive()
scatter_plot.save('exhbit000a1_numb_locatoins_to_avg_loss_count.png', scale_factor=3.0)
scatter_plot


# #### 👆🏽Thoughts about this? 👆🏽
# - Pretty cool!
# - Would definitely want to ask the team more about this.

# In[20]:


year_founded_loss_cost_breakdown = pd.read_sql("""
select
year_founded,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
year_founded_loss_cost_breakdown.head(10)


# #### 👆🏽Thoughts about this? 👆🏽
# - There defintiely seems to be a trend here, but the null count (see row 4) is too high.
# - I will likely drop the entire column. 
# - Further, this looks like selection bias...
#   - If there are older companies in the system, then they are the ones left that are not as prone to loss. 
#   - (ie, the famous example of the WW2 fighter planes coming back with holes only on the body, so that meant that the British Air Force need to fix the wings).
# 

# In[21]:


employee_count_loss_cost_breakdown = pd.read_sql("""
select
employees,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
employee_count_loss_cost_breakdown.head(10)


# #### 👆🏽Thoughts about this? 👆🏽
# - Something really intersting is going on here. 
# - It looks like the fewer the employees, the greater the loss.
# - It look so strong, that I would want to figure out if this variable is direclyt related to how the loss is calculated.
# - For now, let's graph it!

# In[22]:


scatter_plot = alt.Chart(employee_count_loss_cost_breakdown) .mark_circle(size=45) .encode(
    x= alt.X('employees',scale=alt.Scale(domain=[0, 300]),\
            axis=alt.Axis(title="Count of Employees")),
    y= alt.Y('loss_per_company',scale=alt.Scale(domain=[0, 1000]),\
            axis=alt.Axis(title='Average Loss Cost')))\
 .properties(
    width=300,
    height=300  
)#.interactive()
scatter_plot.save('exhbit001_employee_count_to_avg_loss_count.png', scale_factor=3.0)
scatter_plot


# #### 👆🏽Thoughts about this? 👆🏽
# - This is amazing. 
# - Would definitely want to ask the team more about this. 
#   - To see if this is a powerful insight, or just part of the way units are calculated.

# In[23]:


risk_zone_loss_cost_breakdown = pd.read_sql("""
select
risk_zone,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
risk_zone_loss_cost_breakdown


# #### 👆🏽Thoughts about this? 👆🏽
# - Even though this has a lot of nulls, I would leave it in.
# - Why?
#   - We can see a stark differnce between the "5" and "4" risk zones (which I'd assume to mean the riskiest) and everyone else.
# - It might make sense to set this up as an ordinal categorical column. 
#   - "5", "4," and all else getting a "2."
#       - Choosing "2" is an initial choice, and it can be changed later on. It could be "1.5" or "1", for example.
#   - This is a creative way of showing that "5" and "4" are greater in magnitude than "2" is. And we'd miss that with a regular categorical. 
# - Would definitely want to ask the team more about this. 

# In[24]:


legal_involvement_breakdown = pd.read_sql("""
select
legal_involvement,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
legal_involvement_breakdown


# #### 👆🏽Thoughts about this? 👆🏽
# - At first blush, this looks like it will be meaningful. 
# - For further investiagation we could run a two sample proportion t-test to see if this difference in loss_per_company is meaningful or simply the byproduct of chance. 
# - For now, we'll definitely want to include legal_involvement in the model's first drafts.

# In[25]:


internet_complaints_breakdown = pd.read_sql("""
select
internet_complaints,
round(avg(loss_cost),2) as loss_per_company,
round(std(loss_cost),2) as std_of_loss,
count(*) as count_of_companies_with_a_claim
from sys.insurance
where was_there_a_loss = 1
group by 1
order by 2 desc;
""", con=db)
internet_complaints_breakdown.head(10)


# #### 👆🏽Thoughts about this? 👆🏽
# - This does not seem to have a strong relationship.
# - I'll graph it to show that some variables that you might expect to have one relationshp can actually have another or none at all.

# In[26]:


scatter_plot = alt.Chart(internet_complaints_breakdown) .mark_circle(size=150) .encode(
    x= alt.X('internet_complaints',scale=alt.Scale(domain=[0, 12]),\
            axis=alt.Axis(title="Number of Internet Complaints")),
    y= alt.Y('loss_per_company',scale=alt.Scale(domain=[0, 300]),\
            axis=alt.Axis(title='Average Loss Cost')))\
 .properties(
    width=300,
    height=300  
)#.interactive()
scatter_plot.save('exhbit002_number_of_internet_complaints_to_avg_loss_count.png', scale_factor=3.0)
scatter_plot


# #### 👆🏽Thoughts about this? 👆🏽
# - Perhaps there is a slight relationship, but it doesn't seem too strong.

# In[27]:


how_rare_are_losses = pd.read_sql("""
select
case when was_there_a_loss = 1 then "Loss"
when was_there_a_loss = 0 then "No Loss" end as was_there_a_loss,
count(*) as count_of_companies_with_a_claim,
round((count(*) / (select count(*) from sys.insurance)),3) as prct_of_whole
from sys.insurance
group by 1
order by 2 desc;
""", con=db)
how_rare_are_losses


# #### 👆🏽Thoughts about this? 👆🏽
# - This was just to get an idea of how common or rare losses are in the dataset. 
# - I'll put a pie chart together below.

# In[28]:


pie_labels = how_rare_are_losses.was_there_a_loss
sizes = how_rare_are_losses.count_of_companies_with_a_claim
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
colors_nt = ['pink', 'hotpink']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=pie_labels, autopct='%1.1f%%', colors=colors_nt,
        shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.suptitle("Insurance Claim Loss vs. No Loss Breakdown", fontsize = 14, fontweight = 'bold')
plt.savefig('exhbit003_loss_to_no_loss_breakdown.png',dpi=300, bbox_inches='tight')
plt.show()


# #### 👆🏽Thoughts about this? 👆🏽
# - So losses do happen fairly frequetly. 
# - But the norm will still be No Loss.
# - Since the assignment here is to linearly regress (loss claim amount / exposure units) = loss cost, I will focus my models on the 24.4% of the claims here that actually had a loss. 
# - **But** I will also, as a side project, train some classifiers to see if they can predict when a loss occurs at all.
#   - This was why I included a was_there_a_loss binary column at the outset. 

# [Back to Top](#a)

# ## <a id='j'></a> Step 3 - Continue to Explore the Data with Correlation Matrix and Pair Plot
# - I will use the correlation matirx to help in feature selection, when we get to that stage.
# - The Pair Plot is like an en masse version of what I did above with my SQL queries.
#   - I use it to verify some of what I found above, and to potentially think of new ideas.
# 

# In[31]:


#first we pull out a good number of the variables
insurance = pd.read_sql("""
select
state, company_type, industry_code, sector_code, locations, year_founded,
employees, risk_zone, revenue, cost_of_revenue, legal_involvement, internet_complaints,
exposure_units, extract(month from claim_date) as claim_month, claim_amount, was_there_a_loss, loss_cost
from sys.insurance
where 1=1
#and was_there_a_loss = 1
and country = 'USA';
""", con=db)


# In[32]:


#all credit due to: Pedro Marcelino 
#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#correlation matrix
corrmat = insurance.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.savefig('exhibit_0a_correlation_matrix.png',dpi=300, bbox_inches='tight')


# #### 👆🏽Thoughts about this? 👆🏽
# - The correlation matrix helps us see relationships we may otherwise miss.
# - For instance, we can see that employees and exposure_units seem to be very related.
#   - I'm not sure why.
#     - It could be that somehow employees counts are involved how the exposure_units are calculated. Different insurance companies calculated exposure units in different ways.
# - We also can see that number of locations and exposure units are correlated. 
# - Yet the year founded and exposure units are not correlated.
#   - Though note that year founded has many Nulls, and, as such, will soon be dropped from the data set.

# In[33]:


#all credit due to: Pedro Marcelino 
#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#scatterplot
sns.set()
#cols = ['column1', 'column2']
sns.pairplot(insurance, size = 2.5)
#plt.savefig('exhibit_0b_pair_plot.png',dpi=600, bbox_inches='tight')
#600 dpi is commented out. It's useful for viewing all the data up close on your desktop, but slows down the script.
plt.savefig('exhibit_0b_pair_plot_lower_res.png',dpi=150, bbox_inches='tight')
plt.show()


# #### 👆🏽Thoughts about this? 👆🏽
# - There is a lot of information packed in this dense photo.
# - Please note: because these are just scatter plots, and bar graphs, as opposed to box-and-whisker or violin plots, we don't have a sense of the variance or the count of each of the features. 
# - The bottom row is important because that's the row that has the independent variable in the y axis.
# - The data graphs in the SQL section where magnified so as to look at relationships in the majority of the dataset, but these graphs show the full dataset.
#   - As such, you can see how the outliars factor in.
# - A few intersting things to note:
#   - Lower company types seem to have lower loss.
#   - Lower sector codes seem to have higher loss.
#     - There may also be a particular sector code with much lower losses; this will come out in the dummy variables. 
#   - Fewer locatins seem to have higher loss.
#   - More recent year-foundeds seem to have higher loss, but this might be a selection bias.
#   - Lower numbers of employees seems to have higher loss. As state previously previously, we would want to further investigate how exposure units are calcualted because that informs how the loss will be calculated. There might be a definitional relationship between the number of empoyees and the exposure units.
#   - Higher risk zone, seems to have higher loss.
#   - Here, legal involvement does not seem to have a strong relationship with loss. 
#   - Here, there seems to be a relationship for the more internet complaints, the lower the loss.
#   - Claim month does not seem to have a strong relationship with loss. 
#   

# [Back to Top](#a)

# ## <a id='k'></a> Step 4 - Wrangle the Data, More Feature Engineering, Get Dataset Ready for Modeling
# - I will begin to group and bin data together, where appropriate, as following some of the insights seen above.
# - I will create dummy variables, merge them in, and then eliminate their 'parent' columns.
# - Because the independent variable is highly skewed, I will log-transform it so that it better approximates a normal distribution.
#   - This is very helpful for many of the models.
# - I will also make decisions on how to handle varous null values.
#   - When possible I prefer to cut out a null data observation, rather than impute a mean or median because, in my opinion, this better preserves the integrity of each given observation. 
#     - However, I do understand that this is not always possible to do in instances with a lower number of  n observations. 
# - I will log-transform the most skewed features.
# - I view the percentiles of the independent variable to get an idea for the prominence of outliars.
# - I will split the data up into two data sets.
#   - One for classification, which will incude both losses and non-lossed.
#   - One for regression, which will include **only** losses.
# - I will drop key-like features, such as ID and address.
# - I have a preference to drop a feature if more than 15% of it's observations are null. 

# In[34]:


msno.matrix(insurance,  color = (.0,.0,.2))


# #### 👆🏽Thoughts about this? 👆🏽
# - Solid bars mean no nulls. 

# In[35]:


#missing data
total = insurance.isnull().sum().sort_values(ascending=False)
percent = (insurance.isnull().sum()/insurance.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# #### 👆🏽Thoughts about this? 👆🏽
# - Claim amount and claim revenue are expeected to have that many nulls because this data set also includes observations where no loss occurred.
# - Risk zone is going to be treated as previously specificed, with a "5","4", then "2" ordinal framework. The nulls will be captured by the "2" categorical. 
# - Year founded will be eliminated due to too many nulls.
# - Revenue and cost of revnue will be eliminated **not** because of nulls, but because I view them as potentially related to the premium number, which we are not allowed to consider in this data set. I am unsure if revenue is referring to company reveneue or reveneu from the company to the insurance company. For now, I am assuming it is the latter, and I am eliminating the entire feature.

# In[36]:


insurance_processed = insurance.drop(['cost_of_revenue','revenue','year_founded'],                                    axis = 1)


# In[37]:


risk_zone = []
for i in range(len(insurance_processed.risk_zone)):
    if insurance_processed.risk_zone[i] == 5.0:
        risk_zone.append(5)
    elif insurance_processed.risk_zone[i] == 4.0:
        risk_zone.append(4)
    else:
        risk_zone.append(2) #I am choosing this from a rough approx of what i saw in the sql


# In[38]:


insurance_processed.risk_zone = risk_zone


# In[39]:


company_type = []
for i in range(len(insurance_processed.company_type)):
    if insurance_processed.company_type[i] == 2:
        company_type.append(2)
    elif insurance_processed.company_type[i] == 1:
        company_type.append(1)
    else:
        company_type.append('Other')


# In[40]:


insurance_processed.company_type = company_type


# In[41]:


insurance_processed = insurance_processed.dropna(subset=['locations', 'employees','exposure_units'])


# In[42]:


#sometimes a number is being used as a categorical, so we need to turn it into a string so the computer knows too
insurance_processed['industry_code'] = insurance_processed['industry_code'].apply(str)
insurance_processed['sector_code'] = insurance_processed['sector_code'].apply(str)


# In[43]:


dummy = pd.get_dummies(insurance_processed[['state',                                          'company_type','industry_code',                                          'sector_code']], drop_first=True)


# #### 👆🏽Thoughts about this? 👆🏽
# - We make sure to set the 'drop_first' parameter to **`True`** so as to avoid the 'dummy variable trap' of multicolinearity.

# In[44]:


insurance_ml = pd.concat([insurance_processed, dummy], axis = 1)


# In[45]:


insurance_ml = insurance_ml.drop(['state','company_type','industry_code','sector_code'],                                axis = 1)


# In[46]:


msno.matrix(insurance_ml, color = (.0,.0,.2))


# #### 👆🏽Thoughts about this? 👆🏽
# - The two white-speckled columns are "claim_month" and "claim_amount" because some observations do not have a loss. 
#   - The white shading in these columns is about 75% of the area.
#   - The dark blue shading in these columnd is about 25% of the area.

# In[47]:


#for classification work to find loss or not
#splitting up the data-set for the classification side-project
insurance_ml_classification = insurance_ml


# In[48]:


#for regression work to find total loss
#here, we only want to look at what happens WHEN there is a loss
insurance_ml_regression = insurance_ml[insurance_ml.was_there_a_loss == 1]


# In[49]:


#we may want to eliminate the outliars of the dependent variable for regression
#for classification, b/c there is a threshold approach and everything goes between 1 and -1 via the sigmoid, I'm 
#less concerned
insurance_ml_regression.loss_cost.describe()


# ####  👇🏽Percentile Check of the Independent Variable👇🏽

# In[50]:


feature_to_check = 'loss_cost'
percentile_check = [1,25,50,75,90,95,97.0,98.0,99.0,99.7,99.90, 99.99]
percentile_values = []
y_pos = np.arange(len(percentile_check))
for i in percentile_check:
    percentile = i
    value = round(np.percentile(insurance_ml_regression[feature_to_check], i),2)
    percentile_values.append(value)
    print('Percentile '+str(i)+": "+str(value))
fig, axs = plt.subplots(ncols=2, figsize=(14,4))
#using a subplot method coupled with an inline parameter to have high resolution
axs[1].set_axis_off()
axs[0].bar(y_pos, percentile_values, align='center', alpha=1)
axs[0].set_ylabel('Values')
axs[0].set_xticks(y_pos)
axs[0].set_xticklabels(percentile_check, fontsize=10)
axs[0].set_xlabel('Percentile')
plt.suptitle('Value By Percentile: Understanding Outliers',x=0.44,              horizontalalignment='right' )
axs[0].set_title(feature_to_check, fontweight = 'bold')

extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('exhibit_a_percentile_look',dpi=300, bbox_inches=extent.expanded(1.38, 1.35))
plt.show()


# #### 👆🏽Thoughts about this? 👆🏽
# - We can see just how much larger the top tenths of a percent are than the rest of the data set.
# - We defintiely have major outliars here.
# - As outlined at the start, I have decided to keep them in this model. 
# - Although this will make the model worse at fitting the majority of more 'usual' standard claims, because insurance is so predicated on outliar losses, I feel uncomfortable removing them until we talk with the team and the domain experts.
# - Thus, I am making the decision to leave these outliars in, but we can see how powerfully they shall affect the data from this chart.

# In[51]:


fig, axs = plt.subplots(ncols=2, figsize=(12,4))
axs[1].set_axis_off()
axs[0].hist(insurance_ml_regression.loss_cost, bins=100)
axs[0].set_title('Histogram of Dependent Variable', fontweight = 'bold')
axs[0].set_ylabel('Frequency')
axs[0].set_xlabel('Values')
plt.show()


# #### 👆🏽Thoughts about this? 👆🏽
# - This is another way of showing what we just saw above.

# In[53]:


insurance_ml_regression.loss_cost.skew()


# #### 👆🏽Thoughts about this? 👆🏽
# - This is highly, positively skewed.
# - Skewness "is the degree of distortion from the symmetrical bell curve or the normal distribution." -Diva Jain
# - "If the skewness is less than -1(negatively skewed) or greater than 1(positively skewed), the data are highly skewed." - Diva Jain

# In[54]:


insurance_ml_regression.loss_cost.kurt()


# #### 👆🏽Thoughts about this? 👆🏽
# - The independent variable exhibits very high kurtosis.
# - Kurtosis is "is used to describe the extreme values in one versus the other tail. It is actually the measure of outliers present in the distribution." -Diva Jain
#   - Leptokurtic (Kurtosis > 3): Distribution is longer, tails are fatter. Peak is higher and sharper than Mesokurtic, which means that data are heavy-tailed or profusion of outliers. Outliers stretch the horizontal axis of the histogram graph, which makes the bulk of the data appear in a narrow (“skinny”) vertical range, thereby giving the “skinniness” of a leptokurtic distribution. 
#   - Platykurtic: (Kurtosis < 3): Distribution is shorter, tails are thinner than the normal distribution. The peak is lower and  broader than Mesokurtic, which means that data are light-tailed or lack of outliers. The reason for this is because the extreme values are less than that of the normal  distribution. -Diva Jain

# In[55]:


fig, axs = plt.subplots(ncols=2, figsize=(12,4))
axs[1].set_axis_off()
stats.probplot(insurance_ml_regression.loss_cost, plot=axs[0])
plt.show()


# #### 👆🏽Thoughts about this? 👆🏽
# - The independent variable is not normal.
# - We can begin to fix this, given the positive skew, with a log-transformation. 

# In[56]:


insurance_ml_regression['loss_cost_logged'] =   np.log1p(insurance_ml_regression['loss_cost'])


# In[57]:


fig, axs = plt.subplots(ncols=2, figsize=(12,4))
axs[1].set_axis_off()
axs[0].hist(insurance_ml_regression.loss_cost_logged, bins=100)
axs[0].set_title('Histogram of Log-Transformed Dependent Variable', fontweight = 'bold')
axs[0].set_ylabel('Frequency')
axs[0].set_xlabel('Values')
plt.show()


# In[58]:


print("Skew: "+str(insurance_ml_regression.loss_cost_logged.skew()))
print("Kurtosis: "+str(insurance_ml_regression.loss_cost_logged.kurt()))


# In[59]:


fig, axs = plt.subplots(ncols=2, figsize=(12,4))
axs[1].set_axis_off()
stats.probplot(insurance_ml_regression.loss_cost_logged, plot=axs[0])
plt.show()


# #### 👆🏽Thoughts about this? 👆🏽
# - Now things look more normal.
# - Please note I will use the np.exp() method to bring the models' predictions back to the original units.

# In[61]:


#finally, I will also run the log-transform on any continous variables in the data that also show positive skewness
#For this block, credit goes to Alexandru Papiu 
#(https://www.kaggle.com/apapiu/regularized-linear-models)
#log transform skewed numeric features:
continous_features = ['locations','employees','internet_complaints','exposure_units']
skewed_feats = insurance_ml_regression[continous_features].apply(lambda x: stats.skew(x)) 
#compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

insurance_ml_regression[skewed_feats] = np.log1p(insurance_ml_regression[skewed_feats])


# [Back to Top](#a)

# ###   <font color="blue"> <a id='l'></a> Problem 2: Modeling & Statistical Learning </font>
# 
# <font color = 'blue'> Using what you’ve learned from the data exploration exercise above, try building a few different models to predict **loss cost** (defined above). Some suggestions to help you start:
# - Create some new features using the insights you’ve gained from your data exploration exercise.
# - Use some different classes of models: linear models, tree-based models, or any others you can think of for predicting a numeric response. There are many options, pick at least two as you will be asked to compare your modeling results in the next exercise.
# 
# Note: Do not use premium as an independent variable, as in reality premium would only be available after a predicted loss cost is calculated.

# ### <font color="blue"> Problem 3: Model Evaluation
# 
# <font color = 'blue'> You've built a few models in the previous exercise:
# 
# - How would you evaluate the performance of the models you built? Try calculating some validation statistics as well as building some graphs/diagrams to illustrate the lift provided by each model.
# - Why did you choose the evaluation metric(s) you selected?
# - What are some advantages and disadvantages of the models you compared, and which model would you recommend?

# [Back to Top](#a)

# ## <a id='m'></a> Step 5 - Model the Data, and Evaluate the Models
# - I will now begin to fit models to the data!
# - Training will be 80%, testing will be 20%, cross validation will be run on the training data with cv=10 (10 folds).
# - I will use:
#   - XGBoost
#     - With a GridSearch for Optimal Parameters
#     - With XGBoost's special 'feature importance' array.
#   - Standard Linear Regression w/ no Standardized Scaling
#   - Support Vector Regression, Polynomial w/ Standardized Scaling
#   - Lasso Regression w/ Standardized Scaling
# - All models are cross-validated on the training data via sklearn's cross_val_score method.
# - The way I evaluate these models:
#   - R^2 score
#   - Mean Absolute Error in Original Units (log predictions brougth back to original units via np.exp())
#       - I also check the standard deviation and the mean of the training data.
#       - My thought is that if the absoltute mean error is somehow greater than the standard deviation, then the model is perhap worse than no model at all.
#   - Skew and Kurtosis of the Residuals.
#   - Plot Actual (x axis) vs. Predicted (y axis), with graph focused on 98% of the claims losses
#   - Plot Actual (x axis) vs. Predicted (y axis) for 100%
#   - Overfitting vs. Undefitting
#      - Train to Test Mean Absolute Error Ratio
#         - The higher the ratio, the more overfit the model is.
# 

# In[63]:


features = insurance_ml_regression.drop(['loss_cost','claim_amount','was_there_a_loss',                                        'loss_cost_logged','exposure_units'], axis = 1)
#we can't have claim amount b/c it's also in the independent variable
#we also know from the correlation matrix that exposure units & employees are highly correlated, so we only need one
y = insurance_ml_regression.loss_cost_logged #y = insurance_ml_regression.loss_cost


# [Back to Top](#a)

# # <a id='n'></a>📈 XGBoost 📈

# In[64]:


#xgb, just done again to have it all in a nice place
X_train_xgb, X_test, y_train_xgb, y_test = train_test_split(features, y, test_size = 0.2, random_state = 1)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_model.fit(X_train_xgb,y_train_xgb)
y_pred_xgb = xgb_model.predict(X_test)
#print r^2 and plot xgb
print("R^2:"+str(round(r2_score(y_test, y_pred_xgb),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_test),                                                                               np.exp(y_pred_xgb)),2)))
print("STD of Training Data (in original units): "+str(round(np.std(np.exp(y_test)),2)))
print("Mean of Training Data (in original units): "+str(round(np.mean(np.exp(y_test)),2)))
print("Residual Skew: "+str(round(stats.skew(np.exp(y_pred_xgb)-np.exp(y_test)),2)))
print("Residual Kurtosis: "+str(round(stats.kurtosis(np.exp(y_pred_xgb)-np.exp(y_test)),2)))
fig, axs = plt.subplots(ncols=(3), nrows=(2), figsize=(15,8))
plt.subplots_adjust(top = 0.95, bottom=0.01, hspace=0.25, wspace=0.16)
axs[0,0].scatter(np.exp(y_test), np.exp(y_pred_xgb))
axs[0,0].set_title('Actuals (x) vs Predicteds (y) \n For 98% of Losses',fontweight = 'bold')
axs[0,0].set_xlim([0, 1000])
axs[0,0].set_ylim([0, 1000])
axs[0,1].scatter(np.exp(y_test), (np.exp(y_pred_xgb)-np.exp(y_test)))
axs[0,1].set_title('Actuals (x) vs Residuals [pred. - actual] (y)\n For 98% of Losses',fontweight = 'bold')
axs[0,1].set_xlim([0, 1000])
axs[0,1].set_ylim([-1000, 1000])
axs[0,2].hist((np.exp(y_pred_xgb)-np.exp(y_test)), bins=100)
axs[0,2].set_title('Histogram of Residuals [predicted - actual]\n For 100% of Losses',fontweight = 'bold')
axs[1,0].scatter(np.exp(y_test), np.exp(y_pred_xgb))
axs[1,0].set_title('Actuals (x) vs Predicteds (y) \n For 100% of Losses',fontweight = 'bold')
axs[1,1].scatter(np.exp(y_test), (np.exp(y_pred_xgb)-np.exp(y_test)))
axs[1,1].set_title('Actuals (x) vs Residuals [pred. - actual] (y)\n For 100% of Losses',fontweight = 'bold')
axs[1,2].set_axis_off()
plt.savefig('exhibit_b_xgb_regression',dpi=300, bbox_inches='tight')
plt.show()


# In[65]:


#cross-val
print("Cross-Validation Scoring for XGBoost")
#print('Mean Absolute Error: {}'.format(-1*round(cross_val_score(xgb_model, X_train_xgb, y_train_xgb, \
 #                                                               cv=10, scoring='neg_mean_absolute_error').mean(),2)))
#no option for RMSE in cross val score, so looking at mean absolute error
#multiply by -1 to get a positve value to take a look at it
print('R^2: {}'.format(round(cross_val_score(xgb_model, X_train_xgb, y_train_xgb, cv=10, scoring='r2').mean(),2)))


# In[66]:


#thinking about overfitting
print("XGB Overfitting Check")
print(" ")
y_pred_xgb_from_test = xgb_model.predict(X_test)
y_pred_xgb_from_train = xgb_model.predict(X_train_xgb)

print("_________Scoring Against Training Data (Scoring Against Itself)_________")
print(" ")
print("R^2:"+str(round(r2_score(y_train_xgb, y_pred_xgb_from_train),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_train_xgb),                                                                               np.exp(y_pred_xgb_from_train)),2)))
print(" ")
print("_________Scoring Against Test Data_________")
print(" ")
print("R^2:"+str(round(r2_score(y_test, y_pred_xgb_from_test),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_test),                                                                               np.exp(y_pred_xgb_from_test)),2)))

print(" ")
print("_________Train to Test MAE Ratio_________")
print(
    round(
mean_absolute_error(np.exp(y_test), np.exp(y_pred_xgb_from_test))
/
mean_absolute_error(np.exp(y_train_xgb),np.exp(y_pred_xgb_from_train))
        ,2)
)


# In[67]:


#grid search for tuning paramteters #should run in 30 second
#With all due credit to: Omar Essam in Egypt
#https://www.kaggle.com/omarito/gridsearchcv-xgbregressor-0-556-lb
# A parameter grid for XGBoost
params_xgb_searching = {'min_child_weight':[4,5], 'max_depth': [2,3,4]}

# Initialize XGB and GridSearch
xgb_searching = xgb.XGBRegressor(nthread=-1, objective="reg:squarederror", random_state=42) 

grid = GridSearchCV(xgb_searching, params_xgb_searching)
grid.fit(X_train_xgb, y_train_xgb)

# Print the r2 score
print("Potential Optimum Results from XGBoost Hyperparameter Tuning via Grid Search")
print("R^2 Score: "+str(round(r2_score(y_test, grid.best_estimator_.predict(X_test)),2))) 
#print("Mean Absolute Error: "+str(round(mean_absolute_error(y_test, grid.best_estimator_.predict(X_test)),2)))   
#commented out because these log units are not comparable


# In[68]:


top_10_xgb_features = pd.DataFrame(sorted(list(zip(features,xgb_model.feature_importances_))       ,key = lambda x: abs(x[1]),reverse=True)[:10], columns=['Feature', 'XGBoost Importance'])
top_10_xgb_features


# In[69]:


#plt.xticks(rotation=-25)
bar_count = range(len(top_10_xgb_features.Feature))
fig, axs = plt.subplots(ncols=2, figsize=(14,4))
#using a subplot method coupled with an inline parameter to have high resolution
#note: "[::-1]" reverses the column in a pandas dataframe
axs[1].set_axis_off()
axs[0].barh(bar_count, top_10_xgb_features['XGBoost Importance'][::-1],                 align='center', alpha=1)
axs[0].set_xlabel('Values')
axs[0].set_yticks(bar_count)
axs[0].set_yticklabels(top_10_xgb_features.Feature[::-1], fontsize=10)
axs[0].set_xlabel('XGBoost Importance')
axs[0].set_title("XGBoost's Feature Importances",fontweight = 'bold')

extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('exhibit_c_xgbfeatures',dpi=300, bbox_inches=extent.expanded(1.5, 1.5))
plt.show()


# [Back to Top](#a)

# # <a id='o'></a>📈 Standard Linear Regression 📈

# In[70]:


X_train1, X_test, y_train1, y_test = train_test_split(features, y, test_size = 0.2, random_state = 1)
model = LinearRegression()
model.fit(X_train1,y_train1)
y_slr_predict = model.predict(X_test)
#linear regression standard
#print r^2 and plot linear regression
print("R^2:"+str(round(r2_score(y_test, y_slr_predict),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_test),                                                                               np.exp(y_slr_predict)),2)))
print("STD of Training Data (in original units): "+str(round(np.std(np.exp(y_test)),2)))
print("Mean of Training Data (in original units): "+str(round(np.mean(np.exp(y_test)),2)))
print("Residual Skew: "+str(round(stats.skew(np.exp(y_slr_predict)-np.exp(y_test)),2)))
print("Residual Kurtosis: "+str(round(stats.kurtosis(np.exp(y_slr_predict)-np.exp(y_test)),2)))
fig, axs = plt.subplots(ncols=(3), nrows=(2), figsize=(15,8))
plt.subplots_adjust(top = 0.95, bottom=0.01, hspace=0.25, wspace=0.16)
axs[0,0].scatter(np.exp(y_test), np.exp(y_slr_predict))
axs[0,0].set_title('Actuals (x) vs Predicteds (y) \n For 98% of Losses',fontweight = 'bold')
axs[0,0].set_xlim([0, 1000])
axs[0,0].set_ylim([0, 1000])
axs[0,1].scatter(np.exp(y_test), (np.exp(y_slr_predict)-np.exp(y_test)))
axs[0,1].set_title('Actuals (x) vs Residuals [pred. - actual] (y)\n For 98% of Losses',fontweight = 'bold')
axs[0,1].set_xlim([0, 1000])
axs[0,1].set_ylim([-1000, 1000])
axs[0,2].hist((np.exp(y_slr_predict)-np.exp(y_test)), bins=100)
axs[0,2].set_title('Histogram of Residuals [predicted - actual]\n For 100% of Losses',fontweight = 'bold')
axs[1,0].scatter(np.exp(y_test), np.exp(y_slr_predict))
axs[1,0].set_title('Actuals (x) vs Predicteds (y) \n For 100% of Losses',fontweight = 'bold')
axs[1,1].scatter(np.exp(y_test), (np.exp(y_slr_predict)-np.exp(y_test)))
axs[1,1].set_title('Actuals (x) vs Residuals [pred. - actual] (y)\n For 100% of Losses',fontweight = 'bold')
axs[1,2].set_axis_off()
plt.savefig('exhibit_d_standard_linear_regression',dpi=300, bbox_inches='tight')
plt.show()


# In[71]:


#cross-val
print("Cross-Validation Scoring for Standard Linear Regression")
#print('Mean Absolute Error: {}'.format(-1*round(cross_val_score(model, X_train1, y_train1, \
 #                                                               cv=10, scoring='neg_mean_absolute_error').mean(),2)))
#commented out because these log units are not comparable
#no option for RMSE in cross val score, so looking at mean absolute error
#multiply by -1 to get a positve value to take a look at it
print('R^2: {}'.format(round(cross_val_score(LinearRegression(), X_train1, y_train1, cv=10, scoring='r2').mean(),2)))


# #### 👆🏽Thoughts about this? 👆🏽
# - The cross val score is computing logged scores, and I can't go 'underneath the hood' and use np.exp() to convert the logs back to original units. I think this magnifies the non-linearity and causes the standard linear regression to blow up. 
#   - I think this tells us not to trust the standard linear regression model. 

# In[72]:


#thinking about overfitting
print("Standard Linear Regression Overfitting Check")
print(" ")
y_pred_slr_from_test = model.predict(X_test)
y_pred_slr_from_train = model.predict(X_train1)

print("_________Scoring Against Training Data (Scoring Against Itself)_________")
print(" ")
print("R^2:"+str(round(r2_score(y_train1, y_pred_slr_from_train),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_train1),                                                                               np.exp(y_pred_slr_from_train)),2)))
print(" ")
print("_________Scoring Against Test Data_________")
print(" ")
print("R^2:"+str(round(r2_score(y_test, y_pred_slr_from_test),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_test),                                                                               np.exp(y_pred_slr_from_test)),2)))

print(" ")
print("_________Train to Test MAE Ratio_________")
print(
    round(
mean_absolute_error(np.exp(y_test), np.exp(y_pred_slr_from_test))
/
mean_absolute_error(np.exp(y_train1),np.exp(y_pred_slr_from_train))
        ,2)
)


# [Back to Top](#a)

# # <a id='p'></a>📈 Support Vector Regression 📈

# In[73]:


X_train2, X_test, y_train2, y_test = train_test_split(features, y, test_size = 0.2, random_state = 1)

#svm requires standardization becauses it uses regularization
scaler = StandardScaler()
#"To determine the scaling factors and apply the scaling to the feature data:" -Codecademy
svr_train_features = scaler.fit_transform(X_train2)
svr_test_features = scaler.transform(X_test)
#"To apply the scaling to the test data:" -Codecademy
##we do NOT want to fit to the test
##svr_train_independent_variable = scaler.fit_transform(y_train2.values.reshape(-1, 1))
#^^if we ever need it

svr_poly = SVR(kernel='poly', C=1, degree=3, gamma='auto')
#it should be ok not to scale the dependent variable, if the features are scaled
y_poly_pred = svr_poly.fit(svr_train_features, y_train2).predict(svr_test_features)
#print r^2 and plot svm
print("R^2:"+str(round(r2_score(y_test, y_poly_pred),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_test),                                                                               np.exp(y_poly_pred)),2)))
print("STD of Training Data (in original units): "+str(round(np.std(np.exp(y_test)),2)))
print("Mean of Training Data (in original units): "+str(round(np.mean(np.exp(y_test)),2)))
print("Residual Skew: "+str(round(stats.skew(np.exp(y_poly_pred)-np.exp(y_test)),2)))
print("Residual Kurtosis: "+str(round(stats.kurtosis(np.exp(y_poly_pred)-np.exp(y_test)),2)))
fig, axs = plt.subplots(ncols=(3), nrows=(2), figsize=(15,8))
plt.subplots_adjust(top = 0.95, bottom=0.01, hspace=0.25, wspace=0.16)
axs[0,0].scatter(np.exp(y_test), np.exp(y_poly_pred))
axs[0,0].set_title('Actuals (x) vs Predicteds (y) \n For 98% of Losses',fontweight = 'bold')
axs[0,0].set_xlim([0, 1000])
axs[0,0].set_ylim([0, 1000])
axs[0,1].scatter(np.exp(y_test), (np.exp(y_poly_pred)-np.exp(y_test)))
axs[0,1].set_title('Actuals (x) vs Residuals [pred. - actual] (y)\n For 98% of Losses',fontweight = 'bold')
axs[0,1].set_xlim([0, 1000])
axs[0,1].set_ylim([-1000, 1000])
axs[0,2].hist((np.exp(y_poly_pred)-np.exp(y_test)), bins=100)
axs[0,2].set_title('Histogram of Residuals [predicted - actual]\n For 100% of Losses',fontweight = 'bold')
axs[1,0].scatter(np.exp(y_test), np.exp(y_poly_pred))
axs[1,0].set_title('Actuals (x) vs Predicteds (y) \n For 100% of Losses',fontweight = 'bold')
axs[1,1].scatter(np.exp(y_test), (np.exp(y_poly_pred)-np.exp(y_test)))
axs[1,1].set_title('Actuals (x) vs Residuals [pred. - actual] (y)\n For 100% of Losses',fontweight = 'bold')
axs[1,2].set_axis_off()
plt.savefig('exhibit_e_svm_regression',dpi=300, bbox_inches='tight')
plt.show()


# In[74]:


#cross-val
print("Cross-Validation Scoring for Support Vector Regression")
#print('Mean Absolute Error: {}'.format(-1*round(cross_val_score(svr_poly, X_train2, y_train2, \
 #                                                               cv=10, scoring='neg_mean_absolute_error').mean(),2)))
#commented out because these log units are not comparable
#no option for RMSE in cross val score, so looking at mean absolute error
#multiply by -1 to get a positve value to take a look at it
print('R^2: {}'.format(round(cross_val_score(svr_poly, svr_train_features, y_train2, cv=10, scoring='r2').mean(),2)))


# In[75]:


#thinking about overfitting
print("Support Vector Machine Overfitting Check")
print(" ")
y_pred_svm_from_test = svr_poly.predict(X_test)
y_pred_svm_from_train = svr_poly.predict(X_train2)

print("_________Scoring Against Training Data (Scoring Against Itself)_________")
print(" ")
print("R^2:"+str(round(r2_score(y_train2, y_pred_svm_from_train),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_train2),                                                                               np.exp(y_pred_svm_from_train)),2)))
print(" ")
print("_________Scoring Against Test Data_________")
print(" ")
print("R^2:"+str(round(r2_score(y_test, y_pred_svm_from_test),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_test),                                                                               np.exp(y_pred_svm_from_test)),2)))

print(" ")
print("_________Train to Test MAE Ratio_________")
print(
    round(
mean_absolute_error(np.exp(y_test), np.exp(y_pred_svm_from_test))
/
mean_absolute_error(np.exp(y_train2),np.exp(y_pred_svm_from_train))
        ,2)
)


# [Back to Top](#a)

# # <a id='q'></a>📈 Lasso Regression 📈

# In[76]:


X_train3, X_test, y_train3, y_test = train_test_split(features, y, test_size = 0.2, random_state = 1)

#lasso requires standardization becauses it uses regularization
scaler = StandardScaler()
#"To determine the scaling factors and apply the scaling to the feature data:" -Codecademy
lasso_train_features = scaler.fit_transform(X_train3)
#"To apply the scaling to the test data:" -Codecademy
lasso_test_features = scaler.transform(X_test) #we do NOT want to fit to the test

reg = linear_model.Lasso(alpha=0.0005)
reg.fit(lasso_train_features,y_train3)  #i do not think it is needed to standarize the independent variable, 
#...so long as you have standardized the features
y_lasso_predict = reg.predict(lasso_test_features)
#lasso
print("R^2:"+str(round(r2_score(y_test, y_lasso_predict),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_test),                                                                               np.exp(y_lasso_predict)),2)))
print("STD of Training Data (in original units): "+str(round(np.std(np.exp(y_test)),2)))
print("Mean of Training Data (in original units): "+str(round(np.mean(np.exp(y_test)),2)))
print("Residual Skew: "+str(round(stats.skew(np.exp(y_lasso_predict)-np.exp(y_test)),2)))
print("Residual Kurtosis: "+str(round(stats.kurtosis(np.exp(y_lasso_predict)-np.exp(y_test)),2)))
fig, axs = plt.subplots(ncols=(3), nrows=(2), figsize=(15,8))
plt.subplots_adjust(top = 0.95, bottom=0.01, hspace=0.25, wspace=0.16)
axs[0,0].scatter(np.exp(y_test), np.exp(y_lasso_predict))
axs[0,0].set_title('Actuals (x) vs Predicteds (y) \n For 98% of Losses',fontweight = 'bold')
axs[0,0].set_xlim([0, 1000])
axs[0,0].set_ylim([0, 1000])
axs[0,1].scatter(np.exp(y_test), (np.exp(y_lasso_predict)-np.exp(y_test)))
axs[0,1].set_title('Actuals (x) vs Residuals [pred. - actual] (y)\n For 98% of Losses',fontweight = 'bold')
axs[0,1].set_xlim([0, 1000])
axs[0,1].set_ylim([-1000, 1000])
axs[0,2].hist((np.exp(y_lasso_predict)-np.exp(y_test)), bins=100)
axs[0,2].set_title('Histogram of Residuals [predicted - actual]\n For 100% of Losses',fontweight = 'bold')
axs[1,0].scatter(np.exp(y_test), np.exp(y_lasso_predict))
axs[1,0].set_title('Actuals (x) vs Predicteds (y) \n For 100% of Losses',fontweight = 'bold')
axs[1,1].scatter(np.exp(y_test), (np.exp(y_lasso_predict)-np.exp(y_test)))
axs[1,1].set_title('Actuals (x) vs Residuals [pred. - actual] (y)\n For 100% of Losses',fontweight = 'bold')
axs[1,2].set_axis_off()
plt.savefig('exhibit_f_lasso_regression',dpi=300, bbox_inches='tight')
plt.show()


# In[77]:


#cross-val
print("Cross-Validation Scoring for Lasso Regression")
#print('Mean Absolute Error: {}'.format(-1*round(cross_val_score(reg, X_train3, y_train3, \
 #                                                               cv=10, scoring='neg_mean_absolute_error').mean(),2)))
#commented out because these log units are not comparable
#no option for RMSE in cross val score, so looking at mean absolute error
#multiply by -1 to get a positve value to take a look at it
print('R^2: {}'.format(round(cross_val_score(reg, lasso_train_features, y_train3, cv=10, scoring='r2').mean(),2)))


# In[78]:


#thinking about overfitting
print("Lasso Overfitting Check")
print(" ")
y_pred_lasso_from_test = reg.predict(X_test)
y_pred_lasso_from_train = reg.predict(X_train3)

print("_________Scoring Against Training Data (Scoring Against Itself)_________")
print(" ")
print("R^2:"+str(round(r2_score(y_train3, y_pred_lasso_from_train),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_train3),                                                                               np.exp(y_pred_lasso_from_train)),2)))
print(" ")
print("_________Scoring Against Test Data_________")
print(" ")
print("R^2:"+str(round(r2_score(y_test, y_pred_lasso_from_test),4)))
print("Mean Absolute Error (in original units): "+str(round(mean_absolute_error(np.exp(y_test),                                                                               np.exp(y_pred_lasso_from_test)),2)))

print(" ")
print("_________Train to Test MAE Ratio_________")
print(
    round(
mean_absolute_error(np.exp(y_test), np.exp(y_pred_lasso_from_test))
/
mean_absolute_error(np.exp(y_train3),np.exp(y_pred_lasso_from_train))
        ,2)
)


# [Back to Top](#a)

# ## <a id='r'></a>Step 6 - Side Project - Loss Classification 
# - In practice, we would be concerned not only to predict the amount of a loss, but who would experince a loss.
# - This can be complciatd because of unbalanced datasets of losses vs. non-losses.
# - For this overall project, I focused on predcting loss amount, when we knew there was a loss. 
# - **But**, for this particular side project, I want to show how we might go about solving that problem.
# - For classification I will use:
#   - Standard Logistic Regression Classifier
#   - XGBoost
# - **Please note:** I have not adjusted the probability threshold for either classification model. As such, although accuracy is decent, recall is too low. We can balance the recall versus the precision versus the accuracy by adjusting the classification model's probability threshold (how it decides what numerical the threshold is to determine a given classification).
#   - Please also note, to my understanding, that some classification model, especially scikit-learn's logistic regression classifier make it difficult to adjust this threshold, but there are workaround.
# - Metrics Used to Evaluate the Classifications:
#   - Overfitting
#     - Accuracy on Train vs. Accuracy on Test
#     - The higher the ratio, the more overfit
#   - Accuaracy
#   - Precision
#   - Recall
# - All models are cross validated on the training data for accuracy, precision, and recall.

# In[80]:


#prepare the classification Dataframe from before
insurance_ml_classification.shape


# In[81]:


msno.matrix(insurance_ml_classification,  color = (.0,.0,.2))


# #### 👆🏽Thoughts about this? 👆🏽, and the next few blocks of code....
# - These two white-speckeled columns, just as before, are claim amount and claim month.
#   - We can drop them because that's not what we're looking for here.
# - We also don't care about loss_cost anymore because we're just asking the binary question: Loss? Yes or No 
# - We will also drop exposure units for the same reasons as we did above (it correaltes very highly with employee).

# In[82]:


insurance_ml_classification_processed = insurance_ml_classification.drop(['claim_amount',                                                   'loss_cost','claim_month', 'exposure_units'],axis = 1)


# In[83]:


msno.matrix(insurance_ml_classification_processed,  color = (.0,.0,.2))


# In[84]:


#address skew in features
#For this block, credit goes to Alexandru Papiu 
#(https://www.kaggle.com/apapiu/regularized-linear-models)
#log transform skewed numeric features:
continous_features_classification = ['locations','employees','internet_complaints'] 
#exposure units have already been taken out
skewed_feats = insurance_ml_classification_processed[continous_features_classification].apply(                                                                                lambda x: stats.skew(x)) 
#compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

insurance_ml_classification_processed[skewed_feats] = np.log1p(insurance_ml_classification_processed[skewed_feats])


# [Back to Top](#a)

# # <a id='s'></a> 📊 Logistic Regression Classification 📊

# In[86]:


classification_features = insurance_ml_classification_processed.drop(['was_there_a_loss'], axis = 1)
classification_outcome = insurance_ml_classification_processed.was_there_a_loss
train_features, test_features, train_labels, test_labels = train_test_split(classification_features,                                                                            classification_outcome, test_size = 0.2)
#normalize this, since sklearn's logistic regression uses regularization
scaler = StandardScaler()
#"To determine the scaling factors and apply the scaling to the feature data:" -Codecademy
classification_train_features = scaler.fit_transform(train_features)
#"To apply the scaling to the test data:" -Codecademy
classification_test_features = scaler.transform(test_features) #we do NOT want to fit to the test

#run model
log_model = LogisticRegression(solver="liblinear") #to remove warning
log_model.fit(classification_train_features, train_labels)
print("_________Logistic Regression Classification_________")
print("")
print("Scored Against Itself")
print('Accuracy Score: {}'.format(round(log_model.score(classification_train_features, train_labels),3)))
print("")
print("Scored Against Test Data")
print('Accuracy Score: {}'.format(round(log_model.score(classification_test_features, test_labels),3)))


# In[87]:


#cross-val
print("_________Cross-Validation Scoring for Logistic Regression Classification_________")
print('Accuracy: {}'.format(round(cross_val_score(log_model, classification_train_features, train_labels,                                                                 cv=10, scoring='accuracy').mean(),3)))
print('Precision: {}'.format(round(cross_val_score(log_model, classification_train_features, train_labels,                                                                 cv=10, scoring='precision').mean(),3)))
print('Recall: {}'.format(round(cross_val_score(log_model, classification_train_features, train_labels,                                                                 cv=10, scoring='recall').mean(),3)))


# In[88]:


log_model_predictions = log_model.predict(classification_test_features)
cm = confusion_matrix(test_labels, log_model_predictions)
#print(cm) #this is the barebones confusion matrix

#all credit due to: Michael Galarnyk, "Logistic Regression using Python (scikit-learn)", Towards Data Science 
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual Label');
plt.xlabel('Predicted Label');
all_sample_title = 'Logistic Regression Classification \n Accuracy Score: {0:.3f}'.format(                                                log_model.score(classification_test_features, test_labels))
plt.title(all_sample_title, size = 15)
plt.savefig('exhibit_g_logistic_regression_confusion_matrix',dpi=300, bbox_inches='tight')
print("Accuracy: "+str(accuracy_score(test_labels, log_model_predictions))) #this is just a little check at the end
print("Precision: "+str(precision_score(test_labels, log_model_predictions))) #this is just a little check at the end
print("Recall: "+str(recall_score(test_labels, log_model_predictions))) #this is just a little check at the end


# [Back to Top](#a)

# # <a id='t'></a> 📊 XGBoost Classification 📊

# In[89]:


classification_features_xgb = insurance_ml_classification_processed.drop(['was_there_a_loss'], axis = 1)
classification_outcome_xgb = insurance_ml_classification_processed.was_there_a_loss
train_features, test_features, train_labels, test_labels = train_test_split(classification_features_xgb,                                                                    classification_outcome_xgb, test_size = 0.2)

xgb_classy = xgb.XGBClassifier()

xgb_classy.fit(train_features,train_labels)

xgb_classy_predictions = xgb_classy.predict(test_features)


#run model
print("_________XGBoost Regression Classification_________")
print("")
print("Scored Against Itself")
print('Accuracy Score: {}'.format(round(xgb_classy.score(train_features, train_labels),3)))
print("")
print("Scored Against Test Data")
print('Accuracy Score: {}'.format(round(xgb_classy.score(test_features, test_labels),3)))


# In[91]:


#cross-val #this will run about 70 seconds per print
print("_________Cross-Validation Scoring for XGBoost Classification_________")
print('Accuracy: {}'.format(round(cross_val_score(xgb_classy, train_features, train_labels,                                                                 cv=10, scoring='accuracy').mean(),3)))
print('Precision: {}'.format(round(cross_val_score(xgb_classy, train_features, train_labels,                                                                 cv=10, scoring='precision').mean(),3)))
print('Recall: {}'.format(round(cross_val_score(xgb_classy, train_features, train_labels,                                                                 cv=10, scoring='recall').mean(),3)))


# In[90]:


xgb_classy_predictions = xgb_classy.predict(test_features) #redundant, but copied here too so I can see it
xgb_cm = confusion_matrix(test_labels, xgb_classy_predictions)
#print(cm) #this is the barebones confusion matrix

#all credit due to: Michael Galarnyk, "Logistic Regression using Python (scikit-learn)", Towards Data Science 
plt.figure(figsize=(9,9))
sns.heatmap(xgb_cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual Label');
plt.xlabel('Predicted Label');
all_sample_title = 'XGBoost Regression Classification \n Accuracy Score: {0:.3f}'.format(                                                                xgb_classy.score(test_features, test_labels))
plt.title(all_sample_title, size = 15)
plt.savefig('exhibit_h_xgboost_regression_confusion_matrix',dpi=300, bbox_inches='tight')
print("Accuracy: "+str(accuracy_score(test_labels, xgb_classy_predictions))) #this is just a little check at the end
print("Precision: "+str(precision_score(test_labels, xgb_classy_predictions))) #this is just a little check at the end
print("Recall: "+str(recall_score(test_labels, xgb_classy_predictions))) #this is just a little check at the end


# [Back to Top](#a)

# ### <font color = blue> <a id='u'></a>  Problem 4: Text Mining </font>
# 
# <font color = blue> You are given a dataset of 25,000 news article titles about company activities scrapped from the web (**News_Titles.csv**). Each news article has been manually categorized into one of 5 categories:
# - Earnings Release
# - Business Expansions
# - Key Personnel Changes
# - Conferences & Presentations
# - Mergers & Acquisitions
# 
# Build a text classification model to automatically classify future news articles into one of these 5 categories, and present some measurements for how well your model performs. Some suggestions to start:
# - You're given unstructured text data, can you extract some useful features from it?
# - Are there any modifications or standardizations you can apply to the text?
# - What are the top keywords/phrases by category?
# 
# This final exercise is independent from the previous ones. </font>

# [Back to Top](#a)

# ## <a id='v'></a>Step 6 - Text Classification of News Articles 
# - This kind of problem can be elegantly solved by a Mulinomial Naive Bayes classifier.
# - The solution I use here is inspired by Jake VanderPlas, the author of "The Python Data Science Handbook."
#   - I met Jake in the comments section of an altair-related question I once posted in Stack Overflow

# In[92]:


news = pd.read_csv('News_Titles.csv')


# In[93]:


news.head(3)


# In[94]:


news_labels = news.news_category
news_features = news.news_article_title


# In[95]:


X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(news_features, news_labels,                                                                        test_size=0.20, random_state=42)


# In[96]:


mnb_model.fit(X_train_news, y_train_news)
labels = mnb_model.predict(X_test_news)


# In[97]:


print("______Example of the Model at Work______")
print(" ")
print("Actual News Headline:")
print(news.news_article_title[0])
print(" ")
print("Predicted Category:")
print(mnb_model.predict([news.news_article_title[0]]))
print(" ")
print("Actual Category:")
print(news.news_category[0])
print(" ")
print("✅✅✅✅")


# In[98]:


accuracy_score(y_test_news, labels)


# In[102]:


mat = confusion_matrix(y_test_news, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=unique_labels(y_test_news, labels), yticklabels=unique_labels(y_test_news, labels))
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.savefig('exhibit_x9_text_classification_confusion_matrix',dpi=300, bbox_inches='tight')
print("Overall Accuracy: "+str(accuracy_score(y_test_news, labels))) 

uniques = unique_labels(y_test_news, labels)
prec_score = precision_score(y_test_news, labels, average=None)
reca_score = recall_score(y_test_news, labels, average=None)
print(" ")
print("_______________Precision_______________")
for i in range(len(uniques)):
    print(str(uniques[i])+" "+str(prec_score[i]))
print(" ")
print("_______________Recall_______________")
for i in range(len(uniques)):
    print(str(uniques[i])+" "+str(reca_score[i]))


# [Back to Top](#a)

# # <a id='w'></a> Thank you so much for your time and consideration!

# ## Best,
# ## <font color = green> George John Jordan Thomas Aquinas Hayward, Optimist </font>
# 

# <a href="https://www.georgejohnjordanthomasaquinashayward.com/"><img src="https://i.ibb.co/1ZHzKHP/george-hayward-data-scientist.jpg" alt="george-hayward-data-scientist" border="0"></a>

# [#TheLordAlwaysDelivers](https://youtu.be/in2MOJ-Hb8g)
