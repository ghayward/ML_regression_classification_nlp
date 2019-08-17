## Executive Summary

This analysis seeks two objectives:

* (1)   discover an insured company's most important features for determining its loss cost, given that a loss occurs.
* (2)   classify a news article's category based on its headline.

First, this analysis concludes that the following insured company features matter most in determining loss cost:

* number of employees
* sector
* risk zone
* industry

The winning model was XGBoost Regression, with an **R^2 of 0.8346** and a Mean Absolute Error of **77.79 units of loss cost.**

* _Credit is given to [Tianqi Chen](https://github.com/tqchen) for creating XGBoost._

Second, this analysis correctly picked the category of the news headline **98.26%** of the time.

The winning model was a Multinomial Naïve Bayes Classifier enhanced with a TF-IDF vectorizer.

* _Credit is given to [Jake VanderPlas](https://github.com/jakevdp) for posting the working pipeline on his Github._

Credit to God, my Mother, family and friends.

All errors are my own.

Best, 
George John Jordan Thomas Aquinas Hayward, Optimist
