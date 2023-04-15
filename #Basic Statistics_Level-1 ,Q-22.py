# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:35:48 2023

@author: kailas
"""

Q 22) Calculate the Z scores of  90% confidence interval,94% confidence interval, 60% confidence interval 



from scipy import stats
from scipy.stats import norm 
# Z-score of 90% confidence interval 
stats.norm.ppf(0.95)
1.6448536269514722
# Z-score of 94% confidence interval
stats.norm.ppf(0.97)
1.8807936081512509
# Z-score of 60% confidence interval
stats.norm.ppf(0.8)
0.8416212335729143
