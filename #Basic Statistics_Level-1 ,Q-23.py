# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:40:04 2023

@author: kailas
"""

Q 23) Calculate the t scores of 95% confidence interval, 96% confidence interval, 99% confidence interval for sample size of 25



from scipy import stats
from scipy.stats import norm
# t scores of 95% confidence interval for sample size of 25
stats.t.ppf(0.975,24)  # df = n-1 = 24
2.0638985616280205
# t scores of 96% confidence interval for sample size of 25
stats.t.ppf(0.98,24)
2.1715446760080677
# t scores of 99% confidence interval for sample size of 25
stats.t.ppf(0.995,24)
2.796939504772804
