# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:43:34 2023

@author: kailas
"""
Q 24)   A Government  company claims that an average light bulb lasts 270 days. A researcher randomly selects 18 bulbs for testing. The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. If the CEOs claim were true, what is the probability that 18 randomly selected bulbs would have an average life of no more than 260 days
Hint:  
   rcode   pt(tscore,df)  
 df  degrees of freedom






from scipy import stats
from scipy.stats import norm
# Assume Null Hypothesis is: Ho = Avg life of Bulb >= 260 days
# Alternate Hypothesis is: Ha = Avg life of Bulb < 260 days
# find t-scores at x=260; t=(s_mean-P_mean)/(s_SD/sqrt(n))
t=(260-270)/(90/18**0.5)
t
-0.4714045207910317
# Find P(X>=260) for null hypothesis
# p_value=1-stats.t.cdf(abs(t_scores),df=n-1)... Using cdf function
p_value=1-stats.t.cdf(abs(-0.4714),df=17)
p_value
0.32167411684460556
#  OR p_value=stats.t.sf(abs(t_score),df=n-1)... Using sf function
p_value=stats.t.sf(abs(-0.4714),df=17)
p_value
0.32167411684460556
