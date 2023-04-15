# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:18:47 2023

@author: kailas
"""
Q 21) Check whether the data follows normal distribution
a)	Check whether the MPG of Cars follows Normal Distribution 
        Dataset: Cars.csv


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
cars=pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/Cars (2).csv")
cars.head()

cars['MPG'].mean()
34.422075728024666
cars['MPG'].median()
35.15272697
cars['MPG'].mode()
cars['MPG'].hist()

sns.distplot(cars['MPG'])
plt.grid(True)
plt.show()


cars['MPG'].skew()
-0.17794674747025727
cars['MPG'].kurt()
-0.6116786559430913
#From above plot and values we can say that data is fairly symmetrical, i.e fairly normally distributed.
 

############################################################################################################################

b)	Check Whether the Adipose Tissue (AT) and Waist Circumference(Waist)  from wc-at data set  follows Normal Distribution 
       Dataset: wc-at.csv



import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
df=pd.read_csv("D:/data science assignment/Assignments/#Basic Statistics_Level-1/wc-at.csv")
df.head()
df.mean()
df.median()
df.mode()
# waist is multimodal, AT is bimodal data

sns.distplot(df['Waist'])
plt.show()


sns.distplot(df['AT'])
plt.show()


sns.boxplot(df['AT'])
plt.show()

# mean> median, right whisker is larger than left whisker, data is positively skewed.


sns.boxplot(df['Waist'])
plt.show()

## mean> median, both the whisker are of same lenght, median is slightly shifted towards left. Data is fairly symetrically distributed.


 
 
 
 
 
 
 
 
 
 
 
 
 