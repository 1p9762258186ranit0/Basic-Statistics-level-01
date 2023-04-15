# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:08:15 2023

@author: kailas
"""
"D:\data science assignment\Assignments\#Basic Statistics_Level-1\Q7.csv"




import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.stats as stats
question 7
q7 = pd.read_csv("C:/Users/Moin Dalvi/Documents/EXcelR Study and Assignment Material/Data Science Assignments/Basic Statistics 1/Q7.csv")
q7
Unnamed: 0	Points	Score	Weigh
0	Mazda RX4	3.90	2.620	16.46
1	Mazda RX4 Wag	3.90	2.875	17.02
2	Datsun 710	3.85	2.320	18.61
3	Hornet 4 Drive	3.08	3.215	19.44
4	Hornet Sportabout	3.15	3.440	17.02
5	Valiant	2.76	3.460	20.22
6	Duster 360	3.21	3.570	15.84
7	Merc 240D	3.69	3.190	20.00
8	Merc 230	3.92	3.150	22.90
9	Merc 280	3.92	3.440	18.30
10	Merc 280C	3.92	3.440	18.90
11	Merc 450SE	3.07	4.070	17.40
12	Merc 450SL	3.07	3.730	17.60
13	Merc 450SLC	3.07	3.780	18.00
14	Cadillac Fleetwood	2.93	5.250	17.98
15	Lincoln Continental	3.00	5.424	17.82
16	Chrysler Imperial	3.23	5.345	17.42
17	Fiat 128	4.08	2.200	19.47
18	Honda Civic	4.93	1.615	18.52
19	Toyota Corolla	4.22	1.835	19.90
20	Toyota Corona	3.70	2.465	20.01
21	Dodge Challenger	2.76	3.520	16.87
22	AMC Javelin	3.15	3.435	17.30
23	Camaro Z28	3.73	3.840	15.41
24	Pontiac Firebird	3.08	3.845	17.05
25	Fiat X1-9	4.08	1.935	18.90
26	Porsche 914-2	4.43	2.140	16.70
27	Lotus Europa	3.77	1.513	16.90
28	Ford Pantera L	4.22	3.170	14.50
29	Ferrari Dino	3.62	2.770	15.50
30	Maserati Bora	3.54	3.570	14.60
31	Volvo 142E	4.11	2.780	18.60
q7.describe()
Points	Score	Weigh
count	32.000000	32.000000	32.000000
mean	3.596563	3.217250	17.848750
std	0.534679	0.978457	1.786943
min	2.760000	1.513000	14.500000
25%	3.080000	2.581250	16.892500
50%	3.695000	3.325000	17.710000
75%	3.920000	3.610000	18.900000
max	4.930000	5.424000	22.900000
stats.median(q7["Points"])
3.6950000000000003
q7.median()
Points     3.695
Score      3.325
Weigh     17.710
dtype: float64
q7.mode()
Points	Score	Weigh
0	3.07	3.44	17.02
1	3.92	NaN	18.90
stats.mode(q7['Points'])
3.92
stats.mode(q7['Score'])
3.44
stats.mode(q7['Weigh'])
17.02
q7.var()
Points    0.285881
Score     0.957379
Weigh     3.193166
dtype: float64
q7.rename(columns={'Unnamed: 0':'Cars'}, inplace = True)
q7
Cars	Points	Score	Weigh
0	Mazda RX4	3.90	2.620	16.46
1	Mazda RX4 Wag	3.90	2.875	17.02
2	Datsun 710	3.85	2.320	18.61
3	Hornet 4 Drive	3.08	3.215	19.44
4	Hornet Sportabout	3.15	3.440	17.02
5	Valiant	2.76	3.460	20.22
6	Duster 360	3.21	3.570	15.84
7	Merc 240D	3.69	3.190	20.00
8	Merc 230	3.92	3.150	22.90
9	Merc 280	3.92	3.440	18.30
10	Merc 280C	3.92	3.440	18.90
11	Merc 450SE	3.07	4.070	17.40
12	Merc 450SL	3.07	3.730	17.60
13	Merc 450SLC	3.07	3.780	18.00
14	Cadillac Fleetwood	2.93	5.250	17.98
15	Lincoln Continental	3.00	5.424	17.82
16	Chrysler Imperial	3.23	5.345	17.42
17	Fiat 128	4.08	2.200	19.47
18	Honda Civic	4.93	1.615	18.52
19	Toyota Corolla	4.22	1.835	19.90
20	Toyota Corona	3.70	2.465	20.01
21	Dodge Challenger	2.76	3.520	16.87
22	AMC Javelin	3.15	3.435	17.30
23	Camaro Z28	3.73	3.840	15.41
24	Pontiac Firebird	3.08	3.845	17.05
25	Fiat X1-9	4.08	1.935	18.90
26	Porsche 914-2	4.43	2.140	16.70
27	Lotus Europa	3.77	1.513	16.90
28	Ford Pantera L	4.22	3.170	14.50
29	Ferrari Dino	3.62	2.770	15.50
30	Maserati Bora	3.54	3.570	14.60
31	Volvo 142E	4.11	2.780	18.60
q7.set_index(('Cars'), inplace = True)
q7
Cars	Points	Score	Weigh
0	Mazda RX4	3.90	2.620	16.46
1	Mazda RX4 Wag	3.90	2.875	17.02
2	Datsun 710	3.85	2.320	18.61
3	Hornet 4 Drive	3.08	3.215	19.44
4	Hornet Sportabout	3.15	3.440	17.02
5	Valiant	2.76	3.460	20.22
6	Duster 360	3.21	3.570	15.84
7	Merc 240D	3.69	3.190	20.00
8	Merc 230	3.92	3.150	22.90
9	Merc 280	3.92	3.440	18.30
10	Merc 280C	3.92	3.440	18.90
11	Merc 450SE	3.07	4.070	17.40
12	Merc 450SL	3.07	3.730	17.60
13	Merc 450SLC	3.07	3.780	18.00
14	Cadillac Fleetwood	2.93	5.250	17.98
15	Lincoln Continental	3.00	5.424	17.82
16	Chrysler Imperial	3.23	5.345	17.42
17	Fiat 128	4.08	2.200	19.47
18	Honda Civic	4.93	1.615	18.52
19	Toyota Corolla	4.22	1.835	19.90
20	Toyota Corona	3.70	2.465	20.01
21	Dodge Challenger	2.76	3.520	16.87
22	AMC Javelin	3.15	3.435	17.30
23	Camaro Z28	3.73	3.840	15.41
24	Pontiac Firebird	3.08	3.845	17.05
25	Fiat X1-9	4.08	1.935	18.90
26	Porsche 914-2	4.43	2.140	16.70
27	Lotus Europa	3.77	1.513	16.90
28	Ford Pantera L	4.22	3.170	14.50
29	Ferrari Dino	3.62	2.770	15.50
30	Maserati Bora	3.54	3.570	14.60
31	Volvo 142E	4.11	2.780	18.60
plt.hist(q7["Points"], bins = 10, edgecolor= 'black')
plt.show()

plt.boxplot(x = 'Points', data =q7)
plt.xlabel('Points')
plt.ylabel('Density')
plt.savefig("PointsInferences.png")
plt.show()

plt.hist(q7["Score"], bins = 20, edgecolor = 'y')
plt.show()

plt.boxplot(x = 'Score', data= q7)
plt.xlabel('Scores')
plt.ylabel('Density')
plt.savefig("ScoresInferences.png")
plt.show()

plt.hist(q7["Weigh"], bins=20, edgecolor = 'red')
plt.show()

plt.boxplot(x= "Weigh", data = q7)
plt.xlabel('Weigh')
plt.ylabel('Density')
plt.savefig("WeighInferences.png")
plt.show()

plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Points"])
plt.yticks(fontsize=14)
plt.show()

plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Score"])
plt.yticks(fontsize=14)
plt.show()

plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Weigh"])
plt.yticks(fontsize=14)
plt.show()

 
 
question 6
def expected_value(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)
    return (values * weights).sum() / weights.sum()
c_count = [1,4,3,5,6,2]
ch_prob = [0.015,0.20,0.65,0.005,0.01,0.120]
expected_value(c_count, ch_prob)
3.09
question 8
weigh = [108,110,123,134,135,145,167,187,199]
probs = [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]
expected_value(weigh, probs)
145.33333333333331
ch = 1/9
ch
0.1111111111111111
question 11
from scipy import stats
conf_94 =stats.t.interval(alpha = 0.97, df=1999, loc=200, scale=30/np.sqrt(2000))
print(np.round(conf_94,0))
print(conf_94)
conf_94 =stats.t.interval(alpha = 0.94, df=1999, loc=200, scale=30/np.sqrt(2000))
print(np.round(conf_94,0))
print(conf_94)
[199. 201.]
(198.7376089443071, 201.2623910556929)
conf_98= stats.t.interval(alpha = 0.98, df = 1999, loc = 200, scale=30/np.sqrt(2000))
print(np.round(conf_98,0))
print(conf_98)
[198. 202.]
(198.4381860483216, 201.5618139516784)
conf_96 = stats.t.interval( alpha = 0.96, df = 1999 , loc = 200 , scale = 30/np.sqrt(2000))
print(np.round(conf_96,0))
print(conf_96)
[199. 201.]
(198.6214037429732, 201.3785962570268)
conf_z_94 = stats.norm.interval(0.94, loc = 200, scale = 30/np.sqrt(2000))
np.round(conf_z_94,0)
array([199., 201.])
conf_z_96 = stats.norm.interval(0.96, loc = 200, scale = 30/np.sqrt(2000))
np.round(conf_z_94,0)
array([199., 201.])
conf_z_98 =  stats.norm.interval(0.98, loc=200,scale=30/np.sqrt(2000))
np.round(conf_z_98,0)
array([198., 202.])
 
stats.t.ppf(0.03,df=1999)
-1.8818614764780115
stats.t.ppf(0.01,df=1999)
-2.3282147761069725
stats.t.ppf(0.02,df=1999)
-2.055089962825778
question 12
q12 = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
stat.mean(q12)
41
stat.median(q12)
40.5
stat.variance(q12)
25.529411764705884
stat.stdev(q12)
5.05266382858645
q12_df.describe()
students	marks
count	18.000000	18.000000
mean	9.500000	41.000000
std	5.338539	5.052664
min	1.000000	34.000000
25%	5.250000	38.250000
50%	9.500000	40.500000
75%	13.750000	41.750000
max	18.000000	56.000000
q12_df = pd.DataFrame({'students':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                    'marks':(q12)})
q12_df
students	marks
0	1	34
1	2	36
2	3	36
3	4	38
4	5	38
5	6	39
6	7	39
7	8	40
8	9	40
9	10	41
10	11	41
11	12	41
12	13	41
13	14	42
14	15	42
15	16	45
16	17	49
17	18	56
q12_df.set_index('students')
marks
students	
1	34
2	36
3	36
4	38
5	38
6	39
7	39
8	40
9	40
10	41
11	41
12	41
13	41
14	42
15	42
16	45
17	49
18	56
Question 24
x_bar = 260
pop_mean = 270
t_value = (260-270)/(90/np.sqrt(18))
t_value
-0.4714045207910317
1-stats.t.cdf(abs(t_value),df = 17)
0.32167253567098353
question 20
q20 = pd.read_csv("C:/Users/Moin Dalvi/Documents/EXcelR Study and Assignment Material/Data Science Assignments/Basic Statistics 1/Cars.csv")
q20
HP	MPG	VOL	SP	WT
0	49	53.700681	89	104.185353	28.762059
1	55	50.013401	92	105.461264	30.466833
2	55	50.013401	92	105.461264	30.193597
3	70	45.696322	92	113.461264	30.632114
4	53	50.504232	92	104.461264	29.889149
...	...	...	...	...	...
76	322	36.900000	50	169.598513	16.132947
77	238	19.197888	115	150.576579	37.923113
78	263	34.000000	50	151.598513	15.769625
79	295	19.833733	119	167.944460	39.423099
80	236	12.101263	107	139.840817	34.948615
81 rows × 5 columns

from scipy import stats
q20.describe()
HP	MPG	VOL	SP	WT
count	81.000000	81.000000	81.000000	81.000000	81.000000
mean	117.469136	34.422076	98.765432	121.540272	32.412577
std	57.113502	9.131445	22.301497	14.181432	7.492813
min	49.000000	12.101263	50.000000	99.564907	15.712859
25%	84.000000	27.856252	89.000000	113.829145	29.591768
50%	100.000000	35.152727	101.000000	118.208698	32.734518
75%	140.000000	39.531633	113.000000	126.404312	37.392524
max	322.000000	53.700681	160.000000	169.598513	52.997752
Prob_MPG_greater_than_38 = np.round(1 - stats.norm.cdf(38, loc= q20.MPG.mean(), scale= q20.MPG.std()),3)
print('P(MPG>38)=',Prob_MPG_greater_than_38)
P(MPG>38)= 0.348
prob_MPG_less_than_40 = np.round(stats.norm.cdf(40, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('P(MPG<40)=',prob_MPG_less_than_40)
P(MPG<40)= 0.729
prob_MPG_greater_than_20 = np.round(1-stats.norm.cdf(20, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('p(MPG>20)=',(prob_MPG_greater_than_20))
p(MPG>20)= 0.943
prob_MPG_less_than_50 = np.round(stats.norm.cdf(50, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('P(MPG<50)=',(prob_MPG_less_than_50))
P(MPG<50)= 0.956
prob_MPG_greaterthan20_and_lessthan50= (prob_MPG_less_than_50) - (prob_MPG_greater_than_20)
print('P(20<MPG<50)=',(prob_MPG_greaterthan20_and_lessthan50))
P(20<MPG<50)= 0.013000000000000012
Question 22
# z value for 90% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.05),4))
Z score for 60% Conifidence Intervla = -1.6449
# z value for 94% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.03),4))
Z score for 60% Conifidence Intervla = -1.8808
# z value for 60% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.2),4))
Z score for 60% Conifidence Intervla = -0.8416
# t score for 95% confidence interval
print('T score for 95% Confidence Interval =',np.round(stats.t.ppf(0.025,df=24),4))
T score for 95% Confidence Interval = -2.0639
# t value for 94% confidence interval
print('T score for 94% Confidence Inteval =',np.round(stats.t.ppf(0.03,df=24),4))
T score for 94% Confidence Inteval = -1.974
# t value for 99% Confidence Interval
print('T score for 95% Confidence Interval =',np.round(stats.t.ppf(0.005,df=24),4))
T score for 95% Confidence Interval = -2.7969
Question 9
q9a = pd.read_csv("C:/Users/Moin Dalvi/Documents/EXcelR Study and Assignment Material/Data Science Assignments/Basic Statistics 1/Q9_a.csv", index_col = 'Index')
q9a
speed	dist
Index		
1	4	2
2	4	10
3	7	4
4	7	22
5	8	16
6	9	10
7	10	18
8	10	26
9	10	34
10	11	17
11	11	28
12	12	14
13	12	20
14	12	24
15	12	28
16	13	26
17	13	34
18	13	34
19	13	46
20	14	26
21	14	36
22	14	60
23	14	80
24	15	20
25	15	26
26	15	54
27	16	32
28	16	40
29	17	32
30	17	40
31	17	50
32	18	42
33	18	56
34	18	76
35	18	84
36	19	36
37	19	46
38	19	68
39	20	32
40	20	48
41	20	52
42	20	56
43	20	64
44	22	66
45	23	54
46	24	70
47	24	92
48	24	93
49	24	120
50	25	85
print('For Cars Speed', "Skewness value=", np.round(q9a.speed.skew(),2), 'and' , 'Kurtosis value=', np.round(q9a.speed.kurt(),2))
For Cars Speed Skewness value= -0.12 and Kurtosis value= -0.51
print('Skewness value =', np.round(q9a.dist.skew(),2),'and', 'Kurtosis value =', np.round(q9a.dist.kurt(),2), 'for Cars Distance')
Skewness value = 0.81 and Kurtosis value = 0.41 for Cars Distance
q9b =pd.read_csv("C:/Users/Moin Dalvi/Documents/EXcelR Study and Assignment Material/Data Science Assignments/Basic Statistics 1/Q9_b.csv")
q9b
Unnamed: 0	SP	WT
0	1	104.185353	28.762059
1	2	105.461264	30.466833
2	3	105.461264	30.193597
3	4	113.461264	30.632114
4	5	104.461264	29.889149
...	...	...	...
76	77	169.598513	16.132947
77	78	150.576579	37.923113
78	79	151.598513	15.769625
79	80	167.944460	39.423099
80	81	139.840817	34.948615
81 rows × 3 columns

q9b.rename(columns = {'Unnamed: 0':'Index'}, inplace = True)
q9b
Index	SP	WT
0	1	104.185353	28.762059
1	2	105.461264	30.466833
2	3	105.461264	30.193597
3	4	113.461264	30.632114
4	5	104.461264	29.889149
...	...	...	...
76	77	169.598513	16.132947
77	78	150.576579	37.923113
78	79	151.598513	15.769625
79	80	167.944460	39.423099
80	81	139.840817	34.948615
81 rows × 3 columns

q9b
Index	SP	WT
0	1	104.185353	28.762059
1	2	105.461264	30.466833
2	3	105.461264	30.193597
3	4	113.461264	30.632114
4	5	104.461264	29.889149
...	...	...	...
76	77	169.598513	16.132947
77	78	150.576579	37.923113
78	79	151.598513	15.769625
79	80	167.944460	39.423099
80	81	139.840817	34.948615
81 rows × 3 columns

print('For SP Skewness =', np.round(q9b.SP.skew(),2), 'kurtosis =', np.round(q9b.SP.kurt(),2))
For SP Skewness = 1.61 kurtosis = 2.98
print('For WT Skewness =', np.round(q9b.WT.skew(),2), 'Kurtosis =', np.round(q9b.WT.kurt(),2))
For WT Skewness = -0.61 Kurtosis = 0.95
Question 21
q21a = pd.read_csv("C:/Users/Moin Dalvi/Documents/EXcelR Study and Assignment Material/Data Science Assignments/Basic Statistics 1/Cars.csv")
q21a
HP	MPG	VOL	SP	WT
0	49	53.700681	89	104.185353	28.762059
1	55	50.013401	92	105.461264	30.466833
2	55	50.013401	92	105.461264	30.193597
3	70	45.696322	92	113.461264	30.632114
4	53	50.504232	92	104.461264	29.889149
...	...	...	...	...	...
76	322	36.900000	50	169.598513	16.132947
77	238	19.197888	115	150.576579	37.923113
78	263	34.000000	50	151.598513	15.769625
79	295	19.833733	119	167.944460	39.423099
80	236	12.101263	107	139.840817	34.948615
81 rows × 5 columns

import numpy as np
import matplotlib.pyplot as plt

mean, cov = [0, 0], [(1, .6), (.6, 1)]
x, y = np.random.multivariate_normal(mean, cov, 100).T
y += x + 1

f, ax = plt.subplots(figsize=(6, 6))

ax.scatter(x, y, c=".3")
ax.set(xlim=(-3, 3), ylim=(-3, 3))

# Plot your initial diagonal line based on the starting
# xlims and ylims.
diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

def on_change(axes):
    # When this function is called it checks the current
    # values of xlim and ylim and modifies diag_line
    # accordingly.
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    diag_line.set_data(x_lims, y_lims)

# Connect two callbacks to your axis instance.
# These will call the function "on_change" whenever
# xlim or ylim is changed.
ax.callbacks.connect('xlim_changed', on_change)
ax.callbacks.connect('ylim_changed', on_change)

plt.show()

plt.hist(q21a["MPG"], bins = 20, edgecolor=  'black')
plt.show()

plt.boxplot(x= 'MPG', data =q21a)
plt.show()

import statsmodels.api as sm
sm.qqplot(q21a['MPG'])
plt.xlabel('MPG', color ='red')
plt.savefig('MPG of cars.png')
plt.show()
C:\Users\Moin Dalvi\anaconda3\lib\site-packages\statsmodels\graphics\gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
  ax.plot(x, y, fmt, **plot_style)

import scipy.stats as stats
stats.probplot(q21a['MPG'], dist="norm", plot=plt)
plt.xlabel('MPG', color ='red')
plt.savefig('MPG of cars.png')
plt.show()

sn.distplot(q21a['MPG'],kde=True, bins =10)
plt.show()
C:\Users\Moin Dalvi\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)

q21b = pd.read_csv("C:/Users/Moin Dalvi/Documents/EXcelR Study and Assignment Material/Data Science Assignments/Basic Statistics 1/wc-at.csv")
q21b
Waist	AT
0	74.75	25.72
1	72.60	25.89
2	81.80	42.60
3	83.95	42.80
4	74.65	29.84
...	...	...
104	100.10	124.00
105	93.30	62.20
106	101.80	133.00
107	107.90	208.00
108	108.50	208.00
109 rows × 2 columns

plt.hist(q21b['Waist'], edgecolor= 'red')
plt.show()

plt.boxplot(x = 'Waist', data= q21b)
plt.title("Waist")
plt.savefig('Waist.png')
plt.show()

sn.distplot(q21b['Waist'], 
             bins=10,
            kde = True
            )
plt.show()
C:\Users\Moin Dalvi\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)

import statsmodels.api as sm
sm.qqplot(q21b['Waist'])
plt.show()
C:\Users\Moin Dalvi\anaconda3\lib\site-packages\statsmodels\graphics\gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
  ax.plot(x, y, fmt, **plot_style)

stats.probplot(q21b['Waist'], dist = 'norm', plot = plt)
plt.xlabel('Waist', color= 'red')
plt.savefig('Waist.png')
plt.show()

sn.distplot(q21b['AT'], bins =10, kde=True)
plt.show()
C:\Users\Moin Dalvi\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)

import statsmodels.api as sm
sm.qqplot(q21b['AT'])
plt.show()
C:\Users\Moin Dalvi\anaconda3\lib\site-packages\statsmodels\graphics\gofplots.py:993: UserWarning: marker is redundantly defined by the 'marker' keyword argument and the fmt string "bo" (-> marker='o'). The keyword argument will take precedence.
  ax.plot(x, y, fmt, **plot_style)

stats.probplot(q21b['AT'], dist = 'norm', plot = plt)
plt.xlabel('AT', color= 'red')
plt.savefig('AT.png')
plt.show()

 