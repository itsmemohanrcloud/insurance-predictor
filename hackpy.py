import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


percentages = []
df = pd.read_csv('dataset\\insurance3r2.csv')
df.head()
numerical_columns = [col for col in df.columns if (df[col].dtype=='int64' or df[col].dtype=='float64') and col != 'Exited']

df[numerical_columns].describe().loc[['min','max', 'mean','50%'],:]

#Percentage of gender to Claim Insurance
#------------------------------------------------------------------------------------------------------
percentages = []
for gen in list(df["sex"].unique()):
    
    p = round((df["insuranceclaim"][df["sex"]==gen].value_counts()[1]/df["insuranceclaim"][df["sex"]==gen].value_counts().sum())*100 , 2)
    
    percentages.append(p)
    print(gen,"(% to claim) : ", p)

plt.bar(0, percentages[0])
plt.bar(1, percentages[1])
plt.xticks((0,1), ('Female','Male'))
plt.xlabel("sex")
plt.ylabel("Percentage")
plt.title("Percentage of gender to Claim Insurance")
plt.show()
#------------------------------------------------------------------------------------------------------

#Percentage of gender who Exited
#------------------------------------------------------------------------------------------------------

percentages = []
for gen in list(df["sex"].unique()):
    
    p = round((df["Exited"][df["sex"]==gen].value_counts()[1]/df["Exited"][df["sex"]==gen].value_counts().sum())*100 , 2)
    
    percentages.append(p)
    print(gen,"(% to Exit) : ", p)

plt.bar(0, percentages[0])
plt.bar(1, percentages[1])
plt.xticks((0,1), ('Female','Male'))
plt.xlabel("sex")
plt.ylabel("Percentage")
plt.title("Percentage of gender who Exited")
plt.show()
#---------------------------------------------------------------------------------------------------

#Percentage of gender to Smoker
#-----------------------------------------------------------------------------------------------------
percentages = []
for gen in list(df["sex"].unique()):
    
    p = round((df["smoker"][df["sex"]==gen].value_counts()[1]/df["smoker"][df["sex"]==gen].value_counts().sum())*100 , 2)
    
    percentages.append(p)
    print(gen,"(% to claim) : ", p)

plt.bar(0, percentages[0])
plt.bar(1, percentages[1])
plt.xticks((0,1), ('Female','Male'))
plt.xlabel("sex")
plt.ylabel("Percentage")
plt.title("Percentage of gender to Smoker")
plt.show()
#-------------------------------------------------------------------------------------------------

#People who Exited (Exited = 1)
#-------------------------------------------------------------------------------------------------
plt.scatter(x=range(len(list(df["age"][df["Exited"]==1]))),y=df["age"][df["Exited"]==1],s=1)
plt.ylabel("Age")
plt.xlabel("People (rows)")
plt.title("People who Exited (Exited = 1)")
plt.show()
#------------------------------------------------------------------------------------------------
