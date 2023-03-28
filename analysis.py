# analysis.py
# Pands project for Programming and Scripting
# Author: Nur Bujang
# To write a program called analysis.py that:
# 1. Outputs a summary of each variable to a single text file,
# 2. Saves a histogram of each variable to png files,
# 3. Outputs a scatter plot of each pair of variables.
# 4. Perform any other analysis I think is appropriate

# General Process: Load data, Analyze/visualize dataset, Model training, Model Evaluation, Model Testing, 

# save iris.data file from UCI website and upload into pands-project folder

# import modules
import pandas as pd # for data loading from other sources and processing
import numpy as np # for computational operations
import matplotlib.pyplot as plt # for data visualization
import matplotlib.patches as mpatches # for data visualization
import matplotlib.lines as mlines # for data visualization
import seaborn as sns # for data visualization
import matplotlib.patches as mpatches # to customize legend
import matplotlib.lines as mlines # to customize legend

# Load data and add column header
columns = ['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)', 'Iris species'] # define column headers
df=pd.read_csv('iris.data', names=columns) # read the csv file and assign each column name
print(df.head(5)) # print out first 5 lines

# check for missing values
df.info() # returns RangeIndex: 150 entries, 0 to 149, so no missing values
print(df.info()) # print out data info
df.isna().sum() # get number of missing values in each column
print(df.isna().sum()) # print out number of missing values

# 1. Summary into text file, containing basic statistical analysis
df.describe() # to get basic statistical analysis data
print (df.describe()) # print out description of data
text_file = open("summary.txt", "wt") # to write the string into a text file
n = text_file.write(df.describe().to_string()) # export into text file called summary, convert pandas dataframe to string
text_file.close() # always close file

# 2. Histogram into png file - Visualizing 4 histograms of each column is not very informative, so overlapping histogram is chosen
sns.set(style="whitegrid")  
fig,axs = plt.subplots(2,2, figsize = (8,10))
sns.histplot(data=df, x="Sepal length (cm)", kde=True, color=["fuschia"], ax=axs[0, 0])
sns.histplot(data=df, x="Sepal width (cm)", kde=True, color=["blueviolet"], ax=axs[0, 1])
sns.histplot(data=df, x="Petal length (cm)", kde=True, color=["chartreuse"], ax=axs[1, 0])
sns.histplot(data=df, x="Petal width (cm)", kde=True, color=["forestgreen"], ax=axs[1, 1])
fig.tight_layout()

# fig = plt.figure(figsize = (5,6))
# ax = fig.gca()
# df.hist(ax=ax, color = "skyblue", edgecolor = "gold")
# plt.show()
# fig = plt.figure(figsize=(10,6)) # 20 width. 15 height
# plt.title('Petal and sepal dimensions of three Iris species', color ='#191970', fontweight='bold') # customize plot title
# plt.xlabel('Measurement (cm)', color ='#00008B', style='oblique', fontweight='bold') # x-axis
# # to create a little space between y label and y-axis
# plt.ylabel('Frequency', color ='#00008B', style='oblique', fontweight='bold', labelpad=12) # y-axis
# plt.hist(df['Sepal length (cm)'], alpha=0.5, label='Sepal length (cm)', color='olive', edgecolor='gainsboro') # alpha is transparency parameter to visualize overlapping histograms better
# plt.hist(df['Sepal width (cm)'], alpha=0.5, label='Sepal width (cm)', color='lime', edgecolor='gainsboro')
# plt.hist(df['Petal length (cm)'], alpha=0.5, label='Petal length (cm)', color='red', edgecolor='gainsboro')
# plt.hist(df['Petal width (cm)'], alpha=0.5, label='Petal width (cm)', color='fuchsia', edgecolor='gainsboro')
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig('Iris.png')










# # add grid on y-axis
# plt.grid(axis = 'y', color = '#8FBC8F', linestyle = '--', linewidth = 0.3)
# # show plot
# plt.show()




# # 3. Scatter plot



# # show plot
# plt.show()

# 4. Other analysis
