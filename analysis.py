# analysis.py
# Pands project for Programming and Scripting
# Author: Nur Bujang
# To write a program called analysis.py that:
# 1. Outputs a summary of each variable to a single text file,
# 2. Saves a histogram of each variable to png files,
# 3. Outputs a scatter plot of each pair of variables.
# 4. Perform any other analysis I think is appropriate

# General Process: Load data, Analyze/visualize dataset, Model training, Model Evaluation, Model Testing, 

# 1. Summary
# LOADING THE DATA

# import modules
import pandas as pd # for data loading from other sources and processing
import numpy as np # for computational operations
import matplotlib.pyplot as plt # for data visualization
import matplotlib.patches as mpatches # for data visualization
import matplotlib.lines as mlines # for data visualization
import seaborn as sns # for data visualization

# add column header




# 2. Histogram





# create normal distribution curve, mean=loc, standard deviation=scale
normData=np.random.normal(loc = 5, scale = 2, size = 1000)
plt.hist(normData, color='royalblue', edgecolor='#6495ED', linewidth=1)
# add grid on y-axis
plt.grid(axis = 'y', color = '#8FBC8F', linestyle = '--', linewidth = 0.3)


# 3. Scatter plot



# show plot
plt.show()


# 4. Other analysis
