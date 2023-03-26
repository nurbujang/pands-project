# analysis.py
# Pands project for Programming and Scripting
# Author: Nur Bujang
# To write a program called analysis.py that:
# 1. Outputs a summary of each variable to a single text file,
# 2. Saves a histogram of each variable to png files,
# 3. Outputs a scatter plot of each pair of variables.
# 4. Perform any other analysis I think is appropriate

# import modules
import pands as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# 1. Summary





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
