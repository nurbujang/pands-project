# pands-project
Pands project for Programming and Scripting

Author: Nur Bujang

## **analysis.py**

### Dataset Summary
Dataset iris.data (1) contains 5 columns and 150 lines (replicates). From that, it contains 3 classes of 50 instances each. A quick look through the data file indicates that the dataset consists of four measurement of a particular external flower structure to morphologically determine/identify whether it is one of three species: *Iris setosa, I. versicolor or I. virginica*. However, a closer look at the description file indicates that these measurements are the length (in cm) and width (in cm) of two structures: the sepal and petal. The color attribute is probably unreliable because of color polymorphism within the species or population. The dataset itself is complete and contains no missing and duplicating values.

Morphological analysis for species determination is used to assess biodiversity in an ecosystem. It could be used to identify new species or rectify previous species misidentification. While there are other more accurate methods for species identification such as DNA Barcoding and protein-based methods, morphological analysis is quick, cheap and particularly useful for researchers in the field. 

This data can be used to develop Interactive Identification Keys for future taxonomists and researchers as well as species determination using pattern recognition in Machine Learning. It can also be used alongside phylogenetic studies to identify which species are more closely related to each other and share a common ancestor. 

### Task Description:
*The task is to write a program called analysis.py that:*

*1. Outputs a SUMMARY OF EACH VARIABLE to a single text file,*
	
*2. Saves a HISTOGRAM OF EACH VARIABLE to png files,*
	
*3. Outputs a SCATTER PLOT OF EACH PAIR OF VARIABLES.*
	
*4. Performs any other analysis that I think is appropriate*

### Method:
1. The general processes in data analysis are data loading, dataset analyzing and visualization, model training, model evaluation, model testing (1). The iris.data file was saved from the UCI website (2) and uploaded into pands-project folder.
2. First, I imported the necessary packages/modules for the project as shown in W3Schools (3-6) and scikit-learn (7).
3. Then, I created a list of column names, read the data (8-10) and added the column header to the dataframe (11-13).
4. I looked at the first 2 lines (14) to see if the column names were added correctly.
5. A quick lookover the dataset was done to see the replicates on each species (15) and other basic information about the dataset (16)
6. I also checked for missing value using the isnull and isna function that returns a True/false if a value is missing (17-21) and duplicates (22-23).
7. For Question 1, I used df.describe to get a summary of each variable (24-25). Then I used a built-in open function, converted pandas dataframe to string (26-28) and exported/wrote the converted string into a text file (29-30).
8. For Question 2, I set the background using seaborn, set the grid position of the subplots, and plotted the histogram (31-32). I also added the  kernel density estimation (KDE) according to (33) and added the supertitle (34-35). Tight layout automatically adjusts the subplots to fit nicely into the figure area (36). Then, I saved the histogram into iris.png file using plt.savefig according to (37).
9. For Question 3, I used seaborn pairplot according to (38-41) based on the hue (42). I customized the color palette according to (43-44). I also added regression lines (45-47) and adjusted the figure size (48).
10. I performed 10 other analysis for Question 4, consisting of more data visualization techniques and basic Machine Learning analysis. 
11. The first Pearson Correlation analysis was done on all the 4 variables on all three species grouped together using corr with Pearson Correlation as default (49-53). I eliminated the upper triangle for better readability (54). np.ones was used to create a boolean matrix with the same size as the correlation matrix (55) and astype(bool) converts the upper triangle values to False, while the lower triangle will have the True values (56-57). Then, I used pandas corr.where to return the same-sized dataframe, but False is converted to NaN on the upper triangle and finally, I printed the result in the terminal output (56-57). For better visualization, I also a created a correlation matrix using seaborn heatmap according to (58-59), customized the color bar (60) and modified the ticklabels (61-65). 
12. The second Pearson Correlation analysis was done by species group, to better understand which attributes are highly correlated within each species. This was done using df.groupby class with Iris species (66-68). The results were printed in the terminal. I found that this method offers more information than doing correlation on attributes for all three species grouped together.
13. For another data visualization, I created a Seaborn Box Plot (69-71) and added a Seaborn Jitter Plot over it (72-74). The grid arrangement for the subplots followed that of (75-76). The super title was added as usual and plt.show displayed the plot.
14. I also created a Seaborn Violin Plot for each variable in a 1x4 grid (77-80) as another data visualization method. 
15. In preparation for basic machine learning analysis, split the data into training (80%) and testing (20%) to detect overfitting, and later, the testing dataset will be used to check the accuracy of the model (81-85). While most ML examples available uses only 2 variables (sepal length and width only or petal length and width only) to simplify analysis, I decided to use all 4 variables because the best determinants are still unknown at this point (whether sepal or petal attribute is better than the other).
16. For k-Nearest Neighbor Classifier, I performed the analysis in a range of 7 to 10 (86-90). I kept the number small to prevent overfitting because there are only 3 species (91-101). According to the accuracy score results, k=7 is enough to get a very high score on the model. I printed out the accuracy score using the format as shown in (102-104).
17. For Logistic Regression, 
18. For Decision Tree Classification
19. For Support Vector Machine
20. For Random Forest
21. The last analysis done was Naïve Bayes Classifier using Gaussian Naïve Bayes

### Conclusion:
A program that outputs a summary of each variable to a single text file, saves a histogram of each variable to png files, and a scatter plot of each pair of variables was written. Additional analyses done were two types of Pearson Correlations, Box plot, Violin Plot, k-Nearest Neighbor Classifier, Logistic Regression, Decision Tree Classification, Support Vector Machine Classifier, Random Forest and Naïve Bayes Classifier. 

### References:
1. https://data-flair.training/blogs/iris-flower-classification/
2. http://archive.ics.uci.edu/ml/datasets/Iris
3. https://www.w3schools.com/python/pandas/default.asp
4. https://www.w3schools.com/python/numpy/numpy_intro.asp
5. https://www.w3schools.com/python/matplotlib_pyplot.asp
6. https://www.w3schools.com/python/numpy/numpy_random_seaborn.asp
7. https://scikit-learn.org/stable/modules/classes.html
8. https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
9. https://www.w3schools.com/python/pandas/pandas_csv.asp
10. https://towardsdatascience.com/how-to-read-csv-file-using-pandas-ab1f5e7e7b58
11. https://stackoverflow.com/questions/34091877/how-to-add-header-row-to-a-pandas-dataframe
12. https://www.geeksforgeeks.org/how-to-add-header-row-to-a-pandas-dataframe/
13. https://www.angela1c.com/projects/iris_project/downloading-iris/
14. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
15. https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
16. https://www.w3schools.com/python/pandas/ref_df_info.asp
17. https://note.nkmk.me/en/python-pandas-nan-judge-count/
18. https://practicaldatascience.co.uk/data-science/how-to-use-isna-to-check-for-missing-values-in-pandas-dataframes
19. https://datatofish.com/count-nan-pandas-dataframe/
20. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
21. https://towardsdatascience.com/handling-missing-values-with-pandas-b876bf6f008f
22. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
23. https://notebook.community/mnnit-workspace/Logical-Rhythm-17/Class-4/Introduction%20to%20Pandas%20and%20Exploring%20Iris%20Dataset
24. https://www.w3schools.com/python/pandas/ref_df_describe.asp
25. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
26. https://pythonexamples.org/python-write-string-to-text-file/
27. https://www.tutorialkart.com/python/python-write-string-to-text-file/
28. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_string.html
29. https://phoenixnap.com/kb/file-handling-in-python
30. https://www.freecodecamp.org/news/file-handling-in-python/
31. https://medium.com/@ooemma83/how-to-construct-cool-multiple-histogram-plots-using-seaborn-and-matplotlib-in-python-6c6c7ba6c10b
32. https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
33. https://seaborn.pydata.org/tutorial/distributions.html
34. https://stackoverflow.com/questions/68948981/create-a-subplot-of-multiple-histograms-with-titles
35. https://www.statology.org/seaborn-title/
36. https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html
37. https://towardsdatascience.com/histograms-with-pythons-matplotlib-b8b768da9305
38. https://pro.arcgis.com/en/pro-app/latest/help/analysis/geoprocessing/charts/scatter-plot-matrix.htm
39. https://www.statology.org/pairs-plot-in-python/
40. https://vitalflux.com/what-when-how-scatterplot-matrix-pairplot-python/
41. https://plotly.com/python/splom/
42. https://www.statology.org/seaborn-pairplot-hue/
43. https://seaborn.pydata.org/tutorial/color_palettes.html
44. https://www.codecademy.com/article/seaborn-design-ii
45. https://towardsdatascience.com/seaborn-pairplot-enhance-your-data-understanding-with-a-single-plot-bf2f44524b22
46. https://python-charts.com/correlation/scatter-plot-regression-line-seaborn/
47. https://waynestalk.com/en/python-regression-plot-en/
48. https://pythonguides.com/matplotlib-subplots_adjust/
49. https://www.kaggle.com/code/danalexandru/simple-analysis-of-iris-dataset/notebook
50. https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/
51. https://www.hackersrealm.net/post/iris-dataset-analysis-using-python
52. https://www.kaggle.com/code/dronio/iris-plots-correlation-matrix
53. https://www.kaggle.com/code/sureshmecad/iris-flower-classification
54. https://numpy.org/doc/stable/reference/generated/numpy.tril.html
55. https://www.digitalocean.com/community/tutorials/numpy-ones-in-python
56. https://stackoverflow.com/questions/34417685/melt-the-upper-triangular-matrix-of-a-pandas-dataframe
57. https://cmdlinetips.com/2020/02/lower-triangle-correlation-heatmap-python/
58. https://www.educba.com/seaborn-heatmap-size/
59. https://linuxhint.com/seaborn-heatmap-size/
60. https://www.geeksforgeeks.org/how-to-change-the-colorbar-size-of-a-seaborn-heatmap-figure-in-python/
61. https://www.kaggle.com/code/sejalkshirsagar/customize-seaborn-heatmaps
62. https://www.tutorialspoint.com/rotate-xtick-labels-in-seaborn-boxplot-using-matplotlib
63. https://stackoverflow.com/questions/44954123/rotate-xtick-labels-in-seaborn-boxplot
64. https://stackoverflow.com/questions/27037241/changing-the-rotation-of-tick-labels-in-seaborn-heatmap
65. https://copyprogramming.com/howto/rotate-axis-tick-labels-of-seaborn-plots
66. https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/correlation-pearson-kendall-spearman/
67. https://www.angela1c.com/projects/iris_project/iris_notebook/
68. https://community.rstudio.com/t/how-to-apply-corrr-correlate-by-group/6236
69. https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/amp/
70. https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/
71. https://www.nickmccullum.com/python-visualization/boxplot/
72. https://datagy.io/seaborn-stripplot/
73. https://seaborn.pydata.org/generated/seaborn.stripplot.html
74. https://www.geeksforgeeks.org/stripplot-using-seaborn-in-python/
75. https://stackoverflow.com/questions/3584805/what-does-the-argument-mean-in-fig-add-subplot111
76. https://www.kaggle.com/code/kstaud85/iris-data-visualization
77. https://deepnote.com/@econdesousa/ViolinPlotvsBoxPlot-aadf0c53-53b4-4221-89b9-4388c54c68bd
78. https://www.nickmccullum.com/python-visualization/subplots/
79. https://matplotlib.org/stable/gallery/statistics/customized_violin.html
80. https://seaborn.pydata.org/generated/seaborn.violinplot.html
81. https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-k-nearest-neighbors-algorithm-exercise-1.php
82. https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-k-nearest-neighbors-algorithm-exercise-2.php
83. https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-k-nearest-neighbors-algorithm-exercise-3.php
84. https://stackoverflow.com/questions/37512079/python-pandas-why-does-df-iloc-1-values-for-my-training-data-select-till
85. https://www.shanelynn.ie/pandas-iloc-loc-select-rows-and-columns-dataframe/
86. https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-k-nearest-neighbors-algorithm-exercise-5.php
87. https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-k-nearest-neighbors-algorithm-exercise-6.php
88. https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-k-nearest-neighbors-algorithm-exercise-7.php
89. https://deepchecks.com/how-to-check-the-accuracy-of-your-machine-learning-model/
90. https://numpy.org/doc/stable/reference/generated/numpy.arange.html
91. https://numpy.org/doc/stable/reference/generated/numpy.arange.html
92. https://thatascience.com/learn-machine-learning/iris-dataset/
93. https://thatascience.com/learn-machine-learning/build-knearestneighbors/
94. https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
95. https://scikit-learn.org/stable/modules/neighbors.html
96. https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
97. https://vitalflux.com/k-nearest-neighbors-explained-with-python-examples/
98. https://rstudio-pubs-static.s3.amazonaws.com/369869_fe1a8a1a1b1c4145b5b6f22b96df8345.html
99. https://www.hackersrealm.net/post/iris-dataset-analysis-using-python
100. https://deepnote.com/@ndungu/Implementing-KNN-Algorithm-on-the-Iris-Dataset-e7c16493-500c-4248-be54-9389de603f16
101. https://www.datacamp.com/tutorial/introduction-machine-learning-python
102. https://www.edureka.co/community/162168/what-the-difference-between-and-in-python-string-formatting
103. https://stackoverflow.com/questions/4288973/whats-the-difference-between-s-and-d-in-python-string-formatting
104. https://www.geeksforgeeks.org/difference-between-s-and-d-in-python-string/






