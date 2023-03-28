# pands-project
Pands project for Programming and Scripting

### Dataset Summary
Dataset iris.data (1) contains 5 columns and 150 lines (replicates). A quick look through the data file indicates that it is four measurement of a particular external flower structure to morphologically determine/identify whether it is one of three species: *Iris setosa, I. versicolor or I. virginica*. However, a closer look at the description file indicates that these measurements are the length (in cm) and width (in cm) of two structures: the sepal and petal. The color attribute is probably unreliable because of color polymorphism within the species or population. The dataset itself is complete and contains mo missing values.

Morphological analysis for species determination is used to assess biodiversity in an ecosystem. It could be used to identify new species or rectify previously misidentified species. While there are other more accurate methods for species identification such as DNA Barcoding and protein-based methods, morphological analysis is quick, cheap and particularly useful for researchers in the field. 

This data can be used to develop Interactive Identification Keys for future taxonomists and researchers and species determination based on pattern recognition in Machine Learning. It can also be used to which species are more closely related to each other and share a common ancestor. 

## **analysis.py**

### Task Description:
*The task is to write a program called analysis.py that:*

*1. Outputs a summary of each variable to a single text file,*
	
*2. Saves a histogram of each variable to png files,*
	
*3. Outputs a scatter plot of each pair of variables.*
	
*4. Performs any other analysis I think is appropriate*

### Method:
1. The general processes in data analysis are data loading, dataset analyzing and visualization, model training, model evaluation, model testing (1). The iris.data file was saved from the UCI website (2) and uploaded into pands-project folder.
2. First, I imported the necessary packages/modules for the project (3-6).
3. Then, I defined the column names, loaded the data () and added the column header to the dataframe ().
4. checked for missing value using the isna() function that returns a True/false if a value is missing ().
5. print out df.describe into csv ()


### Conclusion:


### References:
1. https://data-flair.training/blogs/iris-flower-classification/
2. http://archive.ics.uci.edu/ml/datasets/Iris
3. https://www.w3schools.com/python/pandas/default.asp
4. https://www.w3schools.com/python/numpy/numpy_intro.asp
5. https://www.w3schools.com/python/matplotlib_pyplot.asp
6. https://www.w3schools.com/python/numpy/numpy_random_seaborn.asp
https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
https://www.w3schools.com/python/pandas/pandas_csv.asp
https://towardsdatascience.com/how-to-read-csv-file-using-pandas-ab1f5e7e7b58
https://stackoverflow.com/questions/34091877/how-to-add-header-row-to-a-pandas-dataframe
https://www.geeksforgeeks.org/how-to-add-header-row-to-a-pandas-dataframe/
https://practicaldatascience.co.uk/data-science/https://practicaldatascience.co.uk/data-science/how-to-use-isna-to-check-for-missing-values-in-pandas-dataframes
https://datatofish.com/count-nan-pandas-dataframe/
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
https://towardsdatascience.com/handling-missing-values-with-pandas-b876bf6f008f
https://www.w3schools.com/python/pandas/ref_df_describe.asp
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
https://pythonexamples.org/python-write-string-to-text-file/
https://www.tutorialkart.com/python/python-write-string-to-text-file/
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_string.html
https://www.geeksforgeeks.org/overlapping-histograms-with-matplotlib-in-python/
https://datavizpyr.com/overlapping-histograms-with-matplotlib-in-python/
https://softwarenotebook.com/2022/03/07/overlapping-histograms-with-matplotlib-library-in-python/
https://medium.com/@ooemma83/how-to-construct-cool-multiple-histogram-plots-using-seaborn-and-matplotlib-in-python-6c6c7ba6c10b
https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
https://seaborn.pydata.org/tutorial/distributions.html
https://www.statology.org/seaborn-title/
https://towardsdatascience.com/histograms-with-pythons-matplotlib-b8b768da9305


 





