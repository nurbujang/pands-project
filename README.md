# pands-project
Pands project for Programming and Scripting

Author: Nur Bujang

## **analysis.py**

### Dataset Summary
Dataset iris.data (1) contains 5 columns and 150 lines (replicates). From that, it contains 3 classes of 50 instances each. A quick look through the data file indicates that the dataset consists of four measurement of a particular external flower structure to morphologically determine/identify whether it is one of three species: *Iris setosa, I. versicolor or I. virginica*. However, a closer look at the description file indicates that these measurements are the length (in cm) and width (in cm) of two structures: the sepal and petal. The color attribute is probably unreliable because of color polymorphism within the species or population. The dataset itself is complete and contains no missing and duplicating values.

Morphological analysis for species determination is used to assess biodiversity in an ecosystem. It could be used to identify new species or rectify previously misidentified species. While there are other more accurate methods for species identification such as DNA Barcoding and protein-based methods, morphological analysis is quick, cheap and particularly useful for researchers in the field. 

This data can be used to develop Interactive Identification Keys for future taxonomists and researchers as well as species determination using pattern recognition in Machine Learning. It can also be used in phylogenetic studies to identify which species are more closely related to each other and share a common ancestor. 

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
8. 

histogram
scatterplot
png
visualize the entire dataset using sns.pairplot(1) adjust figsize
boxplot
Pearson Correlation
k-Nearest Neighbor Classification
Logistic Regression
Decision Tree Classification
Support Vector Machine
Random Forest
Naive Bayes Classifier


### Conclusion:
A program that outputs a summary of each variable to a single text file, saves a histogram of each variable to png files, and a scatter plot of each pair of variables was written. Additional analyses done were Pearson Correlation, box plot, k-Nearest Neighbor, Logistic Regression, Decision Tree Classification, Support Vector Machine and Random Forest. Model accuracy score was displayed for each test.

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







 






