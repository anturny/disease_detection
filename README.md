# Disease Prediction through Machine Learning 
## Executive Summary 
  This project utilizes machine learning classification techniques to predict diseases based on symptom data. In this project, disease prediction is treated as a classification problem in which each patient record is represented by a set of symptom features and is assigned to a specific disease category. The 3 models that were tested: support vector machines (SVM), deep neural networks (DNN), and shallow neural networks (SNN).
  This repository contains source code for data preprossing, model training, and evaluation alongside any figures and documentation of results. The project was conducted utilizing the same dataset and evaluation metrics to ensure fair comparison across the 3 models. 

  This GitHub repository was made in collaboration by [Yarah Al-Fouleh](https://github.com/yarah-yma1), [Senem Keceli](https://github.com/senem584), and [Anthony Nguyen](https://github.com/anturny).

  ## Table Of Contents
- [Introduction](#introduction)
- [Exploration and Quantification](#exploration-and-quantification)
- [Methodology](#methodology)
- [Experimental Setup and Execution](#experimental-setup-and-execution)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction 
Disease prediction in healthcare can be useful in terms of providing accurate and early diagnosis based on patient symptoms. By analyzing the patient symptoms the models can identify the patterns and classify them into specific diseases. 
In this project, disease prediction is treated as a classification problem in which each patient record is represented by a set of symptom features and is assigned to a specific disease category [2].
This dataset is especially important because it shows how through the utilization of ML models they can be applied to real world applications like healthcare for classification tasks with structured data. 
Thesis: Machine learning models can classify different diseases based on symptoms, however, model performance is impacted by dataset structure and appropriate model selection. 

## Exploration and Quantification 
  The dataset that we will be utilizing is structured into a tabular type of data with the symptoms labeled through binary features (0 and 1) to classify the presence or absence of a specific symptom. The disease names in the dataset are string values in which they were converted utilizing a label encoder in Python code before model training and testing [1].  

## Methodology
The machine learning models utilized for this project were selected based on their performance across different dataset types that were experimented on in previous projects. The 3 models that were utilized include: support vector machines (SVM), deep neural networks (DNN), and shallow neural networks (SNN).

#### Support Vector Machines (SVM)
Supervised learning models that find the optimal boundary to separate different groups of data (2D & 3D), maximizing the margin between datapoints of different classes. They are commonly used for classification and regression tasks, especially when the data has clear margins or when a decision boundary needs to be found in high dimensional or complex feature spaces. 

#### Deep Neural Networks 
Complex machine learning models with multiple layers that automatically learn hierarchical features from data, enabling them to perform tasks like image recognition, natural language processing, and speech recognition. They are used when dealing with large, complex datasets that require capturing intricate patterns and relationships beyond the capabilities of shallow models. 

#### Shallow Neural Networks
Simple machine learning models with one or a few layers of neurons that learn to map input data to outputs by adjusting weights through training. They are typically used for basic pattern recognition tasks, small-scale problems, or when interpretability and quick training are more important than capturing complex data structures. 

#### Evaluation 
The model performance was evaluated utilizing the following quantitative metrics: accuracy, precision, recall, and f1-score. These metrics will overall be able to assess if the model is correctly classifying the symptoms into their according diseases. 

## Experimental Setup and Execution

- To run this code, you will need to have a Python environment installed on your computer. It is recommended to use Visual Studio Code as this Python script was written and ran in VSCode. GitBash is also recommended in order to synchronize your VSCode with GitHub.
- In GitHub, click on the green icon labeled "<> CODE" on the top of this page and copy the HTTPS link.
- In VSCode, click on "Clone Git Repository" and paste the copied link from GitHub.
- In the search bar, type in "Python: Create Environment" and then select a preferred environment. This code used .venv as the virtual environment.
- When the virtual environment is open (appears as .venv in the list of items in the left menu), you may navigate to the relevant file in the SRC folder and select it. At this point, you may open your terminal and install the pip requirements for the necessary libraries in order to execute the code. Then, you may hit "Run" on the top right hand corner to execute the code.

All the models were experimentally setup in the same way for overall consistency for accurate results and comparisons. The steps are as follows:
1. Load in the dataset
2. Preprocess the data including label encoding and feature preparation
3. Train and test the data
4. Evaluate utilizing classification reports

All implementations of experimental setup and execution were written and executed through VSCode.

## Results 
The accuracy of the SVM model was reported as 41.50%. The classification report of the model showed that the diseases received moderate scoring in terms of precision, recall, and f1, while other diseases had low scores. 

![alt_text](/media/diseaseSVMClassificationReport.PNG)

The accuracy of the DNN model was reported as 34.00%. The classification report of the model had similar performance to the SVM model in which some diseases received moderate scores, and others received low scores. 

![alt_text](/media/diseaseDNNTestDataReport.PNG)

The accuracy of the SNN model was reported as 41.75% The classification report was similar to the SVM and DNN models. The SNN model had the highest overall accuracy and performance for disease prediction with this specific dataset. 

![alt_text](/media/diseaseSNNTestDataReport.PNG)

## Discussion
We were able to utilize the top classifying models from our previous projects. This included utilizing an SVM, DNN, and  SNN for the disease classifications. The SNN  classification was reported to have the highest accuracy, which implies that this type of model may be best suited for datasets in which complex and non-linear relationships happen between features. This makes sense because neural networks are designed to learn subtle interactions between input features. Both of the other models showed results that were lower than the SNN, although the SVM showed an accuracy very close to the SNN. SVM works by finding the optimal hyperplane that maximizes the margin between classes, so it makes sense that it would provide similar accuracy. The reason in which DNN performed lower is due to attempts to find higher levels of features in which this dataset does not have to due to its simple nature.

## Conclusion
We noticed that the overall disease prediction accuracy is not as high as the accuracy from our previous projects. Our explanation for this matter is that the disease dataset, itself, is more complex. It has more features that are not as distinguishable. Each feature within the disease prediction only has a binary input (0 or 1) that corresponds with “yes” or “no” inputs within each feature overlap for different labels. Two different labels may have the exact same feature values for multiple features, which would decrease the variability for some labels. This is because some symptoms are the same across multiple different diseases. Because of the fact that these models create patterns, when features between different labels become too similar, it can be difficult to create meaningful patterns that accurately predict diseases with high confidence. To obtain more accurate results utilizing these models, the dataset labels would require more distinguishable features.

## References
- [1] V. G. Kadamba (venugopalkadamba), “Disease prediction using machine learning,” GeeksforGeeks, https://www.geeksforgeeks.org/machine-learning/disease-prediction-using-machine-learning/. 
- [2] E. Bonat, “From data to diagnostics: Advanced GENAI Solutions for Disease Prediction,” Medium, https://ernest-bonat.medium.com/from-data-to-diagnostics-advanced-genai-solutions-for-disease-prediction-5aefa726f499. 

