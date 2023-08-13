## Gnaneswar Kolla

## Project Title

### CS5293,Spring 2022 Project 2

## Project Description

The project is to predict cuisine type from a list of ingredients passed from CLI along with predicted score and top n closest cuisines to the given ingredients.

## Installation/Getting Started

1. Pipenv to create and manage virtual environment for the project.     
    > pipenv install
2. [Packages](#packages) required to run this project are in Pipfile file which automatically installs during installation of pipenv in step 1.
3. Once, the packages are successfully installed, the project can be executed using 
    > pipenv run python project2.py --N 5 --ingredient "dijon mustard" --ingredient "eggs" --ingredient "fresh tarragon" --ingredient "tabasco Pepper Sauce" --ingredient "mayonaise" --ingredient "mustard seeds"
4. Pytests can be runnable using below command
    > pipenv run python -m pytest

## Packages

- `pandas` is a python library used to perform data analysis and manipulation.
    - In this project, pandas is used to create a data frame for ingredients data.
- `sklearn` is a python library used for classification, predictive analytics and various machine learning tasks.
    - `TfidfVectorizer` is a python library used to convert raw documents to feature vectors in a matrix.
        - In this project TfidfVectorizer is used to convert json ingredients data into feature values to predict.
    - `KNeighborsClassifier` is a classifier implementing the k-nearest neighbors vote.
        - In this project KNeighborsClassifier is used to predict cuisine based on ingredients and also tp predict score for that particular cuisine.
## Assumptions and Bugs

1. Assuming the ingredients passed in CLI are present in yummly.json file.
2. Assuming value of k in KNeighborsClassifier is value of N passed in CLI.
3. When special symbols are sent in input ingredients the code may give wrong scores.

##  Approach to Developing the code

1. `get_file(args)` 
    This function takes arguments passed in the CLI, reads data from the yummly.json file and creates a dataframe . 

2. `combine_to_vectorize(df,args)` This function takes dataframe and takes all values and appends it to a list along with input ingredients to form a corpus which at once can be used to vectorize instead of vectorizing ingredient data and input data seperately.

    TODO: Here in this method Ingredients can be normalized to get better scores. Ex:I tried lemmatizing the few ingredients and observed changes in score.  

3. `predict_closest(df,corpus,args)` This function predicts cuisine, cuisine score and top n closest cuisines from given data.
    - Firstly I used TfidfVectorizer to convert string to feature vector. I used n-grams as 1 because the similarity scores were better when compared to higher n-grams. 
    - Next I used KNeighborsClassifier to create and train the model. I used ingredient as X and labels from data frame as my Y and predicted the cusine by passing vectorized input list.
    - From KNeighborsClassifier I used predict_proba function which returns probability estimates (i.e the probability that a particular data point falls into the underlying classes) for the test data by passing vectorized input list .
    -  To find similar cuisines I used cosine_similarity method to get all the similarity scores between the input feature vector and ingredient feature vector.  Sorted the array of scores and took the top n values from them.
4. `json_output(pred,pred_cuisine_score,n_closest)` This function takes predicted cuisine,score and top n similar cuisines and formats into a pretty JSON to print the output.

## Tests

1. **`test_get_files.py`**
    | Function | Test Function | Description  |   
    |   --- |   --- |   ---
    |   `get_file(args)`    |    `test_get_files(capfd)`    |    checks whether data is read.
   
2. **`test_combine_to_vectorize.py`**
    | Function | Test Function | Description  |   
    |   --- |   --- |   ---
    |   `combine_to_vectorize(df,args)`    |    `test_compute(get_data,capfd)`    |    Tests whether the data is combined to vectorize.
    
3. **`test_predict_closest.py`**
    
    | Function | Test Function | Description  |   
    |   --- |   --- |   ---
    |   `predict_closest(df,corpus,args)`    |    `test_predict_closest(get_data,capfd)`    |    Test whether the cuisine ,score and top n values are generated.

4. **`test_json_output.py`**
    | Function | Test Function | Description  |   
    |   --- |   --- |   ---
    |   `json_output(pred,pred_cuisine_score,n_closest)`    |    `test_json_output(capfd)`    |    Tests whether the the string is of json.