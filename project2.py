import argparse
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def get_file():
  '''
    To open the JSON file, read data and pass to Dataframe.

    Parameter
    ----------
    args : Namespace
        The parameter contains list of passed arguments in CLI.

    Returns
    -------
        dataframe object of yummly.json file
    '''
  with open('docs/yummly.json', 'r') as file:
      data = json.load(file)
  df = pd.DataFrame(data)
  return df

def combine_to_vectorize(df,args):
  '''
    To combine corpus with datset and input ingredients to vectorize them at once

    Parameter
    ----------
    df : DataFrame
        Json data in dataframe.

    args : Namespace
        The parameter contains list of passed arguments in CLI.

    Returns
    -------
        list of dataset ingredients and input ingredients
    '''
  input_ingredients = args.ingredient
  ingredients_list = df['ingredients']
  corpus = []
  for ing in ingredients_list:
    corpus.append(' '.join(ing))
  input_string = ' '.join(input_ingredients)
  corpus.append(input_string)
  return corpus

def predict_closest(df,corpus,args):
  '''
    To predict cuisine, cuisine score and top n closest cuisines. 

    Parameter
    ----------
    df : DataFrame
        Json data in dataframe.

    corpus : list
        list of dataset and input ingredients.

    args : Namespace
        The parameter contains list of passed arguments in CLI.

    Returns
    -------
        predicted cuisine, predicted cuisine score and top n closest cuisines
    '''
  n = args.N
  labels = df['cuisine']
  ids = list(df['id'])
  n_closest = []
  
  #Vectorize the data
  tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
  ingredient_features = tfidf_vectorizer.fit_transform(corpus)
  ingredients = ingredient_features[:-1]
  input = ingredient_features[-1]
  
  #create, train and predict
  X_train, X_test, y_train, y_test = train_test_split(ingredients, labels, test_size = 0.2, random_state = 100)
  classifier = KNeighborsClassifier(n_neighbors=args.N)
  classifier.fit(X_train,y_train)
  pred = classifier.predict(input)
  
  #score of predicted cuisine 
  predict_proba_input =classifier.predict_proba(input)[0]
  pred_cuisine_score = np.amax(predict_proba_input)
  
  #n closest cuisine
  scores = cosine_similarity(input,ingredients)
  sort_scores = scores[0].argsort()[::-1]
  
  #n values
  n_values = sort_scores[:n]
  closest_n_data = [(index,scores[0][index]) for index in n_values]
  # for index,score in closest_n_data:
  #   if pred[0] == cuisine[index]:
  #     pred_cuisine_score = score
  #     break
  for index,score in closest_n_data:
    n_closest.append((ids[index],score))
  return pred,pred_cuisine_score,n_closest

def json_output(pred,pred_cuisine_score,n_closest):
  '''
    To convert output into JSON format.

    Parameter
    ----------
    pred : list
        list contains predicted cuisine

    pred_cuisine_score : float
        predicted cuisine score
        
    n_closest : list
        list of tuples which contains id and scores of top n closest cuisines.

    Returns
    -------
        prints output in Json format.
    '''
  output = {}
  n_closest_list = []
  for tup in n_closest:
    temp = {}
    temp = {
        "id" : tup[0],
        "score" : round(tup[1],2)
    }
    n_closest_list.append(temp)
  output = {
    "cuisine" : pred[0],
    "score" : round(pred_cuisine_score,2),
    "closest" : n_closest_list
  }
  obj = json.loads(json.dumps(output))  
  json_formatted_str = json.dumps(obj, indent=4)
  print(json_formatted_str)

def main(parser):
  """
    Command line parsing and cuisine prediction
    Parameter
    ---------
    parser : Argumentparser
    """
  args=parser.parse_args()
  df = get_file()
  corpus = combine_to_vectorize(df,args)
  pred,pred_cuisine_score,n_closest = predict_closest(df,corpus,args)
  json_output(pred,pred_cuisine_score,n_closest)

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--N",type=int, required=True,help="It takes an integer number")
    parser.add_argument("--ingredient",type=str,action='append',required=True,help="It helps to get ingredient")
    main(parser)
