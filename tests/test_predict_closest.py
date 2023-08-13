import pytest
from argparse import Namespace
import json
import project2 as p2
import pandas as pd

@pytest.fixture
def get_data():
    args = Namespace(N=5, ingredient=["ground beef", "Mexican cheese blend", "oil", "enchilada sauce"])
    data = {'id':[10259,25693,20130,22213,13162,29109,11462,2238,41882], 'cuisine': ['greek', 'southern_us', 'filipino', 'indian', 'indian','irish','italian','irish','chinese'], 'ingredients': [['water', 'vegetable oil', 'wheat', 'salt'],['romaine lettuce', 'black olives', 'grape tomatoes'],['plain flour', 'ground pepper', 'salt', 'tomatoes'], ['light brown sugar', 'granulated sugar', 'butter'], ['boneless chicken skinless thigh', 'minced garlic'], ['light brown sugar', 'granulated sugar', 'butter'],['KRAFT Zesty Italian Dressing', 'purple onion'], ['eggs', 'citrus fruit', 'raisins', 'sourdough'],['boneless chicken skinless thigh', 'minced garlic']]}
    df = pd.DataFrame(data)
    corpus = ['water vegetable oil wheat salt', 'romaine lettuce black olives grape tomatoes', 'plain flour ground pepper salt tomatoes', 'light brown sugar granulated sugar butter', 'boneless chicken skinless thigh minced garlic', 'light brown sugar granulated sugar butter', 'KRAFT Zesty Italian Dressing purple onion', 'eggs citrus fruit raisins sourdough', 'boneless chicken skinless thigh minced garlic', 'ground beef Mexican cheese blend oil enchilada sauce']
    return df,corpus,args

def test_predict_closest(get_data):
    df,corpus,args = get_data
    pred,pred_cuisine_score,n_closest = p2.predict_closest(df,corpus,args)
