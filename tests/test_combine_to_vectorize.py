import pytest
from argparse import Namespace
import json
import project2 as p2
import pandas as pd


@pytest.fixture
def get_data():
    args = Namespace(N=5, ingredient=["ground beef", "Mexican cheese blend", "oil", "enchilada sauce"])
    data = {'id':[10259,25693,20130,22213,13162], 'cuisine': ['greek', 'southern_us', 'filipino', 'indian', 'indian'], 'ingredients': [['water', 'vegetable oil', 'wheat', 'salt'],['romaine lettuce', 'black olives', 'grape tomatoes'],['plain flour', 'ground pepper', 'salt', 'tomatoes'], ['light brown sugar', 'granulated sugar', 'butter'], ['boneless chicken skinless thigh', 'minced garlic']]}
    df = pd.DataFrame(data)
    return args,df

def test_compute(get_data,capfd):
    args,df = get_data
    actual = p2.combine_to_vectorize(df,args)
    expected = ['water vegetable oil wheat salt', 'romaine lettuce black olives grape tomatoes', 'plain flour ground pepper salt tomatoes', 'light brown sugar granulated sugar butter', 'boneless chicken skinless thigh minced garlic', 'ground beef Mexican cheese blend oil enchilada sauce']
    expected == actual