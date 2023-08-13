import pytest
import project2 as p2
import pandas as pd
from argparse import Namespace

def test_get_files():
    args = Namespace(N=5, ingredient=["ground beef", "Mexican cheese blend", "oil", "enchilada sauce"])
    actual = p2.get_file()
    data = {'id':[10259,25693,20130,22213,13162], 'cuisine': ['greek', 'southern_us', 'filipino', 'indian', 'indian'], 'ingredients': [['water', 'vegetable oil', 'wheat', 'salt'],['romaine lettuce', 'black olives', 'grape tomatoes'],['plain flour', 'ground pepper', 'salt', 'tomatoes'], ['light brown sugar', 'granulated sugar', 'butter'], ['boneless chicken skinless thigh', 'minced garlic']]}
    expected = pd.DataFrame(data)
    assert type(expected) == type(actual)