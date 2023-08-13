import pytest
import project2 as p2


def test_json_output(capfd):
    pred = ['Indian']
    pred_cuisine_score = 0.2
    n_closest = [(1234,0.58),(4567,0.53)]
    p2.json_output(pred,pred_cuisine_score,n_closest)
    out,err = capfd.readouterr()
    assert type(out) == str