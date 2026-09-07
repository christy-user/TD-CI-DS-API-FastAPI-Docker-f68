
import pytest
from fastapi.testclient import TestClient
 
from app.utils import predict
from app.main import app 


client = TestClient(app)

# 1- Test d'une prédiction correcte

def test_predict_correct ():
    features = [1.0, 2.0, 3.0]
    expected = [2.0, 4.0, 6.0]

    result = predict(features)

    assert result == pytest.approx(expected)
 
# 2- test d'une prédiction incorrecte

def test_predict_incorrect():
    features = [1.0, 2.0, 3.0]
    expected = [4.0, 6.0, 8.0]

    result = predict(features)

    assert result != pytest.approx(expected)

# 3- test d'un JSON incorrect

def test_predict_json_incorrect():
    response = client.post(
        "/predict",
        json={"values" : [3.5, 1.2, 4.9] }                                                                                                                              
    )

    assert response.status_code == 422

# -----------------------------------------------------------------------------
# Cas nominaux : entrées valides et représentatives
# -----------------------------------------------------------------------------
 
@pytest.mark.parametrize(
    "features, expected",
    [
        ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        ([5.0], [10.0]),
        ([1.5,2.5] , [3.0,5.0]),
    ],
)
 
def test_predict_nominal_cases(features, expected):
    """La fonction doit retourner une prédiction conforme à la règle y = 2x."""
    result = predict(features)
    assert result == pytest.approx(expected)

#--------------------------------------------------------------------------------
# Cas limites : valeurs particulières aux frontières du domaine 
#--------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "features, expected",
    [
        ([0.0], [0.0]),
        #([], []),
        ([-1.0] , [-2.0]),
        ([-1000.0], [-2000.0]),
        ([1000000.0], [2000000.0]),
    ],
)
 
def test_predict_boundary_cases(features, expected):
    """La fonction doit gérer correctement les valeurs limites ou particulière."""
    result = predict(features)
    assert result == pytest.approx(expected)

#--------------------------------------------------------------------------------
# Cas exceptionnel : entrée techniquement sous forme de liste, mais inexploitable
#--------------------------------------------------------------------------------

def test_predict_empty_list_raises_exception():
    """
    une liste vide ne contient aucun echantillon à prédire.
    Le modèle scikit-learn doit donc lever une exception.
    """
    with pytest.raises(ValueError):
        predict([])   

#--------------------------------------------------------------------------------
# Cas Invalides : données ne respectant pas les préconditions attendues
#--------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_features",
    [
        None,
        "abc",
        {"feature1" : 1.0},
        [1.0, "abc", 3.0],
        [1.0, None, 3.0],
    ],
)
 
def test_predict_invalid_raise_exception(invalid_features):
    """La fonction doit échouer de manière contrôlée avec des entrées invalides."""
    with pytest.raises((TypeError, ValueError)):
        predict(invalid_features)   

#--------------------------------------------------------------------------------
# Propriétés générales : tests plus robustes qu'une simple valeur attendue
#--------------------------------------------------------------------------------

def test_predict_output_is_a_list():
    """ La fonction doit retourner une liste python. """
    result = predict([1.0, 2.0, 3.0, 4.0])
  
    assert isinstance(result, list)

def test_predict_output_size_matches_input_size():
    """ Le nombre de prédictions doit correspondre au nombre d'entrées. """
    features = [1.0, 2.0, 3.0, 4.0]
    result = predict(features)

    assert len(result) == len (features)


