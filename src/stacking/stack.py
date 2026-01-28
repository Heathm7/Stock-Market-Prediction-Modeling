from typing import Any, Dict

class BaseModel:
    def predict(self, X):
        raise NotImplementedError("Predict method not implemented")

class RegimeStack:
    def __init__(self, models: Dict[str, BaseModel]):
        self.models = models
        self.secondary_weight = 0.25

    def predict(self, X: Any, regime_score: float) -> float:
        primary_stack = None
        secondary_stack = None

        if 0 <= regime_score < 1:
            #stagnant
            primary_stack = self.models['tree_0']
            secondary_stack = None
        elif 1 <= regime_score < 2:
            primary_stack = self.models['tree_1']
            secondary_stack = self.models.get('linear_1', None)
        elif 2 <= regime_score <= 3:
            if 2.0 <= regime_score <= 2.5:
                primary_stack = self.models['linear_2a']
                secondary_stack = self.models.get('tree_2a', None)
            else:
                primary_stack = self.models['linear_2b']
                secondary_stack = self.models.get('neural_2b', None)
        elif 3 < regime_score <= 4:
            primary_stack = self.models['neural_3']
            secondary_stack = self.models.get('linear_3', None)
        elif regime_score > 4:
            #high volaility
            primary_stack = self.models['neural_4']
            secondary_stack = None
        else:
            raise ValueError(f"Regime score {regime_score} out of bounds")

        primary_pred = primary_stack.predict(X)

        if secondary_stack:
            secondary_pred = secondary_stack.predict(X)
            final_pred = primary_pred + self.secondary_weight * secondary_pred
        else:
            final_pred = primary_pred

        return final_pred

           



