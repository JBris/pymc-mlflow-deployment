import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

bayes_reg_runner = bentoml.mlflow.get("bayes_reg:latest").to_runner()

svc = bentoml.Service("bayes_reg", runners=[bayes_reg_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    result = bayes_reg_runner.predict.run(input_series)
    return result