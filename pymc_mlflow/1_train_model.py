import bentoml
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from sklearn import svm
from sklearn import datasets
import mlflow
from mlflow.models import infer_signature
from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc_experimental.model_builder import ModelBuilder
import os.path as osp

from numpy.random import RandomState

class LinearModel(ModelBuilder):
    _model_type = "LinearModel"
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        X_values = X["input"].values
        y_values = y.values if isinstance(y, pd.Series) else y
        self._generate_and_preprocess_model_data(X_values, y_values)

        with pm.Model(coords=self.model_coords) as self.model:
            x_data = pm.MutableData("x_data", X_values)
            y_data = pm.MutableData("y_data", y_values)

            a_mu_prior = self.model_config.get("a_mu_prior", 0.0)
            a_sigma_prior = self.model_config.get("a_sigma_prior", 1.0)
            b_mu_prior = self.model_config.get("b_mu_prior", 0.0)
            b_sigma_prior = self.model_config.get("b_sigma_prior", 1.0)
            eps_prior = self.model_config.get("eps_prior", 1.0)

            a = pm.Normal("a", mu=a_mu_prior, sigma=a_sigma_prior)
            b = pm.Normal("b", mu=b_mu_prior, sigma=b_sigma_prior)
            eps = pm.HalfNormal("eps", eps_prior)

            obs = pm.Normal("y", mu=a + b * x_data, sigma=eps, shape=x_data.shape, observed=y_data)

    def _data_setter(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None
    ):
        if isinstance(X, pd.DataFrame):
            x_values = X["input"].values
        else:
            x_values = X[:, 0]

        with self.model:
            pm.set_data({"x_data": x_values})
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})

    @staticmethod
    def get_default_model_config() -> Dict:
        model_config: Dict = {
            "a_mu_prior": 0.0,
            "a_sigma_prior": 1.0,
            "b_mu_prior": 0.0,
            "b_sigma_prior": 1.0,
            "eps_prior": 1.0,
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        sampler_config: Dict = {
            "draws": 100,
            "tune": 100,
            "chains": 4,
            "target_accept": 0.95,
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        return self.model_config

    def _save_input_params(self, idata) -> None:
        pass

    def _generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray]
    ) -> None:
        self.model_coords = None  
        self.X = X
        self.y = y

class BayesModel(mlflow.pyfunc.PythonModel):
    def __init__(self, module):
        self.module = module
        self.model = None

    def load_context(self, context):
        model_instance = self.module()
        idata = context.artifacts["idata"]
        self.model = model_instance.load(idata)

    def predict(self, context, model_input, params={}):
        if self.model is None:
            raise ValueError(
                "The model has not been loaded. "
            )
        
        prediction_data = pd.DataFrame({"input": model_input})
        predict_proba = params.get("predict_proba")
        return self.model.predict(prediction_data)
    
        # if predict_proba:
        #     return self.model.predict_proba(prediction_data)
        # else:
        #     return self.model.predict(prediction_data)
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")

    EXPERIMENT_CONFIG = instantiate(config["experiment"])
    mlflow.set_tracking_uri(EXPERIMENT_CONFIG.tracking_uri)
    experiment_name = EXPERIMENT_CONFIG.name

    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    mlflow.set_tag("task", "pymc_mlflow_model")

    x = np.linspace(start=0, stop=1, num=100)
    X = pd.DataFrame(data=x, columns=["input"])
    y = 0.3 * x + 0.5 + rng.normal(0, 1, len(x))

    model = LinearModel()
    idata = model.fit(X, y)
    fname = osp.join("outdir", "linear_model_v1.nc")
    model.save(fname)

    model_2 = LinearModel.load(fname)
    x_pred = np.random.uniform(low=1, high=2, size=10)
    prediction_data = pd.DataFrame({"input": x_pred})
    model_2.predict(prediction_data)

    signature = infer_signature(
        x, 
        y,
        params = { "predict_proba": True }
    )
    
    model_name = "bayes_reg"
    run_id = mlflow.active_run().info.run_id
    print(run_id)
    bayes_model = BayesModel(module=LinearModel)

    logged_model = mlflow.pyfunc.log_model(
        python_model = bayes_model, 
        artifact_path = model_name, 
        artifacts={"idata": fname},
        signature = signature
    )
    model_uri = logged_model.model_uri
    mlflow.register_model(model_uri, model_name)

    bento_model = bentoml.mlflow.import_model(
        'bayes_reg', 
        model_uri,
        labels=mlflow.active_run().data.tags,
        metadata={
            "metrics": mlflow.active_run().data.metrics,
            "params": mlflow.active_run().data.params,
        }
    )

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    pred_mean = loaded_model.predict(x_pred, params = {})
    # pred_samples = loaded_model.predict(
    #     x_pred, params = { "predict_proba": True }
    # )

    print(pred_mean)
    mlflow.end_run()

if __name__ == "__main__":
    main()