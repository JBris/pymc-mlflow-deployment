#!/usr/bin/env bash

CMD="docker compose exec mlflow python -m pymc_mlflow"

${CMD}.1_train_model

# docker compose exec mlflow bentoml serve --host 0.0.0.0 -p 3000 pymc_mlflow.2_deploy_model:svc
