# Application

## Scoring 

To run data attribution methods for application tasks that include Toxicity/Bias Filtering and Factual Attribution, you would need a configuration stored in **config/**

Example: 
```json
method: LESS
# Supported methods: Grad_Dot, Grad_Sim, LESS, DataInf, EKFAC

task: "Toxicity/Bias"
subset: "XSTest-response-Het"
# Supported subset: XSTest-response-Het, XSTest-response-Hom, ToxicChat-Het, ToxicChat-Hom, XSTest-Het, XSTest-Hom

base_model_path: "EleutherAI/pythia-1b"
# Supported model: Pythia-1b, Llama-3.2-1B, Llama-3.1-8B
checkpoint: "DataAttributionEval/Pythia-1b-XSTest-response-Het"

device: "cuda"

##################################
# hyperparameter used for DataInf
# regularization: regularization term added to hessian
# fim_estimate_data_ratio: ratio of full training data used to estimate hessian
regularization: 1e-5
fim_estimate_data_ratio: 1.0
##################################

##################################
# hyperparameter used for EKFAC
# regularization: regularization term added to hessian
damping: 1e-7
##################################

##################################
# hyperparameter used for LESS
# proj_dim: projection dimension of gradients
proj_dim: 8192
##################################

save_path: ./
```

Then, run the script under **methods/** to generate scores

```sh
bash methods/run-application.sh configs/factual-attribution.yaml
```

```sh
bash methods/run-application.sh configs/toxicity-bias.yaml
```

## Evaluation

To evaluate the generated scores, run the script under **evaluations/**

```sh
bash evaluation/evaluate_application.sh configs/factual-attribution.yaml /path/to/score
```

```sh
bash evaluation/evaluate_application.sh configs/toxicity-bias.yaml /path/to/score
```

