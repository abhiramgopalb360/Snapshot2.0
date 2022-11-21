# Vyper Introduction

Vyper is the Python counterpart to the R version of Viper.

## Installation

Use pip to install a local copy of the package through Vyper folder of the Github directory. For example, if your Github directory is located in your C drive:
```
python -m pip install -e C:\GitHub\Vyper
```

## Basic Usage
### Model Creation
Import the base model package from Vyper:
```
from vyper.user import Model
```

With your pandas DataFrame `df` create a model with parameter values of your choosing or with the default values:
```
m = Model(data=df,
          dependent_variable='y',
          na_drop_threshold=0.85,
          training_percentage=0.75,
          model_type='linear')
```

### TnI
Once your model is created, you can perform transformations and imputations on your data set. Start with the training data set with `tni_smart`. This function has multiple parameter options available but you can also use the default setup:
```
m.tni_smart()
```

After your training data set has been processed, you can perform this same TnI process on your test data set with `tni_transform`:
```
m.tni_transform()
```

### Variable Reduction
After both train and test sets have been processed, reduce the variables selected for the model using `variable_reduction`:
```
m.variable_reduction(cluster_type='hclust',
                     wt_corr_dv=1,
                     wt_univ_reg=1,
                     wt_inf_val=1,
                     wt_clust=1)
```

Additionally, you can further reduce these variables using `lasso_reduction`:
```
m.lasso_reduction()
```

### Fitting
Fit the model and return the results with `fit`:
```
final_results = m.fit()
```

Inside the `final_results` output you can view, `train_out`, `test_out`, `model_descriptor`, `model_metric`, `train_lift`, and `test_lift`.

### Playbook
Create the model playbook with `create_model_playbook` to create the Excel file detailing the model created:
```
m.create_model_playbook("Playbook Title", filepath)
```

If running a linear model, the `width_bin` parameter will need a value.
