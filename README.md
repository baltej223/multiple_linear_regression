# Scratch impl of OLS Multiple regressing pipeline

### Stats

- also data not standardized.

```
actual: 89400.00, predicted: 65105.38, error: 24294.62
RMSE: 66197.29
MAE:  47552.42
R²:   0.6981
```

### Next

```
                RMSE       MAE       R²
Raw features    66197     47552    0.6981
Standardized    ?????     ?????    ?????
```

- Since its an OLS pipeline, I wanted to add ridge or lasso regression for better fit, but for that I need to change it to use gradient decent.
