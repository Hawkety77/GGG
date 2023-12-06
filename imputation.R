library(tidyverse)
library(tidymodels)
library(vroom)


df_missing <- vroom('trainWithMissingValues.csv')
df_train <- vroom('train.csv')

my_recipe <- recipe(type ~ ., data = df_missing) %>%
  # step_impute_linear(var, impute_with= all_numeric_predictors())
  step_impute_mean(all_numeric_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

rmse_vec(df_train[is.na(df_missing)], baked[is.na(df_missing)])
