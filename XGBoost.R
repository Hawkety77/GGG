library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(doParallel)
library(themis)
library(xgboost)

detectCores() #How many cores do I have?
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

df_train <- vroom('train.csv')
df_test <- vroom('test.csv')

my_recipe <- recipe(type ~ ., data = df_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize() %>%
  step_rm(id)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

rand_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

my_model <- boost_tree(mode = "classification", 
                       trees = tune(),
                       tree_depth = tune(),
                       learn_rate = tune()) %>%
  set_engine("xgboost")

param_grid <- grid_regular(
  trees(c(100, 300)),
  tree_depth(),
  learn_rate())

tree_tune <- tune_grid(
  my_model, 
  my_recipe, 
  vfold_cv(df_train, v = 5, repeats = 1), 
  control = control_grid(save_workflow = TRUE), 
  grid = param_grid
)

collect_metrics(tree_tune)
show_best(tree_tune, metric = 'accuracy')

best_tune <- tree_tune %>%
  select_best("accuracy")

final_workflow <- rand_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = df_train)

ggg_predictions <- predict(final_workflow, 
                           new_data = df_test, 
                           type = 'class')

submission <- ggg_predictions %>%
  bind_cols(df_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

write_csv(submission, 'submission_randforest.csv')

stopCluster(cl)

