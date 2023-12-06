library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(doParallel)
library(themis)
library(dbarts)

# detectCores() #How many cores do I have?
# cl <- makePSOCKcluster(7)
# registerDoParallel(cl)

df_train <- vroom('train.csv')
df_test <- vroom('test.csv')

my_recipe <- recipe(type ~ ., data = df_train) %>%
  step_mutate(color = as.factor(color)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_normalize()

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

my_model <- bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("classification")

bart_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

tuning_grid <- grid_regular(trees(), 
                            levels = 2)

folds <- vfold_cv(df_train, v = 5, repeats = 1)

tree_tune <- tune_grid(my_model,
                       my_recipe,
                       folds,
                       grid = tuning_grid)

collect_metrics(tree_tune)
show_best(tree_tune, metric = 'accuracy')

best_tune <- tree_tune %>%
  select_best("accuracy")

final_workflow <- bart_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = df_train)

#########

# my_model <- rand_forest(trees = 1000, 
#                         min_n = 21, 
#                         mtry = 10) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# final_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(my_model) %>%
#   fit(data = df_train)

#########

ggg_predictions <- predict(final_workflow, 
                           new_data = df_test, 
                           type = 'class')

submission <- ggg_predictions %>%
  bind_cols(df_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

write_csv(submission, 'submission_bart.csv')

# stopCluster(cl)
