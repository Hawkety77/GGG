library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(doParallel)
library(themis)
library(naivebayes)
library(discrim)

detectCores() #How many cores do I have?
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

df_train <- vroom('train.csv')
df_test <- vroom('test.csv')

my_recipe <- recipe(type ~ ., data = df_train) %>%
  step_mutate(color = as.factor(color)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) #%>%
  # step_normalize()

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

my_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("naivebayes")

am_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

tuning_grid <- grid_regular(Laplace(), 
                            smoothness(),
                            levels = 5)

folds <- vfold_cv(df_train, v = 5, repeats = 1)

tree_tune <- tune_grid(my_model, 
                       my_recipe, 
                       folds, 
                       control = control_grid(save_workflow = TRUE), 
                       grid = tuning_grid)

collect_metrics(tree_tune)
show_best(tree_tune, metric = 'accuracy')

best_tune <- tree_tune %>%
  select_best("accuracy")

final_workflow <- am_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = df_train)

#########

ggg_predictions <- predict(final_workflow, 
                           new_data = df_test, 
                           type = 'class')

submission <- ggg_predictions %>%
  bind_cols(df_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

write_csv(submission, 'submission_naivebayes.csv')

stopCluster(cl)
