library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(doParallel)
library(themis)
library(keras)

# detectCores() #How many cores do I have?
# cl <- makePSOCKcluster(7)
# registerDoParallel(cl)

df_train <- vroom('train.csv')
df_test <- vroom('test.csv')

my_recipe <- recipe(type ~ ., data = df_train) %>%
  step_mutate(color = as.factor(color)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1) %>%
  step_rm(id) %>%
  step_normalize()

prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
set_engine("nnet") %>% 
  set_mode("classification")

workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 50)),
                            levels= 3)

folds <- vfold_cv(df_train, v = 5, repeats = 1)

tuned_nn <- tune_grid(nn_model,
                       my_recipe,
                       folds,
                       control = control_grid(save_workflow = TRUE),
                       grid = nn_tuneGrid)

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + 
  geom_line()

collect_metrics(tuned_nn)
show_best(tuned_nn, metric = 'accuracy')

best_tune <- tuned_nn %>%
  select_best("accuracy")

final_workflow <- workflow %>%
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

write_csv(submission, 'submission_NeuralNet.csv')

# stopCluster(cl)

