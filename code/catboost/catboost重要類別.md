# 

### CatBoostClassifier
```
catboost.ai/docs/concepts/python-reference_catboostclassifier.html
```
```
class CatBoostClassifier(

iterations=None,
learning_rate=None,
depth=None,
l2_leaf_reg=None,
model_size_reg=None,
rsm=None,
loss_function=None,
border_count=None,
feature_border_type=None,
per_float_feature_quantization=None,                         
input_borders=None,
output_borders=None,
fold_permutation_block=None,
od_pval=None,
od_wait=None,
od_type=None,
nan_mode=None,
counter_calc_method=None,
leaf_estimation_iterations=None,
leaf_estimation_method=None,
thread_count=None,
random_seed=None,
use_best_model=None,
verbose=None,
logging_level=None,
metric_period=None,
ctr_leaf_count_limit=None,
store_all_simple_ctr=None,
max_ctr_complexity=None,
has_time=None,
allow_const_label=None,
classes_count=None,
class_weights=None,
one_hot_max_size=None,
random_strength=None,
name=None,
ignored_features=None,
train_dir=None,
custom_loss=None,
custom_metric=None,
eval_metric=None,
bagging_temperature=None,
save_snapshot=None,
snapshot_file=None,
snapshot_interval=None,
fold_len_multiplier=None,
used_ram_limit=None,
gpu_ram_part=None,
allow_writing_files=None,
final_ctr_computation_mode=None,
approx_on_full_history=None,
boosting_type=None,
simple_ctr=None,
combinations_ctr=None,
per_feature_ctr=None,
task_type=None,
device_config=None,
devices=None,
bootstrap_type=None,
subsample=None,
sampling_unit=None,
dev_score_calc_obj_block_size=None,
max_depth=None,
n_estimators=None,
num_boost_round=None,
num_trees=None,
colsample_bylevel=None,
random_state=None,
reg_lambda=None,
objective=None,
eta=None,
max_bin=None,
scale_pos_weight=None,
gpu_cat_features_storage=None,
data_partition=None
metadata=None, 
early_stopping_rounds=None,
cat_features=None, 
grow_policy=None,
min_data_in_leaf=None,
min_child_samples=None,
max_leaves=None,
num_leaves=None,
score_function=None,
leaf_estimation_backtracking=None,
ctr_history_unit=None,
monotone_constraints=None
)

```

### Pool
```
Pool是catboost中的用於組織資料的一種形式，
也可以用numpy array和dataframe。但更推薦Pool，其記憶體和速度都更優。

Dataset processing
The fastest way to pass the features data to the Pool constructor 
(and other CatBoost, CatBoostClassifier, CatBoostRegressor methods that accept it) 
if most (or all) of your features are numerical is to pass it using FeaturesData class. 

Another way to get similar performance with datasets that contain numerical features only 
is to pass features data as numpy.ndarray with numpy.float32 dtype.
```

```
class Pool(
data, 
label=None,
cat_features=None,
column_description=None,
pairs=None,
delimiter='\t',
has_header=False,
weight=None, 
group_id=None,
group_weight=None,
subgroup_id=None,
pairs_weight=None
baseline=None,
feature_names=None,
thread_count=-1
)

```
