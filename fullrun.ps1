$seed              = 42
$holdout_split_pct = 0.1
$test_split_pct    = 0.3
$feature_count     = 13
$kfolds            = 10
python .\splitter.py   --seed $seed --testsplit $holdout_split_pct .\DATA\telco-churn.csv .\DATA\telco-churn-train-test.csv .\DATA\telco-churn-holdout.csv
python .\dataclean.py  --seed $seed --topk $feature_count .\DATA\telco-churn-train-test.csv .\DATA\telco-churn-train-test-clean.csv
python .\fieldnames.py .\DATA\telco-churn-train-test-clean.csv .\DATA\fieldnames.txt
$fields = Get-Content .\DATA\fieldnames.txt 
python .\dataclean.py  --seed $seed --topk $feature_count --fields=$fields .\DATA\telco-churn-holdout.csv .\DATA\telco-churn-holdout-clean.csv
python .\train.py      --seed $seed --testsplit $test_split_pct --kfolds $kfolds .\DATA\telco-churn-train-test-clean.csv .\MODELS\
python .\ensembler.py  --seed $seed --testsplit $test_split_pct --kfolds $kfolds .\DATA\telco-churn-train-test-clean.csv .\MODELS final_model.pkl
python .\classify.py .\MODELS\final_model.pkl .\DATA\telco-churn-holdout-clean.csv