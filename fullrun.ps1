python .\splitter.py --seed 42 --testsplit 0.1 .\DATA\telco-churn.csv .\DATA\telco-churn-train-test.csv .\DATA\telco-churn-holdout.csv
python .\dataclean.py --seed 42 --topk 15 .\DATA\telco-churn-train-test.csv .\DATA\telco-churn-train-test-clean.csv
python .\fieldnames.py .\DATA\telco-churn-train-test-clean.csv .\DATA\fieldnames.txt
$fields = Get-Content .\DATA\fieldnames.txt 
python .\dataclean.py --seed 42 --topk 15 --fields=$fields .\DATA\telco-churn-holdout.csv .\DATA\telco-churn-holdout-clean.csv
python .\train.py --seed 42 --testsplit 0.3 --kfolds 7 .\DATA\telco-churn-train-test-clean.csv .\MODELS\
python .\ensembler.py --seed 42 --testsplit 0.3 --kfolds 7 .\DATA\telco-churn-train-test-clean.csv .\MODELS final_model.pkl
python .\classify.py .\MODELS\final_model.pkl .\DATA\telco-churn-holdout-clean.csv