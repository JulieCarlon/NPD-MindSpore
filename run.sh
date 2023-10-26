# step1 generate poisoned data sample
python attack/data_poison.py
# step2 train a backdoored model
python attack/train_backdoor.py

# step3 defense the backdoored model
python defense/ft.py
python defense/ft_sam.py
python defense/npd.py
python defense/sau.py

