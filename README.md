# ConfidenceBasedNLPSecurity

The generate_attacked.py file allows for attack generation given a dataset. This can store only the successful attacks, or all attacks depending on if log_to_csv is passed into textattack, and whether or not a log_file is given to creating the attack.

Once generated, logits.py has functions process_csv and process_csv2 that will process the csv. The former is for csv's generated via TextAttack, while the latter is for files generated by our method in the attack.

The current version of the code only supports the former version for analysis, although it can easily be adapted for either. The defense_analysis and visualize_distributions jupyter notebooks can be run with the data files to get the corresponding results. These utilize the defense and util_plotting files, although neither are meant to be run by themselves.

The all_log_file_success_analysis notebook provides information about the success rate of the attack at various levels of confidence.

Attack data files are given for adversarial examples (base), far boundary (0.8), very far boundary (0.97), and very very far boundary (0.997) on the IMDB dataset with DistilBERT as described in the paper.
