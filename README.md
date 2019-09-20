# cybersecurity_evasion


## Project structure

* attack/
  * maximum.txt 
  * minimum.txt
  * model_whole_scenarios19
  * neris_attack.py
  * neris_model_data_utilities.py
  * scaler_scenarios19
* data/
  * features_stat_scenario2.csv - data from the second scenario
* results/
  * botnets.txt - ids of botnet traffic
  * success_rate.py - functionality for plotting success rate
  * plot_ROC_curves.py - functionality for plotting ROC curves

  
  
  ## Data
  Data for testing the attack is under data/ folder, it corresponds to the second scenario.
  
  ## Performing Attack
  In order to perform the attack on the testing data from the second scenario, just neris_attack.py file under attack.py folder.
  
  ## Plotting Results
  In order to plot attack's success rate, run the success_rate.py file under results/ folder, in order to plot ROC curves run plot_ROC_curves.py under results/ folder.
  
  

