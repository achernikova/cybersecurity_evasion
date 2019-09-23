# cybersecurity_evasion


## Project structure

* attack/
  * maximum.txt - maximum feature values
  * minimum.txt - minimum feature values
  * model_whole_scenarios19 - trained model
  * neris_attack.py - attack class
  * neris_model_data_utilities.py - utilities functions
  * scaler_scenarios19 - features scaler
* data/
  * features_stat_scenario2.csv - data from the second scenario
* results/
  * botnets.txt - ids of botnet traffic
  * success_rate.py - functionality for plotting success rate
  * plot_ROC_curves.py - functionality for plotting ROC curves
  
* training/
  * train.py - training model on scenarios 1, 9 (data for scenario 1:https://drive.google.com/open?id=1ZHGihxd2EJ9NuKbeWkqfoL3QfCEktV_D ; data for scenario 9: https://drive.google.com/open?id=14dU9HIDSNS9pcf-3_bnXbfdtzstWiS83 )

  
  ## Training
  For training the model ruh train.py file, it will save the weight of trained model on scenarios 1 and 9 to 'model_whole_scenarios19' file.  
  
  ## Data
  Data for testing the attack is under data/ folder, it corresponds to the second scenario.
  
  ## Performing Attack
  In order to perform the attack on the testing data from the second scenario, just neris_attack.py file under attack.py folder.
  
  ## Plotting Results
  In order to plot attack's success rate, run the success_rate.py file under results/ folder, in order to plot ROC curves run plot_ROC_curves.py under results/ folder.
  
  

