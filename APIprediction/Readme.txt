
# This project aims to put a Machine Learning model into  API (Posman) allowing to test inputs and verify outputs.

Requirement: Python 3.6 64bit Flask 1.0.2 Posman 6.4.4

_Code organization:

All paths required to ...APIprediction/...
	
_Code organization:
	
	* .prediction/:

		- "api.py" : a python method that will do the following:
						+ Load the Machine Learning model containing the model parameters saved under ".csv" file  into memory when the application starts
						+ Create an API  that takes input variables, transforms them into the appropriate format, and returns predictions.

		- "pred2prod.py"  : the Machine Learning model
						

		- "math_utils.py": functions and anything else to implement the algorithm

_ Sample:

	*  .params/:

		- "cluster_model.pkl" : a clustering model serialized and saved  within training task=>to find out the context related to a given moment given weather information

		- "item_context_pred.csv" : prediction based on the actual context

		- "mat_minmax.csv": statistics that will be used as a part of visiting time computation

		- "NN_model.pkl" ( optional): a neural network model serialized and saved  while training tasks; this model can be used to predict the visiting time of a given user

		- "item_event_weighted.csv": a parameter matrix based on all existing items & all possible events

		- "item_context_prob.csv" : a parameter matrix based on all existing items & all pre-defined contexts; mxn matrix where m = number of pre defined contexts (m=10); n= number of items;  

		- "item_profile.csv" : a parameter matrix based on all existing items & user profile; nxk matrix , where n= number of items; k = number of attributes (11) representing a given user 

		- "seen_data.csv" : uploaded data containing informations of each seen user  

		- "visit_duration_stat.csv":  a parameter matrix based on all existing items & user's visiting time: a matrix includes statistics associated to each item,  
		
		- "item_desc.csv" :  data containing product description with  attributs such as:" product group", " working order"," ingredients", " flavors"..., used in NLP tasks to do parameter transfert 
		
		- "products.csv" : data containing product descriptions used to make predictions

		- "X.csv": inputs of the pre-trained model on the reference customer

_Run application: 
1.run the file 'api.py' from the terminal
2. once all files compiled, enter the right Url into Posman ( POST)
3. to test the API, enter under Json format inputs required by the model
Related link: https://www.datacamp.com/community/tutorials/machine-learning-models-api-python
