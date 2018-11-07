# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:09:37 2018

@author: Antoine Pham

Reference : "https://www.datacamp.com/community/tutorials/machine-learning-models-api-python"
"""

# Dependencies
from flask import Flask, request, jsonify
import datetime as datetime
import traceback
import pandas as pd
import numpy as np
import sys

# API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if pred:
        try:
            json_ = request.json # {NbDiners,UserID,CreationDatetime, Event, top, user_age, out_format}
            print(json_)
            query = pd.DataFrame(json_)
            query = query.reindex(columns=['NbDiners','UserID', 'Event','top', 'user_age', 'out_format'])
            print(query)
            CreationDatetime = pd.to_datetime(datetime.datetime.now()- datetime.timedelta(hours=0.4))
            W1,W2,W3=pred.best_w
            prediction=list()
            for i in range(len(query)):
                f = pred.Feature(int(query.loc[i,'NbDiners']),int(query.loc[i,'UserID']), CreationDatetime) # calling the class 'Feature'
                user_profile=f.get_user_profile()
                time_delta=f.get_timedelta()
                # adding events under a sequence
                if i==0:
                    f.add_Event(str(query.loc[i,'Event']))
                    event_vector = f.Event2vec()
                elif query.loc[i,'UserID'] not in query['UserID'].tolist()[:i]:
                    f.add_Event(str(query.loc[i,'Event']))
                    event_vector = f.Event2vec()
                else:
                    query1 = query.loc[query['UserID']==query.loc[i,'UserID'],:]
                    query1=query1.reset_index()
                    print(query1)
                    for j in range(len(query1)):
                        f.add_Event(str(query1.loc[j,'Event']))
                    event_vector = f.Event2vec()
                                    
                # applying conditions about the number of items being displaying and user's age
                if np.isnan(query.loc[i,'top']):
                    if np.isnan(query.loc[i,'user_age']):
                        rec = pred.Recommendation(pred.item_context_pred,pred.Products,user_profile,event_vector,time_delta,W1,W2,W3)
                        prediction.append(rec.get_prediction(out_format=str(query.loc[i,'out_format'])))
                    else:
                        rec = pred.Recommendation(pred.item_context_pred,pred.Products,user_profile,event_vector,time_delta,W1,W2,W3,user_age=int(query.loc[i,'user_age']))
                        prediction.append(rec.get_prediction(out_format=str(query.loc[i,'out_format'])))
                else:
                    if np.isnan(query.loc[i,'user_age']):
                        rec = pred.Recommendation(pred.item_context_pred,pred.Products,user_profile,event_vector,time_delta,W1,W2,W3,top=query.loc[i,'top'])
                        prediction.append(rec.get_prediction(out_format=str(query.loc[i,'out_format'])))
                    else:
                        rec = pred.Recommendation(pred.item_context_pred,pred.Products,user_profile,event_vector,time_delta,W1,W2,W3,top=int(query.loc[i,'top']),user_age=int(query.loc[i,'user_age']))
                        prediction.append(rec.get_prediction(out_format=str(query.loc[i,'out_format'])))
            

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Need to train a model first')
        return ('No model founded')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 23456 # If no port provided the port will be set to 23456

    
    import pred2prod as pred
    print ('Model loaded!')
    

    app.run(port=port, debug=True)