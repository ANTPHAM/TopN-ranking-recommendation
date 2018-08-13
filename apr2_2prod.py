# -*- coding: utf-8 -*-
"""
Title: "Recommender System by Top_N Ranking Based Method_ Application "

	This script can be executed to obtain both external data and the features of the model from streaming input data. 
    
    It also contains an algorithm  which aims to predict items with a high probability of being bought by a customer.
    
    The scrip written under Python can be interacted with C# to deploy the model producing predictions.

	Authors : Antoine Pham
	Date: from July to November 2017
"""


#%%
import numpy as np
from builtins import print
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
from datetime import timedelta
from pandas.io.json import json_normalize
from sklearn.externals import joblib
from urllib.request import urlopen
import json
from math_utils2prod import *
import os
import item_by_context
import schedule
import time
rootPath=os.path.dirname(os.path.realpath(__file__)) + "/../"
paramsPath=rootPath + "params/"
#%%

#%%
#==============================================================================
#                 IMPORT MATRIX CONTAINING FIT PARAMETERS OF THE MODEL  
#==============================================================================
X=pd.read_csv(paramsPath + 'X.csv',sep=',')
X=X.set_index(['D.OrderHeaderID','D.PersonID'])
M_prob=pd.read_csv(paramsPath + 'M_prob.csv',sep=',')
IT_PROFILE=pd.read_csv(paramsPath + 'IT_PROFILE.csv',sep=',')
visit_duration_stat=pd.read_csv(paramsPath + 'visit_duration_stat.csv',sep=',')
It_event_weighted=pd.read_csv(paramsPath + 'It_event_weighted.csv',sep=',')

#some more data preparations 
M_prob=M_prob.set_index(['Unnamed: 0'])
M_prob.index.name=None
IT_PROFILE=IT_PROFILE.set_index(['Unnamed: 0'])
IT_PROFILE.index.name=None
It_event_weighted=It_event_weighted.set_index(['Unnamed: 0'])
It_event_weighted.index.name=None
visit_duration_stat=visit_duration_stat.set_index(['Unnamed: 0'])
visit_duration_stat.index.name=None

seen_data=pd.read_csv(paramsPath + 'seen_data.csv',sep=',')
best_w=[0.8389266,0.15078162,0.01029177] 
Item_context_pred=pd.read_csv(paramsPath + 'Item_context_pred.csv', sep=',')
Item_context_pred.head(5)
Item_context_pred.tail(5)
#%%

#%%
Products=pd.read_csv(paramsPath+"products_Bordeaux.csv",sep=';') 
Products.columns=['ID','ProductID','ProductName','ProductGroupID','ProductGroupName','GrossPrice','NetPrice','GuidedActivityID','WorkingOrder']
Working_Order_type={0:'Menu', 1:'Beverage', 2:'Starter', 3 :'MainDishes',
                    4 :'Dessert', 5:'Coffee', 6:'Digestive', 7:'Others'}   

Products['WorkingOrderType']=Products['WorkingOrder']
Products['WorkingOrderType'].replace(Working_Order_type, inplace=True)

products=pd.DataFrame(Products.groupby(['ProductGroupName','ProductGroupID','ProductName','ProductID',
                                 'WorkingOrder','WorkingOrderType'],as_index=False).sum()) 
products.head(2)
#%%

#%%
alcohol=['KIR','KIR ROYAL','PUNCH Maison','SANGRIA','RHUM ARRANGE','PICHET PUNCH',
				   'PICHET SANGRIA','WHISKY','RHUM','VODKA','GIN','GET','TEQUILA','MALIBU','BAILEYS','RICARD',
				   'PASTIS','PINEAU','PORTO','LILLET','MARTINI','CHAMPAGNE BOUTEILLE','Cocktail Saint Valentin alcoolisé','CHAMPAGNE COUPE','PELFORTH',
				   'PINT PELFORTH','GRIMBERGEN','HOEGGARDEN','DESPERADOS','KEKETTE EXTRA','KEKETTE RED',
				   'BIERE SANS GLUTEN','EXPRESSO','DECA','DOUBLE EXPRESSO','PAPOLLE BLANC SEC','VERRE PAPOLLE BLANC SEC',
		   'Verre de Bordeaux Rouge Agape','PAPOLLE BLANC MOEL','VERRE PAPOLLE BLANC MOEL','PAPOLLE ROSE','VERRE PAPOLLE ROSE','BIERE PRESSION','EXPRESSO SD',
		   'DECA SD','THE SD','INFUSION SD','BIERE',' PRESSION MAX MENU','BIERE PINTH','Verre Rosé la Colombette','VERRE MAX CUVEE',
		   'MAX CUVEE ','Verre de Boisson Rouge','Verre Chardonnay domaine des deux ruisseaux','VERRE PAPOLLE ROUGE','Domaine La Colombette Rosé','Boisson Rouge']
#%%

#%%
#==============================================================================
#                                DATA PREPROCESSING
#==============================================================================
# create a list containing all of these names
oldnames=It_event_weighted.index[It_event_weighted.index.str.contains('_event')]
# remove the string '_event' and create a new list of name
newnames=[oldnames[i][:-6] for i in range(len(oldnames))]
# rename these click events by the list 'newnames'
for i in range(len(It_event_weighted.index)):
    if It_event_weighted.index.values[i] in (oldnames):
        # put  .values
        It_event_weighted.index.values[i]= It_event_weighted.index.values[i][:-6]
    else:
        It_event_weighted.index.values[i]= It_event_weighted.index.values[i]
	
#%%


#==============================================================================
#                   CREATING A CLASS TO GET INPUT DATA AND DO FEATURE ENGINEERING TASKS
#==============================================================================

#%%
class Feature:
    '''
    Get raw inputs and transform these ones into the features which will be feed to the model
    '''

    def __init__(self,NbDiners,UserID,CreationDatetime):
        self.NbDiners = NbDiners
        self.UserID = UserID
        self.CreationDatetime = CreationDatetime
        self.ParameterList = list()
    
    def get_nbpartner(self):
        # getting the 1st input
        nb_partner=list()
        for i in (partner,partner1,partner2,partnergr):
            nb_partner.append(i(self.NbDiners))
        return nb_partner   
    
    def get_visit(self): 
        # define if it is a new user, a returning user or a new user
        def top_visit(UserID):
            if self.UserID in (seen_data['D.PersonID'].values):
                return (np.unique(seen_data.loc[seen_data['D.PersonID']==self.UserID,'nb_visits'])+1)
            else:
                return 1 
        nb_visit=list()
        for j in (novelty,returning,returning1):
            nb_visit.append(j(top_visit(self.UserID)))
        return nb_visit
    
    def get_ticket(self):
        def top_ticket(UserID):
            # define the buying level of the user
            if self.UserID in (seen_data['D.PersonID'].values):
                return (np.unique(seen_data.loc[seen_data['D.PersonID']==self.UserID,'avg_ticketU']))
            else:
                return 0 
        ticket_level=list()
        for k in (level1,level2,level3,level4):
            ticket_level.append(k(top_ticket(self.UserID))) 
        return ticket_level
    
    def get_user_profile(self):
        a = self.get_nbpartner() + self.get_visit() + self.get_ticket()
        return a
    
    def get_timedelta(self,timenow):
        self.timenow = timenow
        time_delta=(self.timenow-self.CreationDatetime).total_seconds()/60
        return time_delta
    
    def add_Event(self, Parameter):
        self.ParameterList.append(Parameter)
        return self.ParameterList

    def Event2vec(self):
        click_event=list()
        for i in range(len(np.unique(self.ParameterList))): 
            click_event.append(np.unique(self.ParameterList)[i])
        #return click_event
        event_detection=pd.DataFrame(np.zeros((0,len(It_event_weighted.index))),columns=It_event_weighted.index) 
        val1=1
        val2=0
        event_vector=list()
        for i in range(len(event_detection.columns)):
            if event_detection.columns[i] in click_event:
                event_vector.append(val1)
            else:
                event_vector.append(val2)
        return event_vector 
    
#%%


#%%
def testtime(epoch):# CreationDateTime changed to ' epoch' in C# code
	t = pd.to_datetime(epoch, unit='s')# to convert epoch type to date type
	return t
#%%

#%%
def Context2Item(Item_context_pred):
    time_to_context = max(Item_context_pred.loc[pd.to_datetime(Item_context_pred['Time'])<datetime.datetime.now()+datetime.timedelta(hours=2),:]['Time']) 
    context2item = Item_context_pred.loc[Item_context_pred['Time']==time_to_context,M_prob.columns].values 
    return context2item
#%%

#%%
def User2Item(user_profile):
    user2item=np.array([cosine_similarity(user_profile,y) for y in IT_PROFILE.values]).reshape(len(np.array([user_profile])),
                                IT_PROFILE.shape[0]) 
    user2item=pd.DataFrame(user2item,columns=IT_PROFILE.index)
    return user2item
#%%

#%%
def Event2Item(event_vector):
    event2item=[cosine_similarity(event_vector,y)  for y in It_event_weighted.T.values] 
    event2item=np.array(event2item).reshape(1,It_event_weighted.shape[1]) 
    event2item= pd.DataFrame( event2item, columns=It_event_weighted.columns)
    return event2item
#%%

#%%
train_visit_duration = np.array(visit_duration_stat.loc[:,['mean_visit_duration','std_visit_duration']]).tolist() 
#%%

def Mah(time_delta):
    mah=[mahalanobis_similarity(time_delta,y,z) for y,z in train_visit_duration] 
    minmah= np.min([i for i in mah if i >0])
    mah=[ i if i >0 else minmah for i in mah]
    mah=np.array(mah).reshape(1,visit_duration_stat.shape[0])
    mah=pd.DataFrame(0.1/mah,columns= visit_duration_stat.index.values)
    return mah
#%%

#%%
W1,W2,W3=best_w
#%%

#%%
class Recommendation:
    '''
    Get raw inputs and transform these ones into the features which will be feed to the model
    '''
    def __init__(self,Item_context_pred,user_profile,event_vector,time_delta,W1,W2,W3):
        self.Item_context_pred = Item_context_pred
        self.user_profile = user_profile
        self.event_vector = event_vector
        self.time_delta = time_delta
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
    
    def get_prediction(self):
        ''' get the global prediction
        '''
        Rating=100*(self.W1*Context2Item(self.Item_context_pred) + self.W2*User2Item(self.user_profile) + self.W2*Event2Item(self.event_vector) + self.W3*Mah(self.time_delta) ) # Compute the matrix of ratings
        Ranking=Rating.T 
        Ranking.rename(columns={0:'score'},inplace=True)
        Ranking=Ranking.sort_values(by=['score'],ascending=False) # ranking by descending order
        Ranking= Ranking.to_dict('index')
        keys=list(Ranking.keys())
        values=[Ranking[k]['score'] for k in list(Ranking.keys())]
        dict_ranking={k: v for k, v in zip(keys, values)}
        return dict_ranking
    
    def get_topNprediction(self, top):
        '''
        get top N of Items based on the global ranking
        '''
        self.top = top
        top_N_ranking = self.get_prediction().index.values
        print('Here is the top %d of Items to recommend to the User:' %self.top)
        for i in top_N_ranking[:self.top]:
            print(i, end = ' *** ')
    
    def get_topNseqprediction(self,top):
        '''
        get top N of Items by filtering sold items
        '''
        self.top = top
        ordered_item=list()
        for i in OrderHeader_ProductID:
            ordered_item.append(i)
            ordered_WO_type=np.ravel(products.loc[products['ProductName'].isin(ordered_item),['WorkingOrderType']].values).tolist()
            print('Following the last choice of the user here is the top %d of Items to recommend :' %top)
        for i in self.get_prediction().index.values[pd.Series(self.get_prediction().index.values).isin (products.loc[~products['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])][:top]:
            print(i, end = ' *** ')
    
    def plot_topNseqprediction(self,top):
        self.top = top
        top_n_ranking = self.get_prediction()
        top_n_ranking1=top_n_ranking[pd.Series(top_n_ranking).isin (p.loc[~products['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])][:top]
        return sns.barplot(y=k(r.loc[r['ProductName'].isin (top_n_ranking1)])['ProductName'][:top],x=k(r.loc[r['ProductName'].isin (top_n_ranking1)]['score'][:top])),sns.plt.suptitle('Top following %d items to recommend with scores after the last choice of the user' %top);
    
    def get_fltrprediction(self,user_age):
        ''' 
        do filtering to exclude items not allowed to sell to child
        '''
        self.user_age = user_age
        self.alcohol = alcohol
        if self.user_age >18:
            return self.get_prediction()
        elif np.isnan(self.user_age):
            return self.get_prediction()
        else:
            d=self.get_prediction().loc[~self.get_prediction().index.isin(alcohol)] 
            return d
    def plot_topNprediction(self,top):
        self.top=top
        top_ranking = self.get_prediction()
        return sns.barplot(y=top_ranking.index[:self.top],x=top_ranking['score'][:self.top]),sns.plt.suptitle('Top %d items to recommend with scores' %self.top);
    
    def get_predictionGroup(self):
        '''
        Make recommendation by each product group
        '''
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) 
        Ranking_final_PGroup=list() 
        for i in np.unique(Ranking_final['ProductGroupID']):
            Ranking_final_PGroup.append(Ranking_final.set_index(['ProductGroupID','ProductGroupName']).loc[i]) 
        for i in range(len(Ranking_final_PGroup)):
            print((Ranking_final_PGroup[i]).loc[:,['ProductName','score']]) 
        #recommend top 3 items for each product group
        for i in range(len(Ranking_final_PGroup)):
            top_group=3
            print(Ranking_final_PGroup[i]['ProductName'][:top_group])
        
    def plot_scores_his(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) 
        f, ax = plt.subplots(figsize=(8,6))
        sns.distplot([value for value in Ranking_final['score'] if not math.isnan(value)])
        plt.xlabel('score')
        plt.show()
     
    def plot_scoresWO(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) 
        f, ax = plt.subplots(figsize=(12,8))
        fig = sns.boxplot(y=Ranking_final['ProductGroupName'], x=Ranking_final['score']) 
        return fig
     
    def get_predictionWO(self):
        '''
        Make recommendation by Working_Order type
        '''
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) 
        Ranking_final_WOtype=list() 
        for i in np.unique(Ranking_final['WorkingOrder']):
            Ranking_final_WOtype.append(Ranking_final.set_index(['WorkingOrder','WorkingOrderType']).loc[i])
        for i in range(len(Ranking_final_WOtype)): 
            print((Ranking_final_WOtype[i]).loc[:,['ProductName','score']])
        
    def get_optprediction(self):
        '''
         Product price based recommendation optimization
        '''
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) 
        Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>6)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>4)),:])
        Ranking_final1.drop_duplicates(['ProductName'], keep='first',inplace=True)
        Ranking_final1.index=range(len(Ranking_final1))
        Ranking_final1.index
        Ranking_final1['score']=Ranking_final1['score'].round(3)
        Ranking_final1['optimized_score']=(Ranking_final1[['NetPrice']].values+Ranking_final1[['score']].values)
        Ranking_opt=Ranking_final1[['ProductGroupName','WorkingOrderType','ProductName','NetPrice','score','optimized_score']].sort_values(ascending=False,by=['optimized_score'])
        Ranking_opt.index=range(len(Ranking_opt))
        print('Price aware recommendation by WorkingOrderType')
        for i in (np.unique(Ranking_opt[['WorkingOrderType']])):
             print(Ranking_opt.loc[Ranking_opt['WorkingOrderType']==i])
        #Recommendation by taking in account the price of item and group by ProductGroup
        print('Price aware recommendation by ProductGroupName')
        for i in (np.unique(Ranking_opt[['ProductGroupName']])):
            print(Ranking_opt.loc[Ranking_opt['ProductGroupName']==i])
        # Create a data set containing recommendation by WorkingOrderType&ProductGroupName
            print('Price aware recommendation by WorkingOrderType&ProductGroupName')
            print(Ranking_opt.set_index(['WorkingOrderType','ProductGroupName']).sort_index())
     
    def plot_optprediction(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) 
        Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>6)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>4)),:])
        Ranking_final1.drop_duplicates(['ProductName'], keep='first',inplace=True)
        Ranking_final1.index=range(len(Ranking_final1))
        fig, ax = plt.subplots()
        print(Ranking_final1.plot.scatter(x='score',y='NetPrice',xlim=(3,17),ylim=(0,22),
                                          s=Ranking_final['score'].values*20,c=Ranking_final1['score'],
                                          cmap="coolwarm",figsize=(20,14),ax=ax));
        for i, txt in enumerate(Ranking_final1['ProductName']):
            ax.annotate(txt, (Ranking_final1['score'][i],Ranking_final1['NetPrice'][i]),
                        textcoords='data', ha='left', va='bottom',rotation=30,
                        bbox=dict(boxstyle='round,pad=0.05', fc='yellow', alpha=0.4),
                        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'));	
        
    def plot_optpredictionWO(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) 
        Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>6)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>4)),:])
        Ranking_final1.drop_duplicates(['ProductName'], keep='first',inplace=True)
        Ranking_final1.index=range(len(Ranking_final1))
        # plot all selected items by WorkingOrederType
        g = sns.FacetGrid(Ranking_final1, col="WorkingOrderType", hue="ProductName")
        print(g.map(plt.scatter, "score", "NetPrice", alpha=.9));
        print(g.add_legend());     

#%%


























