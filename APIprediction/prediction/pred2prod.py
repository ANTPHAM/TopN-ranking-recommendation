# -*- coding: utf-8 -*-
"""
Title: "Recommender System by Top_N Ranking_ Model in API "

Authors : Antoine.P

Date: September 2018
"""

#%%
import os
rootPath=os.path.join(os.getcwd()) + "\\..\\"
paramsPath=rootPath + "params\\"
#%%

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
import re
from math_utils2prod import *

#%%
#==============================================================================
#                 IMPORT MATRIX CONTAINING PARAMETERS FOUNDED BY THE TRAIN MODEL  
#==============================================================================
item_profile=pd.read_csv(paramsPath + 'item_profile.csv',sep=',')
visit_duration_stat=pd.read_csv(paramsPath + 'visit_duration_stat.csv',sep=',')
#doing some data preparations 
item_profile=item_profile.set_index(['Unnamed: 0'])
item_profile.index.name=None
visit_duration_stat=visit_duration_stat.set_index(['Unnamed: 0'])
visit_duration_stat.index.name=None
#%%

#%%
item_context_prob=pd.read_csv(paramsPath + 'item_context_prob.csv', sep=',',header=None,low_memory=False)
item_context_prob.columns=item_context_prob.iloc[0]
item_context_prob = item_context_prob.reindex(item_context_prob.index.drop(0)).reset_index(drop=True)
item_context_prob = item_context_prob.set_index(item_context_prob.iloc[:,0])
item_context_prob.index.name=None
item_context_prob=item_context_prob.iloc[:,1:]
item_context_prob=item_context_prob.astype('float')
item_event_weighted=pd.read_csv(paramsPath + 'item_event_weighted.csv', sep=',',header=None,low_memory=False)
item_event_weighted.columns = item_event_weighted.iloc[0]
item_event_weighted = item_event_weighted.reindex(item_event_weighted.index.drop(0)).reset_index(drop=True)
item_event_weighted = item_event_weighted.set_index(item_event_weighted.iloc[:,0])
item_event_weighted.index.name=None
item_event_weighted=item_event_weighted.iloc[:,1:]
item_event_weighted=item_event_weighted.astype('float')
#%%

#%%
seen_data=pd.read_csv(paramsPath + 'seen_data.csv',sep=',')
#%%

#%%
best_w=[0.8389266,0.15078162,0.01029177] #import the weights founded in training tasks
W1,W2,W3=best_w
#%%

#%%
print(" Parameters of the model loaded!")
#%%

#%%
item_context_pred=pd.read_csv(paramsPath + 'item_context_pred.csv', sep=',')
print(" The matrix 'item_context_pred' dumped! " )
#%%

#%%
Products=pd.read_csv(paramsPath+"products.csv",sep=',') #Import matrix describing items&product group
Products.columns=['ID','ProductID','ProductName','ProductGroupID','ProductGroupName','GrossPrice','NetPrice','GuidedActivityID','WorkingOrder']
Working_Order_type={0:'Menu', 1:'Beverage', 2:'Starter', 3 :'MainDishes',
                    4 :'Dessert', 5:'Coffee', 6:'Digestive', 7:'Others'}   # create a dictionary that stoke WorkingOrderType
Products['WorkingOrderType']=Products['WorkingOrder']# create a column containing numeric values of the column "WorkingOrder"
Products['WorkingOrderType'].replace(Working_Order_type, inplace=True)# replace these numeric values by text values
#%%



#%%
alcohol=['1664','25CL ROSE COTE PROVENCE CAP DES PINS', '25CL ROUGE COTE DU RHONE VINSOBRAISE', '37,5CL VIN BLANC ARBOIS BETHANIE', '37,5CL VIN ROSE ARBOIS POULSARD',
          '37,5CL VIN ROSE BARDOLINO', '37,5CL VIN ROUGE CHIANTI', '46CL ROSE COTE PROVENCE CAP DES PINS', '46CL ROUGE COTE DU RHONE VINSOBRAISE', '75 cl - Bourgogne',
           '75CL CREMANT JURA ARBOIS', '75CL VIN BLANC ARBOIS BETHANIE', '75CL VIN BLANC JURA TRADITION', '75CL VIN ROSE ARBOIS POULSARD', '75CL VIN ROSE BARDOLINO',
            '75CL VIN ROSE ST TROPEZ GOURM', '75CL VIN ROUGE CHIANTI', '75CL VIN ROUGE PETIL LAMBRUSCO', '75CL VIN ROUGE RASTEAU', "BAILEY'S (4cl)_201709",
            'BIERE DE SAISON (25cl)_201709', 'BIERE DE SAISON (50cl)_201709', 'CAFE + 1 GOURMANDISE_201709', 'CAFE LIEGEOIS_201709', "COCKTAIL PIZZ'UP_201709",
             'COGNAC (4cl)_201709', 'COMETE (25cl)_201709', 'COMETE (50cl)_201709', 'COMPOTE DE POMMES_201709', 'DECAFEINE_201709', 'DESPERADOS (33cl)_201709',
              'DOUBLE EXPRESSO_201709', 'EXPRESSO_201709', "FEU D'ARTIFICE SANS ALCOOL_201709", 'GET 27 (4cl)_201709', 'HOEGAARDEN BLANCHE (33cl)_201709',
               'IRISH COFFEE_201709', "JACK DANIEL'S_201709", "KIZZ PIZZ'UP VIN BLANC_201709", 'KIZZ RETRO_201709', 'LIMONCELLO (4cl)_201709',
                'MACVIN DU JURA_201709', 'MARC DU JURA (4cl)_201709', 'MARTINI BLANC_201709', 'MARTINI GIN', 'MARTINI ROUGE_201709', 'MAURESQUE',
                 'MENU MIDI_201709', 'MOJITO', 'MOJITO + PLANCHE APERITIVE', 'MUSCAT', 'PANACHE (25cl)_201709', 'PANACHE (50cl)_201709',
                  'PERROQUET', 'PICHET DE MOJITO', 'PICHET DE SPRITZ', 'PINA COLADA_201709', 'POIRE (4cl)_201709', 'PONTARLIER_201709', 'PORTO BLANC_201709',
                   'PORTO ROUGE_201709', 'RHUM ANANAS CITRON VERT', 'RHUM CB', 'RHUM MANGUE ABRICOT', 'RHUM Orange Passion', 'RHUM Vanille Chocolat',
                    'RICARD_201709', 'ROSE ITALIEN 15cl', 'ROSE ITALIEN 25cl', 'ROSE ITALIEN 46cl', 'Rhum-Raisin au rhum des Antilles françaises_201709',
                    'Rosé - Côte de Provence "Cap des Pins" 15CL', 'Rosé de Californie "Gallo" - 25cl_201709', 'Rosé de Californie "Gallo" - 46cl_201709',
                     'Rosé de Californie "Gallo"-15cl_201709', 'Rouge - Côte du Rhône "Vinsobraise"15cl_201709', 'SANGRIA_201709', 'SPAGHETTI JAMBON_201709',
                      'SUZE', 'TEQUILA SUNRISE_201709', 'TOMATE', "Verre Crémant du Jura brut fruitère d'Arbois - Coupe 12cl_201709", 'Verre Rosé - Côte de Provence "Cap des Pins"',
                       'Verre Rosé de Californie "Gallo"_201709', 'Verre Rouge - Côte du Rhône "Vinsobraise"', 'Verre vin blanc IGP', 'Verre vin rosé IGP', 'Verre vin rouge IGP',
                        'Vin blanc IGP', 'Vin blanc IGP - 25cl_201709', 'Vin blanc IGP - 46cl_201709', 'Vin rosé IGP', 'Vin rosé IGP - 25cl_201709', 'Vin rosé IGP - 46cl_201709',
                         'Vin rouge IGP - 25cl_201709', 'Vin rouge IGP - 46cl_201709', 'Vin rouge IGP_201709', 'WILLIAM LAWSON SCOTCH (NATURE)_201709', 'WILLIAM LAWSON SCOTCH + SODA_201709',
                         'MONACO (50cl)','MONACO (25cl)','Heineken 33cl','75CL ROSE COTE PVCE EMP']
#%%

#%%
#==============================================================================
#                                DATA PREPROCESSING
#==============================================================================
# create a list containing all of these names
oldnames=item_event_weighted.index[item_event_weighted.index.str.contains('_event')]
# remove the string '_event' and create a new list of name
newnames=[oldnames[i][:-6] for i in range(len(oldnames))]
# rename these click events by the list 'newnames'
for i in range(len(item_event_weighted.index)):
    if item_event_weighted.index.values[i] in (oldnames):
        item_event_weighted.index.values[i]= item_event_weighted.index.values[i][:-6]
    else:
        item_event_weighted.index.values[i]= item_event_weighted.index.values[i]
#%%
#%%
print(" Data preprocessing achieved!")
#%%

#%%
#===================================================================================================
#                   CREATING A CLASS TO GET INPUT DATA AND DO FEATURE ENGINEERING TASKS
#===================================================================================================
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
    
    def get_timedelta(self):
        self.timenow = datetime.datetime.now()
        time_delta=(self.timenow-self.CreationDatetime).total_seconds()/60
        return time_delta
    
    def add_Event(self, Parameter):
		'''
		Parameter= an event(a click)
		'''
        self.ParameterList.append(Parameter)
        return self.ParameterList

    def Event2vec(self):
        click_event=list()
        for i in range(len(np.unique(self.ParameterList))): 
            click_event.append(np.unique(self.ParameterList)[i])
        event_detection=pd.DataFrame(np.zeros((0,len(item_event_weighted.index))),columns=item_event_weighted.index) 
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
def testtime(epoch):
	t = pd.to_datetime(epoch, unit='s')# to convert epoch type to date type
	return t
#%%

#%%
def Context2Item(Item_context_pred):
    time_to_context = max(Item_context_pred.loc[pd.to_datetime(Item_context_pred['Time'])<datetime.datetime.now()+datetime.timedelta(hours=2),:]['Time']) 
    context2item = Item_context_pred.loc[Item_context_pred['Time']==time_to_context,item_context_prob.columns].values 
    return context2item
#%%

#%%
def User2Item(user_profile):
    user2item=np.array([cosine_similarity(user_profile,y) for y in item_profile.values]).reshape(len(np.array([user_profile])),
                                item_profile.shape[0]) 
    user2item=pd.DataFrame(user2item,columns=item_profile.index)
    return user2item
#%%

#%%

def Event2Item(event_vector):
    event2item=[cosine_similarity(event_vector,y)  for y in item_event_weighted.T.values] # predict buying probability in according to captured click events
    event2item=np.array(event2item).reshape(1,item_event_weighted.shape[1]) #create a data frame containing all of  items with the corresponded buying probability
    event2item= pd.DataFrame( event2item, columns=item_event_weighted.columns)
    return event2item
#%%

#%%
train_visit_duration = np.array(visit_duration_stat.loc[:,['mean_visit_duration','std_visit_duration']]).tolist() # from the data set 'visit_duration', take  the mean and the standart deviation
#%%

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
class Recommendation():
    
    '''
    Get raw inputs and do feature engineering so that these ones can be fed to the model
    '''
    
    def __init__(self,Item_context_pred,Products,user_profile,event_vector,time_delta,W1,W2,W3, user_age=None, top= None):
        self.Item_context_pred = Item_context_pred
        self.Products=Products
        self.user_profile = user_profile
        self.event_vector = event_vector
        self.time_delta = time_delta
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.OrderedItem=list()
        self.user_age=user_age
        self.top = top
           
    def add_ordereditem(self,item): 
        '''
         adding ordered item ; item= 'ID'
    	'''
        self.OrderedItem.append(item)
        return self.OrderedItem

    def get_prediction(self,out_format = None):
        
        ''' 
        Making global predictions
        
        '''		
        rating=100*(self.W1*Context2Item(self.Item_context_pred) + self.W2*User2Item(self.user_profile) + self.W2*Event2Item(self.event_vector) + self.W3*Mah(self.time_delta)) # Compute the matrix of ratings
        ranking=rating.T 
        ranking.rename(columns={0:'score'},inplace=True)
        ranking=ranking.sort_values(by=['score'],ascending=False) 
        ranking_by_id=ranking 
        newnames=[int(re.search(r'\d+$',ranking_by_id.index[i]).group()) for i in range(len(ranking_by_id.index))]# using regex to extract all Item ID ( as numeric value) 
        ranking_by_id.index=newnames
        ranking_by_id=ranking_by_id.groupby(by=ranking_by_id.index, axis=0).mean()
        ranking_by_id=ranking_by_id.sort_values(by=['score'], ascending = False)
        item_list_reduced=Products 
        item_list_reduced.drop_duplicates(['ProductName','ID'], keep='first',inplace=True)
        names=[item_list_reduced.loc[item_list_reduced['ID']==id,['ProductName']].values[0][0] for id in ranking_by_id.index]
        wo=[item_list_reduced.loc[item_list_reduced['ID']==id,['WorkingOrder']].values[0][0] for id in ranking_by_id.index]
        wot= [item_list_reduced.loc[item_list_reduced['ID']==id,['WorkingOrderType']].values[0][0] for id in ranking_by_id.index]
        ranking_by_id.insert(0, 'ProductName',names )
        ranking_by_id.insert(1, 'WorkingOrder',wo)
        ranking_by_id.insert(2, 'WorkingOrderType',wot)
        ranking_by_id=ranking_by_id.reset_index()
        ranking_by_id=ranking_by_id.rename(columns={'index':'ID'})
        print('')
        '''
        Getting top N of Items by filtering all items having the same WorkingOrder as the sold one
        '''
        
        if len(self.OrderedItem ):
            pass
        else:
            ordered_item=list()
            for i in self.OrderedItem:
                ordered_item.append(i)
            ordered_WO_type=np.ravel(Products.loc[Products['ID'].isin(ordered_item),['WorkingOrder']].values).tolist()
            it=Products.loc[~Products['WorkingOrder'].isin (ordered_WO_type)]['ID']
            ranking_by_id=ranking_by_id.loc[ranking_by_id['ID'].isin(it),:]
            
        '''
		    Output by Top N Items and/ or User age
		'''
        if self.top is None:
    	# get recommended Items as the global ranking
            if self.user_age is None:
			# no discarding No Children Allowed Items
                pass
            elif np.isnan(self.user_age):
                pass
            elif self.user_age >18:
                pass
            else:
			# discarding No Children Allowed Items
                ranking_by_id=ranking_by_id.loc[~ranking_by_id['ProductName'].isin(alcohol)] # discarding products founded in the alcol list 
        elif np.isnan(self.top):
    	# get recommended Items as the global ranking
            if self.user_age is None:
			# no discarding No Children Allowed Items
                pass
            elif np.isnan(self.user_age):
                pass
            elif self.user_age >18:
                pass
            else:
			# discarding No Children Allowed Items
                ranking_by_id=ranking_by_id.loc[~ranking_by_id['ProductName'].isin(alcohol)] # discarding products founded in the alcohol list 
        
        else:
		# get top N of Items based on the global ranking
            if self.user_age is None:
			# without discarding No Children Allowed Items
                ranking_by_id=ranking_by_id.iloc[:self.top,:]
            elif np.isnan(self.user_age):
                ranking_by_id=ranking_by_id.iloc[:self.top,:]
            elif self.user_age >18:
                ranking_by_id=ranking_by_id.iloc[:self.top,:]
            else:
			# discarding No Children Allowed Items
                ranking_by_id=ranking_by_id.loc[~ranking_by_id['ProductName'].isin(alcohol)]
                ranking_by_id=ranking_by_id.iloc[:self.top,:]
        
        return ranking_by_id if out_format is None else ranking_by_id[['ID','score', 'ProductName']].T.to_dict()
        
    def get_PredbyGroup(self, out_format = None, top_group= None):

        '''
         Make outputs following Product Groups
       
         params: 'out_format', ' top_group'
        
         out_format = 'None' (default) :  output as a list containing multiple data frames
        
         out_format = 'dict': output as a list containing multiple dictionaries
         
         top_group: the number of items that will be displayed by prediction
        '''
        r=self.get_prediction().reset_index()
        r=r[['ProductName','score','ID']]
        Ranking_final=pd.merge(Products,r,on=('ProductName','ID'),how='left').sort_values('score',ascending=False) 
        print(Ranking_final.head(2))
               
        Ranking_PGroup=list()
        for i in np.unique(Ranking_final['ProductGroupID']):
            if np.isnan(i):
                pass
            else:
                if top_group is None:
                    Ranking_PGroup.append(Ranking_final.set_index(['ProductGroupID','ProductGroupName']).loc[i].loc[:,['ProductID','ProductName','WorkingOrder','WorkingOrderType','score']])
                else:
                    Ranking_PGroup.append(Ranking_final.set_index(['ProductGroupID','ProductGroupName']).loc[i].loc[:,['ProductID','ProductName','WorkingOrder','WorkingOrderType','score']][:top_group])
        print(Ranking_PGroup[:2])
        Ranking_PGroupNan=Ranking_final.loc[np.isnan(Ranking_final['ProductGroupID']),['ProductID','ProductName','WorkingOrder','WorkingOrderType','score']]
        if Ranking_PGroupNan.empty:
            pass
        else:
            Ranking_PGroup.append(Ranking_PGroupNan)
        print(Ranking_PGroupNan)
        Ranking_PGroup1=list()
        if out_format =='dict':
            for i in range(len(Ranking_PGroup)):
                R=Ranking_PGroup[i].reset_index()
                R=R.to_dict()
                Ranking_PGroup1.append(R)
        else:
            for i in range(len(Ranking_PGroup)):
                Ranking_PGroup1.append(Ranking_PGroup)
        
        return Ranking_PGroup1
    
        
    def get_PredbyWO(self, out_format = None, top_group= None):
       
        '''
        Make outputs following Product Groups
        params: 'out_format', ' top_group'
        out_format = 'None' (default) :  output as a list containing multiple data frames
        out_format = 'dict': output as a list containing multiple dictionaries
        top_group: the number of items that will be displayed by prediction
        '''
        r=self.get_prediction().reset_index()
        r=r[['ProductName','score','ID']]
        Ranking_final=pd.merge(Products,r,on=('ProductName','ID'),how='left').sort_values('score',ascending=False)
        Ranking_PGroup=list()
        for i in np.unique(Ranking_final['WorkingOrder']):
            if np.isnan(i):
                pass
            else:
                if top_group is None:
                    Ranking_PGroup.append(Ranking_final.set_index(['WorkingOrder','WorkingOrderType']).loc[i].loc[:,['ProductID','ProductName','ProductGroupID','ProductGroupName','score']])
                else:
                    Ranking_PGroup.append(Ranking_final.set_index(['WorkingOrder','WorkingOrderType']).loc[i].loc[:,['ProductID','ProductName','ProductGroupID','ProductGroupName','score']][:top_group])
        
        Ranking_PGroupNan=Ranking_final.loc[np.isnan(Ranking_final['WorkingOrder']),['ProductID','ProductName','ProductGroupID','ProductGroupName','score']]
        if Ranking_PGroupNan.empty:
            pass
        else:
            Ranking_PGroup.append(Ranking_PGroupNan)
        
        Ranking_PGroup1=list()
        if out_format =='dict':
            for i in range(len(Ranking_PGroup)):
                R=Ranking_PGroup[i].reset_index()
                R=R.to_dict()
                Ranking_PGroup1.append(R)
        else:
            Ranking_PGroup1.append(Ranking_PGroup)
        
        return Ranking_PGroup1
    
    
    def get_optprediction(self, predtype=None):
        
        self.predtype = predtype
        '''
        Products' price based recommendation optimization

		Output 

		'''
        r=self.get_prediction().reset_index()
        r=r[['ProductName','score','ID']]
        Ranking_final=pd.merge(Products,r,on=('ProductName','ID'),how='left').sort_values('score',ascending=False) 
        Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>9.0)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>10.0)),:]) # try to define a cutoff and plot items with their scores and  net prices   
        Ranking_final1.drop_duplicates(['ProductName'], keep='first',inplace=True)
        Ranking_final1.index=range(len(Ranking_final1))
        Ranking_final1['score']=Ranking_final1['score'].round(4)
        Ranking_final1['optimized_score']=(Ranking_final1[['NetPrice']].values+Ranking_final1[['score']].values)
        print('OPTIMIZATION')
        print(Ranking_final1)
        Ranking_opt=Ranking_final1[['ProductGroupName','WorkingOrderType','ProductName','ID','NetPrice','score','optimized_score']].sort_values(ascending=False,by=['optimized_score'])
        Ranking_opt.index=range(len(Ranking_opt))
        Ranking_optpred1=Ranking_opt[['ID', 'ProductName','optimized_score','score']].T.to_dict()
        Ranking_optpred= list()       
        if predtype is None:
           pass
        elif predtype =='wo':
            print(' ')
            print('PRICE AWARE RECOMMENDATION FOLLOWING "WorkingOrderType"')
            print(' ')
            for i in (np.unique(Ranking_opt[['WorkingOrderType']])):
                Ranking_optpred.append(Ranking_opt.loc[Ranking_opt['WorkingOrderType']==i].loc[:, ['WorkingOrderType','ProductName','ID','NetPrice','score','optimized_score' ]])
        elif predtype =='prdgr' :
			print(' ')
            print('PRICE AWARE RECOMMENDATION FOLLOWING "ProductGroup"')
            print(' ')
            for i in (np.unique(Ranking_opt[['ProductGroupName']])):
                Ranking_optpred.append(Ranking_opt.loc[Ranking_opt['ProductGroupName']==i].loc[:, ['WorkingOrderType','ProductName','ID','NetPrice','score','optimized_score' ]])
        else:
            pass
        return Ranking_optpred1 if predtype is None  else Ranking_optpred
     
    
    def plot_scores_his(self):
        r=self.get_prediction().reset_index()
        r=r[['ProductName','score','ID']]
        Ranking_final=pd.merge(Products,r,on=('ProductName','ID'),how='left').sort_values('score',ascending=False)
        f, ax = plt.subplots(figsize=(8,6))
        sns.distplot([value for value in Ranking_final['score'] if not math.isnan(value)])
        plt.xlabel('score')
        plt.show()
        
        
    def plot_topNprediction(self):
        top_ranking = self.get_prediction()
        return sns.barplot(y=top_ranking['ProductName'][:self.top],x=top_ranking['score'][:self.top]),sns.plt.suptitle('Figure 1: Top %d items to recommend' %self.top)
    
    
    def plot_topNseqprediction(self):
        
        top_n_ranking = self.get_prediction()
        top_n_ranking1=top_n_ranking[pd.Series(top_n_ranking).isin (p.loc[~products['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])][:top]
        return sns.barplot(y=k(r.loc[r['ProductName'].isin (top_n_ranking1)])['ProductName'][:top],x=k(r.loc[r['ProductName'].isin (top_n_ranking1)]['score'][:top])),sns.plt.suptitle('Top following %d items to recommend with scores after the last choice of the user' %top);
    
    
    def plot_scoresWO(self):
        r=self.get_prediction().reset_index()
        r=r[['ProductName','score','ID']]
        Ranking_final=pd.merge(Products,r,on=('ProductName','ID'),how='left').sort_values('score',ascending=False)
        f, ax = plt.subplots(figsize=(12,8))
        fig = sns.boxplot(y=Ranking_final['ProductGroupName'], x=Ranking_final['Score']) 
        return fig
    
    
    def plot_optprediction(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) 
        Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>6)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>4)),:])# let's set a cutoff
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
        g = sns.FacetGrid(Ranking_final1, col="WorkingOrderType", hue="ProductName")
        print(g.map(plt.scatter, "score", "NetPrice", alpha=.9));
        print(g.add_legend());     
#%%
print(" Class already built!")