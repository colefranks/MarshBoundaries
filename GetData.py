# Package ID: knb-lter-gce.657.6 Cataloging System:https://pasta.edirepository.org.
# Data set title: Cross-site comparison of historical trends in marsh change at three LTER sites: GCE, VCR, and PIE..
# Data set creator:    - Georgia Coastal Ecosystems LTER Project 
# Data set creator:  Christine Burns - University of Georgia 
# Metadata Provider:    -  
# Contact:    - GCE-LTER Information Manager   - gcelter@uga.edu
# Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu      
# 
# This program creates numbered PANDA dataframes named dt1,dt2,dt3...,
# one for each data table in the dataset. It also provides some basic
# summaries of their contents. NumPy and Pandas modules need to be installed
# for the program to run. 

import numpy as np
import pandas as pd 

infile1  ="https://pasta.lternet.edu/package/data/eml/knb-lter-gce/657/6/7b5930e4becb0827753600e5196d6284".strip() 
infile1  = infile1.replace("https://","http://")
                 
dt1 =pd.read_csv(infile1 
          ,skiprows=5
            ,sep=","  
                ,quotechar='"' 
           , names=[
                    "Site",     
                    "Transect_ID",     
                    "Start_X",     
                    "Start_Y",     
                    "End_X",     
                    "End_Y",     
                    "Early_Date",     
                    "Last_Date",     
                    "EPR",     
                    "EPR_Error",     
                    "LRR_slope",     
                    "LRR_Rsquared",     
                    "Early_to_Middle",     
                    "Early_to_End",     
                    "Middle_to_End",     
                    "Channel_Order"    ]
# data type checking is commented out because it may cause data
# loads to fail if the data contains inconsistent values. Uncomment 
# the following lines to enable data type checking
         
#            ,dtype={  
#             'Site':'str' ,  
#             'Transect_ID':'str' , 
#             'Start_X':'float' , 
#             'Start_Y':'float' , 
#             'End_X':'float' , 
#             'End_Y':'float' , 
#             'Early_Date':'int' , 
#             'Last_Date':'int' , 
#             'EPR':'float' , 
#             'EPR_Error':'float' , 
#             'LRR_slope':'float' , 
#             'LRR_Rsquared':'float' , 
#             'Early_to_Middle':'float' , 
#             'Early_to_End':'float' , 
#             'Middle_to_End':'float' ,  
#             'Channel_Order':'str'  
#        }
            ,na_values={
                  'Transect_ID':[
                          'NaN',],
                  'Start_X':[
                          'NaN',],
                  'Start_Y':[
                          'NaN',],
                  'End_X':[
                          'NaN',],
                  'End_Y':[
                          'NaN',],
                  'Early_Date':[
                          'NaN',],
                  'Last_Date':[
                          'NaN',],
                  'EPR':[
                          'NaN',],
                  'EPR_Error':[
                          'NaN',],
                  'LRR_slope':[
                          'NaN',],
                  'LRR_Rsquared':[
                          'NaN',],
                  'Early_to_Middle':[
                          'NaN',],
                  'Early_to_End':[
                          'NaN',],
                  'Middle_to_End':[
                          'NaN',],
                  'Channel_Order':[
                          'NaN',],} 
            
    )
# Coerce the data into the types specified in the metadata  
dt1.Site=dt1.Site.astype('category')  
dt1.Transect_ID=dt1.Transect_ID.astype('category') 
dt1.Start_X=pd.to_numeric(dt1.Start_X,errors='coerce') 
dt1.Start_Y=pd.to_numeric(dt1.Start_Y,errors='coerce') 
dt1.End_X=pd.to_numeric(dt1.End_X,errors='coerce') 
dt1.End_Y=pd.to_numeric(dt1.End_Y,errors='coerce') 
dt1.Early_Date=pd.to_numeric(dt1.Early_Date,errors='coerce',downcast='integer') 
dt1.Last_Date=pd.to_numeric(dt1.Last_Date,errors='coerce',downcast='integer') 
dt1.EPR=pd.to_numeric(dt1.EPR,errors='coerce') 
dt1.EPR_Error=pd.to_numeric(dt1.EPR_Error,errors='coerce') 
dt1.LRR_slope=pd.to_numeric(dt1.LRR_slope,errors='coerce') 
dt1.LRR_Rsquared=pd.to_numeric(dt1.LRR_Rsquared,errors='coerce') 
dt1.Early_to_Middle=pd.to_numeric(dt1.Early_to_Middle,errors='coerce') 
dt1.Early_to_End=pd.to_numeric(dt1.Early_to_End,errors='coerce') 
dt1.Middle_to_End=pd.to_numeric(dt1.Middle_to_End,errors='coerce')  
dt1.Channel_Order=dt1.Channel_Order.astype('category') 
      
print("Here is a description of the data frame dt1 and number of lines\n")
print(dt1.info())
print("--------------------\n\n")                
print("Here is a summary of numerical variables in the data frame dt1\n")
print(dt1.describe())
print("--------------------\n\n")                
                         
print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")                 

print(dt1.Site.describe())               
print("--------------------\n\n")
                    
print(dt1.Transect_ID.describe())               
print("--------------------\n\n")
                    
print(dt1.Start_X.describe())               
print("--------------------\n\n")
                    
print(dt1.Start_Y.describe())               
print("--------------------\n\n")
                    
print(dt1.End_X.describe())               
print("--------------------\n\n")
                    
print(dt1.End_Y.describe())               
print("--------------------\n\n")
                    
print(dt1.Early_Date.describe())               
print("--------------------\n\n")
                    
print(dt1.Last_Date.describe())               
print("--------------------\n\n")
                    
print(dt1.EPR.describe())               
print("--------------------\n\n")
                    
print(dt1.EPR_Error.describe())               
print("--------------------\n\n")
                    
print(dt1.LRR_slope.describe())               
print("--------------------\n\n")
                    
print(dt1.LRR_Rsquared.describe())               
print("--------------------\n\n")
                    
print(dt1.Early_to_Middle.describe())               
print("--------------------\n\n")
                    
print(dt1.Early_to_End.describe())               
print("--------------------\n\n")
                    
print(dt1.Middle_to_End.describe())               
print("--------------------\n\n")
                    
print(dt1.Channel_Order.describe())               
print("--------------------\n\n")
                    
                    
                
def get_data():
  return dt1



