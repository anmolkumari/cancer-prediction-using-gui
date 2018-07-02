
#packages
import pandas as pd 
import numpy as np 
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from tkinter import *

from functools import partial



#reading and loading
def read_data():
	
	odf=pd.read_csv("breast-cancer-wisconsin.csv")
	return odf
#******************************************************************************************************
def get_headers(dataset):
	return(dataset.columns.values)
#*************************************************************************************************
def add_headers(dataset,headers):
	dataset.columns=headers
	return dataset
#***************************************************************************************************
#converting back to edit changes in actual file
def data_file_to_csv():
	pd.to_csv("breast-cancer-wisconsin.csv")
#*************************************************************************************************
#splitting the data into train and test data
def split_dataset(dataset,train_percentage,feature_headers,target_headers):

	train_x,test_x,train_y,test_y=train_test_split(dataset[feature_headers],dataset[target_headers],train_size=train_percentage)
#takes all index of x in features header and all y and index in target_header
	return train_x,test_x,train_y,test_y
#***************************************************************************************************
#data manipulation and dealing with missing values
def handle_missing_values(dataset,missing_values_header,missing_label):
	t=missing_values_header
	x=(dataset[t]!=missing_label)
	return dataset[x]
#**************************************************************************************************
#classification and predictions using random forest classifier
#returning value achieved by result
def random_forest_classifier(features,targets,find):
	clf=RandomForestClassifier()
	clf.fit(features,targets)
	val1=clf.predict(find)
	return(val1)
#************************************************************************************************************
#classification and predictions linear_regression
def linear_regression(features,targets,find):
	dlf=LinearRegression()
	dlf.fit(features,targets)
	val2=dlf.predict(find)
	return(val2)
#**************************************************************************************************
#classification and predictions using naive_bayes
#returning val3 as result
def naive_bayes(features,targets,find):
	model = GaussianNB()
	model.fit(features, targets)
	val3=model.predict(find)
	return(val3)
#*************************GUI**************************************************************************
def final(arg,features,targets,find):
	switch= {
		1:random_forest_classifier(features,targets,find),
		2:linear_regression(features,targets,find),
		3:naive_bayes(features,targets,find),
	
		}
	return switch.get(arg,"nothing")	


def main():
	dataset=read_data()
	HEADERS=get_headers(dataset)
	dataset=handle_missing_values(dataset,HEADERS[6],'?')
	train_x,test_x,train_y,test_y=split_dataset(dataset,0.7,HEADERS[1:-1],HEADERS[-1])
#*****************************GUI in main************************************************************
#*******************************call result fetches option and calls using final functn*********************
	def call_result(labelResult,option,features,targets,find):
		option=(option.get())
		result=final(option,features,targets,find)
		str1='\n'.join(str(x) for x in result)#prints line by line
		labelResult.config(text='%s'%str1)#for output
		return


		

	root =Tk()
	root.geometry('400x400')#size of gui window screen

	option=IntVar()

	#radio button
	R1=Radiobutton(root,text="RandomForestClassifier",variable=option,value=1).grid(row=4,column=1)
	R2=Radiobutton(root,text="LinearRegression",variable=option,value=2).grid(row=4,column=2)
	R3=Radiobutton(root,text="naive_bayes",variable=option,value=3).grid(row=4,column=3)


	labelTitle=Label(root,text="Predictors").grid(row=0,column=2)

	labelResult= Label(root)
	labelResult.grid(row=8,column=2)
	call_result=partial(call_result,labelResult,option,train_x,train_y,test_x)#calling the call_result
	buttoncall=Button(root,text="Calculate",command=call_result).grid(row=6,column=1)#button for calc result(handler)

	root.mainloop()	
main()

