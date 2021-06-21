


# # why data visulisation ?
# # human brain can easily understand if the data 
# # in pictorial and graphical format
# # ---
# # visulisation helps to convert those data in pictorail and 
# # graphical format
# # ---
# # what is mtplotlib ?
# # matplotlib is represtation of 2D graph
# # ----
# # types of plotes
# # 1-bar graph
# # 2-histogram
# # 3-scatter plot
# # 4-pie chart
# # 5-hexgonal bin plot
# # 6-area plot

# # METHODS 1
# # from matplotlib import pyplot as plt


# # print( plt.style.available )
# # plt.style.use('seaborn-paper')
# # year = [2002,2003,2004,2004,2005]

# # p_india = [0.23,0.78,0.87,0.88,0.97]
# # p_china = [0.99,0.11,0.87,0.11,0.12]

# # plt.plot(year,p_india)
# # plt.plot(year,p_china)



# # plt.xlabel('year')
# # plt.ylabel('population')
# # plt.title('popluation_of_india vs india')
# # plt.legend(['china','india'])
# # plt.grid()
# # plt.tight_layout()
# # plt.show()

# # help(hasattr)


# # from matplotlib import pyplot as plt

# # plt.style.use()
# # year = ('2000','2001','2002','2003','2004','2005',
# # '2006','2007','2008','2009','2010','2011','2012','2013','2014','2015',
# # '2016','2017','2018','2019','2020','2021')




# # popluation_of_india = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21]

# # popluation_of_china = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.12,0.13,0.14,0.15,0.16,0.17,0.99,0.89,0.29,0.21]

# # plt.subplot(1, 2, 1)
# # plt.plot(year,popluation_of_india,popluation_of_china)
# # plt.title('subplot 1')
# # plt.xlabel('popluation_of_india')
# # plt.ylabel('year')
# # plt.legend(['china','india'])

# # plt.subplot(1, 2, 2)
# # plt.plot(year,popluation_of_india,popluation_of_china)
# # plt.title('subplot 2')
# # plt.xlabel('popluation_of_china')
# # plt.ylabel('year')
# # plt.legend(['china','india'])


# # plt.tight_layout()
# # plt.grid(True)
# # plt.show()
                                        

# # from matplotlib import pyplot as plt
# # import numpy as np
# # import pandas as pd

# # # %reload_ext autoreload
# # # %autoreload 2

# # # funtional tool
# # x_1 = np.linspace(0,5,10)
# # y_1 = x_1**2
# # plt.plot(x_1,y_1)
# # plt.title('days squirrel chart')
# # plt.xlabel('days')
# # plt.ylabel('days squirrel')
# # plt.show()


# # plt.subplot(1,2,1)
# # plt.plot(x_1,y_1,'r')


# # plt.subplot(1,2,2)
# # plt.plot(x_1,y_1,'b')
# # plt.show()

# # METHODS 2
# # import matplotlib.pyplot as plt
# # x = [1,2,3]
# # y = [4,5,1]
# # plt.title('farukh_graph')
# # plt.xlabel('x_axis')
# # plt.ylabel('y_axis')
# # plt.plot(x,y)
# # plt.grid()
# # plt.show()

# # import matplotlib
# # import matplotlib .pyplot as plt
# # from matplotlib import style

# # style.use('ggplot')

# # x = [5,8,10]
# # y = [12,6,6]

# # x1 = [6,9,11]
# # y1 = [6,15,7]
# # plt.plot(x,y,'g',label='Line one ',linewidth =5)
# # plt.plot(x1,y1,'c',label='Line Two')
# # plt.xlabel('x axis')
# # plt.ylabel('y axis')
# # plt.title('bar graph')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

# # bar graph

# # plt.bar([1,3,5,7,9],[5,2,7,8,2],label='1 label')
# # plt.bar([2,4,6,8,10],[8,6,2,5,6],label='2 label')

# # plt.legend()
# # plt.xlabel('bar numbers')
# # plt.ylabel('bar height')
# # plt.title('info')
# # plt.show()

# # histogram

# # population_ages = [22,33,44,12,23,45,75,44,11,33,43,23,21,34]

# # bins = [0,10,20,30,40,50,60,70,80,90,100,120,130]

# # plt.hist(population_ages,bins,histtype='bar',rwidth=0.8)
# # plt.xlabel('x axis')
# # plt.ylabel('y axis')
# # plt.title('histogram')
# # plt.show()

# # scatter plot

# # x = [1,2,3,4,5,6,7,8]
# # y = [3,2,4,5,6,7,8,9]

# # plt.scatter(x,y,label='its me here')
# # plt.legend()
# # plt.show()

# # stack plot

# # days = [1,2,3,4,5]

# # sleeping = [7,8,9,8,2]
# # walking = [2,3,4,1,2]
# # running = [3,4,6,7,8]
# # playing = [3,4,5,6,7]

# # plt.plot([],[],colour='m',label='sleeping',linewidth=5)
# # plt.plot([],[],colour='c',label='walking',linewidth=5)
# # plt.plot([],[],colour='r',label='running',linewidth=5)
# # plt.plot([],[],colour='k',label='playing',linewidth=5)

# # plt.stackplot(days,sleeping,walking,running,playing,colors=['m','c','r','k'])
# # plt.xlabel('x axis')
# # plt.ylabel('y axis')
# # plt.show()

# # import matplotlib .pyplot as plt
# # # import matplotlib style

# # # style.use('ggplot')

# # x = [2,3,4]
# # y = [3,4,5]

# # x1 = [3,4,5]
# # y1 = [3,4,5]

# # plt.plot(x,y,'b',label='Line one',linewidth=5)
# # plt.plot(x1,y1,'g',label='Line two',linewidth=5)

# # plt.tight_layout()
# # plt.title('info')
# # plt.xlabel('x axis')
# # plt.ylabel('y axis')

# # plt.grid(True)
# # plt.show()


# # import matplotlib.pyplot as plt
# # import matplotlib style

# # style.use('ggplot')

# # plt.bar([2,3,4,5],[3,4,4,6],label='one line')
# # plt.bar([3,4,5,6],[5,6,7,8],label='two line')

# # plt.title(' bar graph')
# # plt.legend()
# # plt.xlabel('x axis')
# # plt.ylabel('y axis')
# # plt.grid()
# # plt.show()

# # popution_ages= [22,33,44,55,66,77,88,999,22,33,444,55,5565]

# # bins = [0,10,20,30,40,50,60,70,80,90,100,120,190]

# # plt.hist(popution_ages,bins,histtype='bar',rwidth=0.8)

# # plt.title('histgram')
# # plt.xlabel(' x axis')
# # plt.ylabel('y axis')
# # plt.legend()
# # plt.grid()
# # plt.show()

# # x = [2,3,4,5,56,6,7,8]
# # y = [4,5,6,7,8,9,9,0]

# # plt.scatter(x,y,label='scatter')
# # plt.title('sactter plot')
# # plt.xlabel('x axis')
# # plt.ylabel('y axis')
# # plt.legend()
# # plt.grid()
# # plt.show() 

# # stack-plot

# # days = [1,2,3,4,5]
# # sleeping = [7,8,5,4,4]
# # eating = [4,5,6,6,2]
# # working = [3,4,5,6,8]
# # playing = [4,7,6,5,0]


# # plt.plot([],[],colour='m',label='sleeping')
# # plt.plot([],[],colour='c',label='eating')
# # plt.plot([],[],colour='r',label='working')
# # plt.plot([],[],colour='k',label='playing')

# # plt.stackplot(days,sleeping,eating,working,playing,colors=['m','c','r','k'])

# # plt.xlabel('x axis')
# # plt.ylabel('y axis')
# # plt.legend()
# # plt.grid()
# # plt.show()

# # /piechart

# # import matplotlib.pyplot as plt
# # # 
# # slice = [7,6,5,4]

# # activities = ['eat','sleep','code','other activity']
# # cols = ['c','m','r','g']

# # plt.pie(slice,labels=activities,colors=cols,startangle=90,
# # explode=(0,0.1,0,0),autopct='%1.1f%%')

# # plt.title('pie plot')
# # plt.show()
# # # # 
# # slices = [2,3,4,5]

# # activities = ['eat','sleep','code','reapeat']

# # cols = ['m','r','g','b']

# # plt.pie(slices,labels=activities,colors=cols,
# # startangle=90,explode=(0,0.1,0,0),autopct='%1.1f%%')
# # plt.title('pie chart')
# # plt.show()

# # hexgonal pin plot

# # import numpy as np
# # import matplotlib .pyplot as plt

# # def f(t):
# #     return np.exp(-t) *np.cos(2*np.pi*t)

# # t1 = np.arange(0.0,5.0,0.1)
# # t2 = np.arange(0.0,5.0,0.02)
# # plt.subplot(211)
# # plt.plot(t1,f(t1),'bo',t2,f(t2))
# # plt.subplot(211)
# # plt.plot(t2,np.cos(2*np.pi*t2))
# # plt.show()


# # why data visulasation ?
# # coz , i think human brain can easily understabd if the data in geographical and 
# # pictial format

# # matplotlib helps to convert those data in pictial and binary format
# # its deal with the 2D graph

# # from typing import no_type_check_decorator
# import matplotlib .pyplot as plt
# from matplotlib import style
# # from numpy.lib.histograms import histogram

# # a = ([2,3,4],[5,6,7])
# # b = ([3,4,5],[4,5,6])

# # plt.plot(a,b,label='its point')
# # plt.title('g -graph')
# # plt.xlabel('x axis')
# # plt.ylabel('y axis')
# # plt.grid()
# # plt.show()

# # histogram

# # population_age = [12,13,14,15,16,17,18,19,20]
# # discriptive_order =[10,20,30,40,50,60,70,80,90]


# # plt.hist(population_age,discriptive_order,histtype='bar',rwidth=0.8)
# # plt.title('histogram')
# # plt.xlabel('x axis')
# # plt.ylabel('y label')
# # plt.show()
# # plt.show

# # sctter plt

# # x = [2,3,4,5,6,78,9]
# # y = [3,4,5,6,6,7,8]

# # plt.scatter(x,y,label='its point')
# # plt.title('scatter plot')
# # plt.xlabel('x axis')
# # plt.ylabel('y axis') 
# # plt.show()


# # lets craete a pie no_type_check_decorator


# # import matplotlib.pyplot as plt
# # # 
# # slice = [7,6,5,4]

# # activities = ['eat','sleep','code','other activity']
# # cols = ['c','m','r','g']

# # plt.pie(slice,labels=activities,colors=cols,startangle=90,
# # explode=(0,0.1,0,0),autopct='1%.1f%%')

# # plt.title('pie plot')
# # plt.show()
# # # # 



# import matplotlib .pyplot as plt
# from matplotlib import style

# slices = [1,2,3,4]
# activities = ['eat','sleep','code','repeat']
# col = ['m','r','g','b']

# plt.pie(slices,labels=activities,colors=col,startangle=90,explode=(0,0.1,0,0),autopct='1%.1f%%')
# plt.title('pie-chart')
# plt.show()


# lets draw a hexagonal binplot

# import matplotlib .pyplot as plt
# from scipy import special
# import numpy as np

# def f(t):
#     return np.exp(-t) *np.cos(2*np.pi*t)  

# t1 = np.arange(0.0,5.0,0.1)
# t2 = np.arange(0.0,5.0,0.02)
# plt.subplot(211)
# plt.plot(t1,f(t1),'bo',t2,f(t2))
# plt.subplot(211)
# plt.plot(t2,np.cos(2*np.pi*t2))
# plt.show()



# def funtion(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)
    









# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------






# Data analysis

# years = range(2000,2012)
# apples = [0.44,0.54,0.54,0.55,0.66,0.88,0.66]
# orange = [0.44,0.55,0.34,0.54,0.54,0.76,0.87]

# plt.plot(years,apples)
# plt.plot(years,orange)


# plt.show()
# so the numbers of items that we havce been alredy so/ /?

# from typing import Annotated
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd


# # load data into pandas dataframe
# floower_df = sns.load_dataset('iris')
# print(floower_df)
# print(floower_df.species.unique())

# # plt.plot(floower_df.sepal_length,floower_df.sepal_width)

# # plt.scatter(floower_df.sepal_length,floower_df.sepal_width)


# specifying the hue
# print(sns.scatterplot(floower_df.sepal_length,
# floower_df.sepal_width,
# hue=floower_df.species,s=100))

# plt.show()

from ast import Index, Param
from http.client import PRECONDITION_FAILED, error
from operator import index
from re import I, L, S, T
from types import prepare_class
from typing import Sized, overload
import urllib
from urllib import request
from PIL import Image
from matplotlib import axes, cm, image, markers
import matplotlib.pyplot as plt
from numpy.__config__ import show
from numpy.core.fromnumeric import _size_dispatcher, shape
from numpy.lib.financial import ppmt
from numpy.lib.function_base import average, percentile
from numpy.lib.histograms import histogram
from numpy.lib.index_tricks import AxisConcatenator
from numpy.lib.npyio import _save_dispatcher
from numpy.lib.shape_base import tile
from numpy.lib.type_check import imag
from numpy.testing._private.utils import assert_array_equal, clear_and_catch_warnings
from pandas.core.frame import DataFrame
from pandas.core.tools.datetimes import should_cache
import seaborn as sns
import pandas as pd
from seaborn.matrix import heatmap
from seaborn.rcmod import plotting_context


# flower_df = sns.load_dataset('iris')
# print(flower_df)

# print(flower_df.species.unique())

# print(flower_df.sepal_length,flower_df.sepal_width)


# adding hues
# print(sns.scatterplot(flower_df.sepal_length,flower_df.sepal_width,
# hue=flower_df.species,s=100))

# general plotting
# plt.plot(flower_df.sepal_width,flower_df.sepal_length)
# plt.scatter(flower_df.sepal_width,flower_df.sepal_length)
# plt.legend(['sepal_width','sepal-length'])
# plt.title('sepal Dimention')
# # plt.xlabel('x aixs')
# # plt.ylabel('y aixs')
# plt.show()
# print(flower_df)






# cutomizing_seaborn_figurews

# plt.figure(figsize=12)
# plt.show()

# plotting using pandas DataFrame
# print(sns.scatterplot('sepal_length','sepal_width',hue='species',S=100,data=flower_df))

# with histogram plotting those data

# flower_df = sns.load_dataset('iris')
# print(flower_df)
# print(flower_df.sepal_length)
# print(flower_df.sepal_length.describe())

# print(flower_df.sepal_width)
# print(flower_df.sepal_width.describe())

# plt.title('Decription of sepals width')
# plt.hist(flower_df.sepal_length)
# plt.xlabel('x axis')
# plt.ylabel('y axis')

# plt.title('decription of sepal_width')
# plt.hist(flower_df.sepal_width.describe())
# # plt.xlabel('x axis')
# # plt.ylabel('x axis')

# plt.show()


# # with the help of numpy ,so will gonna use some funtion of numpy

import numpy as np
from seaborn.widgets import choose_dark_palette

a = np.arange(2,5,0.25)
print(a)

# # specifying the boundries of eachline
# # plt.title('boundries_ditribution_using_numpy')
# # plt.hist(flower_df.sepal_width,bins=np.arange(2,5,0.2))
# # plt.show()


# b = np.arange(1,4,3.7)
# print(b)

# specifying the boundries of unequalline
# plt.hist(flower_df.sepal_length,bins=np.arange(1,4,3.7))
# plt.show()

# multiple histogram

# flower_df = sns.load_dataset('iris')
# print(flower_df)
# print(flower_df.sepal_width)
# print(flower_df.sepal_length)
# print(flower_df.species.unique())


# setosa_df = flower_df[flower_df.species =='setosa']
# versicolor_df = flower_df[flower_df.species =='versicolor']
# virginica_df = flower_df[flower_df.species =='virginica']

# print(setosa_df)
# print(versicolor_df)
# print(virginica_df)



# plt.hist(flower_df.sepal_length)
# plt.hist(setosa_df.sepal_length)
# plt.plot(setosa_df.sepal_width)



# plt.hist(versicolor_df.sepal_length)
# plt.hist(versicolor_df.sepal_width)
# plt.show()


# plt.hist(virginica_df.sepal_length)
# plt.hist(virginica_df.sepal_length)
# plt.show()



# plt.hist(setosa_df.sepal_length,alpha=0.4,bins=np.arange([2,5,0.25]))
# plt.show()


# plt.hist(setosa_df.sepal_width,versicolor_df.sepal_width,virginica_df.sepal_width,bins=np.arange([2,5,0.25]))
# plt.show()

# a quick revison

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

flower_df = sns.load_dataset('iris')
# print(flower_df)

# print(flower_df.describe())
# print(flower_df.species.unique())

# print(flower_df.sepal_width)
# print(flower_df.sepal_length)


# plt.plot(flower_df.sepal_width)
# plt.plot(flower_df.sepal_length)
# plt.title('Distribution of sepals')
# plt.grid()
# plt.legend(['sepal_width','sepal_legend'])
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.show()

setosa_df = flower_df[flower_df.species =='setosa']
versicolor_df = flower_df[flower_df.species =='versicolor']
virginica_df = flower_df[flower_df.species=='virginica']

# print(setosa_df)
# print(versicolor_df)
# print(virginica_df)

# # setosa graphical format(using bar)
# plt.plot(setosa_df.sepal_length)
# plt.plot(setosa_df.sepal_width)

# # setosa graphical format(using histogram)
# plt.hist(setosa_df.sepal_length)
# plt.hist(setosa_df.sepal_width)


# # versicolor graphical format(using bar)

# plt.title('distribution')
# plt.grid()
# plt.xlabel('no_of_days')
# plt.ylabel('no_of_heredity')
# plt.legend(['sepal_length','sepal_width'])
# plt.plot(versicolor_df.sepal_length,marker='o')
# plt.plot(versicolor_df.sepal_width,marker='x')

# # versicolor graphical format (using histogram)
# plt.hist(virginica_df.sepal_length)
# plt.hist(virginica_df.sepal_width)
# plt.show()

# plt.histplo(setosa_df.sepal_width,versicolor_df.sepal_width,virginica_df,bins=np.arange(2,5,0.25),stacked=True)





# bar chart

# 
years = [2000,2001,2002,2003,2004,2005]
oranges = [0.32,0.43,0.12,0.43,0.54,0.67]

apple = [0.32,0.12,0.67,0.17,0.67,0.67]

# # plt.plot(years,oranges)
# plt.ylabel('Rate')
# plt.xlabel('year')
# plt.bar(years,oranges)

# plt.bar(years,apple,bottom=oranges)
# plt.legend(['oranges','apple'])
# plt.show()





# bar plotted with average

tips_df = sns.load_dataset('tips')
# print(tips_df)
# print(tips_df.day)
# print(tips_df.total_bill)

# days = ['Thu','Fri','Sat','Sun']

# # avg_total_bill = [15.2,13.7,12.6,17.8]

# plt.bar()
# plt.show()







# # u can directlty call those data from(dataset) in bargraph by uing barplot


# # print(sns.barplot('day','total_bill',data=tips_df))
# print(sns.barplot('tip','size',data=tips_df))
# plt.show()

# tips_df = sns.load_dataset('tips')
# print(tips_df)

# print(tips_df.size)

# tips_df = sns.load_dataset('tips')
# print(tips_df)

# print(tips_df.size)
# # print(tips_df.day)
# # print(tips_df.sex)

# plt.plot(tips_df.day)
# plt.bar(tips_df)
# plt.show()

# print(sns.barplot('day','total_bill',hue='sex',data=tips_df))
# plt.show()
# # print(sns.barplot('size','day',data=tips_df))
# plt.show()
# print(S)


# heatmap

# flights_df = sns.load_dataset('flights').pivot('month','year','passengers')
# print(flights_df)

# plt.title("No of passengers(1000s")
# print(sns.heatmap(flights_df))
# plt.show()


# # using diffrent color maps
# plt.title('No of passengers (1000s)')
# print(sns.heatmap(flights_df,fmt='d',annot=True,cmap='Blues'))
# plt.show()


# plt.title("no of passengers (1000s)")
# # print(sns.heatmap(flights_df))
# print(sns.heatmap(flights_df,fmt='d',annot=True,cmap='Blues'))
# plt.show()
# url = urlretrieve('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.shutterstock.com%2Fcategory%2Fnature&psig=AOvVaw2IMNhZz8Z71Ai6GUbxUX09&ust=1623220540106000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCLDt7fC1h_ECFQAAAAAdAAAAABAD')

# print(url)

# quick revision of heatmap

flights_df = sns.load_dataset('flights').pivot('month','year','passengers')
# print(flights_df)

# plt.title('No of passengers')
# sns.heatmap(flights_df)
# plt.show()

# plt.title('No of passengers')
# sns.heatmap(flights_df,fmt='d',annot=True,cmap='Blues')
# plt.show()


tips_df = sns.load_dataset('tips')
# print(tips_df)
# print(tips_df,'day','total_bill')


# # plt.bar('day','total_bill',data=tips_df)
# # plt.show()

# sns.barplot('day','total_bill',hue='sex',data=tips_df)
# plt.show()

# from urllib .request import urlretrieve

# url = urlretrieve('https://www.google.com',)
# print('mn ')

# from urllib import parse
# a = dir(parse)
# print(a)

# Params = {"y" : "EuC-yVzHhMI","t":"5m56s"}
# # print(Params)

# querystring = parse.urlencode(Params)
# print(querystring)

# url = "http://youtube.com/watch" + "?" + querystring
# print(url)

# resp = request.urlopen(url)
# print(resp)


# html = resp.read().decode("utf-8")
# print(html[:500])


# from urllib import parse

# request.urlretrieve('https://i.imgur.com/SkPbq.jpg','chart.jpg')

# from PIL import Image

# img = Image.open('chart.jpg')

# img.array = np.array(img)
# print(img)

# plt.imshow(img)
# # plt.show()


# 
# lets load the datsets using img from any request.urlretrieve

# from urllib .request import urlretrieve

# url =request.urlretrieve('https://i.imgur.com/SkPbq.jpg','chart.jpg')
# print(url)

# from urllib import parse

# img = Image.open('chart.jpg')
# img_arr= img.array = np.array(img)

# # b = img.array[160,160] = np.array(img)
# # print(b)
# print(img_arr)


# shp = img.size
# print(shp)



# plt.ylabel('y axis')
# plt.xlabel('x axis')
# plt.axis('off')
# plt.grid(False)
# plt.title('A Data Science Meme')
# plt.imshow(img_arr[125:325,105:305])
# plt.show()


# we can alos call the partial of that img


# plt.imshow(img_[125:325,105:305])
# plt.show()


# here the yt code for plotting multiple grap

# uses the area for plotting


# fig,axes = plt.subplots(2,3,figsize=(16,8))

# print(axes[0,0],plt.plot(years,oranges,'s-b'))
# print(axes[0,0],plt.plot(years,apple,'o--r'))
# print(axes[0,0].set_xlabel('year'))
# print(axes[0,0].set_ylabel('yeild (tons per hector)'))
# plt.legend(['oranges','apple'])
# plt.title('crop yields in kanto')
# plt.grid(True)
# # plt.show()


# # pass the axes into seaborn

# axes[0,1].set_title('sepal Length vs.sepal_width')
# sns.scatterplot(flower_df.sepal_length,flower_df.sepal_width,hue=flower_df.species,s=100,ax=axes[0,1])
# # plt.show()

# # uses the axes for plotting
# axes[0,2].set_title('Distribution of sepal_width')
# print(axes[0,2].hist([setosa_df.sepal_width,versicolor_df.sepal_width,virginica_df.sepal_width],bins=np.arange(2, 5, 0.25),stacked=True))
# print(axes[0,2].legend(['setosa','versicolor','virginica']))
# # plt.show()


# # pass the axes into sns
# axes[1,0].set_title('Restaurent Bills')
# print(sns.barplot('day','total_bill',hue='sex',data=tips_df,ax=axes[1,0]))
# # plt.show()


# # pass the axes intp sns
# axes[1,1].set_title('flights traffic')
# print(sns.heatmap(flights_df,cmap='Blues',ax=axes[1,1]))
# # plt.show()


# # plot a image using the axes

# # print(axes[1,2].set_title('Data Science Meme'))
# print(axes[1,2].imshow(img))
# # print(axes[1,2].set_xticks([]))
# # print(axes[1,2].set_yticks([]))
# plt.tight_layout(pad=2)
# plt.show()


# quick revision for img plotting

# from urllib .request import URLopener

# url = request.urlretrieve('https://i.imgur.com/SkPbq.jpg','chart.jpg')

# from urllib import parse

# img = Image.open('chart.jpg')
# print(img)

# img_arr  = img.array = np.array(img)
# print(img_arr)

# shp = img.size
# print(shp)

# plt.title('A Data Sciene Meme')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.grid(False)
# plt.imshow(img)
# plt.show()


# pairplotting
# from sepals datasets
# sns.pairplot(flower_df,hue='species')
# # plt.show()

# # from tips_df dataset
# sns.pairplot(tips_df,hue='sex')
# plt.show()

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------





# import pandas as pd


# loading datasets 
survey_raw_df = pd.read_csv( 'D:\survey_results_public.csv')
# print(survey_raw_df)
# print(type(survey_raw_df))
# print(survey_raw_df.columns)
# print(survey_raw_df.head())
# print(survey_raw_df.shape)



schema_fname =pd.read_csv( 'D:\survey_results_schema.csv')
print(schema_fname)
# print(type(schema_fname))
# print(schema_fname.columns)
# print(schema_fname.head())

# print(survey_raw_df.info)
# print(schema_fname.info)
# print(schema_fname.shape)

schema_raw = pd.read_csv(schema_fname,index_col='Column').QuestionText
print(schema_raw)

# print(S)`2`

# Data preparation & cleaning

# selceted_columns = ['Country','Age','Gender','EdLevel'
# ,'UndergradMajor','Hobbyist','AgelstCode','YearsCode'
# ,'YearsCodePro','LanguageWorkedWith','LanguageDesireNextYear'
# ,'NEWLearn',"NEWStuck",'Employment','DevType'
# ,'WorkWeekHRS','JOBSat','JobFactors','NEWOvertimes']

# s = len(selceted_columns)
# print(s)

# # err
survey_raw = survey_raw_df[selceted_columns].copy()
# print(survey_raw)


survey_raw_df['Age1stCode'] = pd.to_numeric(survey_raw_df.Age1stCode,errors='coefficient')

# print(survey_raw_df)
# print(survey_raw)

# print(selceted_columns.describe)

# print(survey_raw_df.describe)
# # print(schema_fname.Age1stCode)


# print(survey_raw_df.drop(survey_raw_df[survey_raw_df.age<10])) .index,input,inplace=True)
# print(survey_raw_df.drop(survey_raw_df.age< 10)).index,input,inplace=True)

# print(schema_raw('YearCodePro'))

plt.figure(figsize=(12,8))
plt.xticks(rotation=75)
plt.show()