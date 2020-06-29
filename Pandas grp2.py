# Pandas (panal data)
# Series - 1-d
# dataframe - 2d
# panal data - 3d.........

# Series
import pandas as pd

list1 = [1,2,3,4]
ds1 = pd.Series(list1)
print (ds1)

df2 = pd.Series(list1, index = [1,2,3,4,5])
print (df2)

x1 = [5,6,7,8, 9]
y1 = [1,2,3,4, 5]
df3 = pd.Series(x1, index = y1)
print (df3)

df2+df3
df4 = pd.Series(x1, y1)
print (df4)


x2 = ['a','b','c','d']
y2 = [1,2,3,4]
dict1 = dict(zip(x2, y2))
print (dict1)

pds = pd.Series(dict1)
print (pds)


df4+pds
df4+df3


pds['c']
pds[1]
pds[1:3]
pds['a':'d']
pds[:'d']


##### Dataframe
list2 = [1,2,3,4,5]
df1 = pd.DataFrame(list2)
print (df1)

dict2 = {'fruits': ['apple', 'mangos', 'muskmelon'],
         'count': [10, 15, 20]}

df3 = pd.DataFrame(dict2)
print (df3)
df3[1:2]
df3[2:3]

############## Series to dataframe
series1 = pd.Series([5,10], index = ['a', 'b'])
df2 = pd.DataFrame(series1)
print (df2)

# numpy
import numpy as np
arr1 = np.array([[50, 100], ['Python', 'DS']] )
df4 = pd.DataFrame({'name':arr1[1], 'values':arr1[0]})
df5 = pd.DataFrame(arr1[1], arr1[0])

#
A = [1,2,3,4]
B = [5,6,7,8]
C = [9,1,2,3]
D = [4,5,6,7]
E = [8,9,1,2]

df6 = pd.DataFrame([A,B,C,D,E], ['a', 'b', 'c', 'd', 'e'],
                   ['w', 'x', 'y', 'z'])
print (df6)

df6['sum(z,y)'] = df6['z'] + df6['y']
df6['g'] = [1,2,3,4,5]


df6.append(1,2,3,4)

df6.drop('e') # deleting row

df6.drop('g', axis = 1)
df6.drop('c', axis = 0, inplace = True)

df6.drop('z', axis = 1, inplace = True)

########## Conditional accessing
print (df6)
df6>5
df6[df6>5]
df6[df6['y']>5]

df6[df6['y']>5][['y']]
df6[df6['y']>5][['g']]

df6[df6['y']>5][['g', 'y']]

######## and n or
df6[(df6['sum(z,y)']<=5) & (df6['g']>=5)]

df6[(df6['sum(z,y)']<=5) & (df6['g']>5)]

df6[(df6['sum(z,y)']<=5) | (df6['g']>5)]

df6[(df6['sum(z,y)']<=5) ^ (df6['g']>=5)]

df6[(df6['sum(z,y)']<=5) << (df6['g']>=5)]

######### Missing data
import numpy as np
np.nan
dict1 = {'a':[1,2,3,4,5], 'b':[1,2,3,4,np.nan],
         'c': [7,8,9,np.nan,np.nan], 'd':[2,np.nan,np.nan,np.nan,np.nan],
         'e': [np.nan,np.nan,np.nan,np.nan,np.nan]}
print (dict1)

df6 = pd.DataFrame(dict1)
print (df6)

df6.dropna(axis = 0)
df6.fillna(2)

df6['b'].fillna(value = df6['b'].mean())

df6['d'].fillna(value = df6['d'].mean())

df6.drop('e', axis = 1)

############### Group By
shop = {'items': ['milk', 'milk','bread', 'egg', 'tofu', 'egg'],
        'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        'sales': [100,200,50,100,50,200]}

df7 = pd.DataFrame(shop)
print (df7)

grpx = df7.groupby('items')
print (grpx)
grpx.mean()
grpx.std()
grpx.count()
grpx.max()
grpx.min()
grpx.describe()
grpx.describe().transpose()

grpx1 = df7.groupby('items', 'day')

# Merge, Join and concatenation

import pandas as pd

player = ['Virat', 'Rahul', 'Dhoni']
score = [50,45,70]
title = ['captain', 'batsman', 'wckt keeper']

df8 = pd.DataFrame({'Players':player, 'Score':score, 'Title':title})
print (df8)

players = ['Virat', 'Bumrah', 'kartick']
wickets = [2,5,1]
title1 = ['captain', 'boweler', 'none']

df9 = pd.DataFrame({'Players': players, 'Wickets': wickets,
                    'Title':title1})
print (df9)

# Inner merge
pd.merge(df8, df9)
df8.merge(df9)

pd.merge(df8, df9, how = 'inner', on = 'Players')
df8.merge(df9, how = 'inner', on = 'Players')

pd.merge(df8, df9, how = 'left', on = 'Players')
df8.merge(df9, how = 'left', on = 'Players')

pd.merge(df8, df9, how = 'right', on = 'Players')
df8.merge(df9, how = 'right', on = 'Players')

pd.merge(df8, df9, how = 'outer', on = 'Players')
df8.merge(df9, how = 'outer', on = 'Players')


# join
player = ['Virat', 'Rahul', 'Dhoni']
score = [50,45,70]
title = ['captain', 'batsman', 'wckt keeper']

df8 = pd.DataFrame({'Players':player, 'Score':score, 'Title':title})
print (df8)

players = ['Virat', 'Bumrah', 'kartick']
wickets = [2,5,1]
title1 = ['captain', 'boweler', 'none']

df9 = pd.DataFrame({'Players1': players, 'Wickets': wickets,
                    'Title':title1})
print (df9)

df8.join(df9)

df8.join(df9, how = 'inner')
df8.join(df9, how = 'outer')
df8.join(df9, how = 'left')
df8.join(df9, how = 'right')


######### Concatenate
pd.concat([df8, df9])
pd.concat([df8, df9], axis = 1)


pd.concat([df8, df9], axis = 1, ignore_index = True)

pd.concat([df8, df9], axis = 0, ignore_index = True)
dfn = pd.concat([df8, df9], axis = 0)


########### Data Analysis

import pandas as pd

df = pd.read_excel('weather.xlsx', 'Sheet1')

df2 = pd.read_excel('C:\\Users\\DELL\\Desktop\\Python DS\\Pandas DataSets\\weather.xlsx')

df = pd.read_excel('weather.xlsx', skiprows = 2)

df = pd.read_excel('weather.xlsx', header = 1)

df = pd.read_excel('weather.xlsx', header = 2)


df = pd.read_excel('weather.xlsx', header = None)

df = pd.read_excel('weather.xlsx', header = None,
                   names = ['date', 'temp', 'air', 'event'])

df = pd.read_excel('weather.xlsx', header = 0,
                   names = ['date', 'temp', 'air', 'event'])

df = pd.read_excel('weather.xlsx', skiprows = 1)

df = pd.read_excel('weather.xlsx', skiprows = 0,
                   names = ['date', 'temp', 'air', 'event'])

df = pd.read_excel('weather.xlsx', nrows = 5)

df = pd.read_excel('weather.xlsx')

df.columns
df.rows

df.shape
rows, columns = df.shape
print (rows)
print (columns)

df.head()
df.tail()

df.head(2)
df.tail(2)

df[2:5]
df[:]
df[2:5:2]

df.to_csv('test.csv')

df.event
df['event']
df['event', 'temperature']
df[['event', 'temperature']]
df[['event', 'temperature', 'day']]

df['temperature'].max()
df['temperature'].min()

df.mean()
df['temperature'].mean()

df.median()
df['temperature'].median()

df.mode()
df['windspeed'].mode()

df.std()
df['windspeed'].std()

df.describe()
df[df.temperature >= 31 ]

df[df.temperature == df.temperature.max() ]

df[df.temperature == df['temperature'].min() ]

df[df.temperature >= 31]

# df.loc[df.event == 'Snow']
# df.loc[df.event == 'Snow' & 'Rain']

# to pull a range
df[(df['temperature']>=32) & (df['temperature']<=34)]

pd.merge(df[df.temperature < 25],df[df.temperature >=20])


######
df.index

df.set_index('day', inplace = True)

df.reset_index(inplace = True)

df.to_excel('test.xlsx')
df.to_excel('test1.xlsx')
df.to_csv('testcsv.csv')

df.to_csv('test12.csv', index = False)

df.to_csv('test11.csv', columns=['temperature', 'windspeed'])

df.to_csv('test13.csv', columns=['temperature', 'windspeed'], index = False)


#
df1 = df.rename(columns = {'temperature':'temp',
                           'windspeed':'Air'})

df.to_csv('test14.csv', header = False)

df.to_csv('test15.csv', header = False, index = False)

########### Missing data
import pandas as pd
df = pd.read_excel('weather.xlsx', 'Sheet1')

df = pd.read_excel('weather.xlsx', parse_dates = ['day'])

type(df.day[0])

df.set_index('day', inplace = True)

# fill na values

dfc = df.fillna(0)
####
df.temperature = df.temperature.fillna(df.temperature.mean())
df.windspeed = df.windspeed.fillna(df.windspeed.mean())
df.windspeed = df.windspeed.fillna(df.windspeed.median())
df.windspeed = df.windspeed.fillna(df.windspeed.mode())

#####
df2 = df.fillna({'temperature': 10,
                 'windspeed': 2,
                 'event':'No event'})

df3 = df.fillna(method = 'ffill')

df3 = df.fillna(method = 'ffill', axis = 1)

df3 = df.fillna(method = 'ffill', limit = 1)

df4 = df.interpolate()

df5 = df.dropna()

df5 = df.dropna(how = 'all')

######################
df5 = df.dropna(thresh = 2)

#######
dtr = pd.date_range('1/1/2020', '1/6/2020')
dtr = pd.DatetimeIndex(dtr)
df6 = df.reindex(dtr)

# Correlation
print (df[['temperature', 'windspeed']].corr())

print (df[['temperature', 'windspeed', 'event']].corr())

print (df[['temperature', 'windspeed', 'day']].corr())

######### changing the datatype
df.temperature = df.temperature.astype(float)
df.info(null_counts = True)
df.info()

############# loc (location) and iloc (integer loc)
df.iloc[:,3]
df.iloc[:,-1]
df.iloc[:,:]
df.iloc[:5,3]
df.iloc[2:5,1:3]

df.loc[:, 'temperature']

df.loc[2:5, 'temperature']

df.loc[2:5, 'temperature':'event']

df.loc[2:5, 3] # it wont work

######
dub = lambda x:x*2
df['windspeed']=df['windspeed'].apply(dub)
df1 = df.apply(dub)

##### Ascending/Dece
df.sort_values(by = 'windspeed')
df.sort_values(by = 'windspeed', ascending = False)

# Replace
import pandas as pd
import numpy as np
df = pd.read_excel('weather.xlsx', 'Sheet1')
df1 = df.replace(np.NaN, 2)

df2 = df.replace({'temperature':np.nan,
                  'windspeed':np.nan}, 10)

df3 = df.replace({np.nan:10,
                  'no event': 'Sunny'})

# Group by
import pandas as pd
df = pd.read_csv('weather1.csv')
grp = df.groupby('city')

# itration
for city, city_df in grp:
    print (city)
    print (city_df)

for city in grp:
    print (city)

grp.get_group('Delhi')

grp.max()
grp.min()
grp.mean()
grp.describe().transpose()

grp.get_group('Delhi').max()
grp.get_group('Delhi').max()

#### Pivot table
import pandas as pd
df = pd.read_csv('weather1.csv')
df.pivot_table(index = 'day', columns = 'city')
df.pivot_table(index = 'day', columns = 'city', 
               aggfunc = 'count').transpose()
df.pivot_table(index = 'day', columns = 'city', 
               aggfunc = 'sum')
df.pivot_table(index = 'temp', columns = 'humidity')

######Margins
df.pivot_table(index = 'city', columns = 'day', 
               margins = True)

pd.__version__
























