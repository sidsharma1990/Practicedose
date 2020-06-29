# Numpy for Batch 2
# NaN= not a number
# inf = infinite
import numpy as np
list2 = [[1,4,1.5], [2,2,2],[3,3,8]]
arr2d = np.array(list2)

arr2d.itemsize
arr2d.size

print (arr2d[:,:])
print (arr2d[1:2,1:2:2])
print (arr2d[0,::1])

arr2d[0, 0:2:2]



boolarr = arr2d <= 3

np.nan
np.inf
arr2d_1 = arr2d.astype(float)
arr2d_1[0][0] = np.nan
arr2d_1[0][1] = np.inf

np.isnan(arr2d_1)
np.isinf(arr2d_1)

# both together
missing_flag = np.isnan(arr2d_1) | np.isinf(arr2d_1)
print (missing_flag)

arr2d[missing_flag] = 0

# stats
arr2d.mean()
arr2d.max()
arr2d.min()
arr2d.std()
arr2d.var()
arr2d.cumsum()

# array from an array
arr2 = arr2d[:2, :2]
print (arr2)

arr3 = arr2d[1:, 1:]
print (arr3)

np.unique(arr2d)

# Reshape
arr_re = arr2d.reshape(1,9)
arr_re = arr2d.reshape(9, 1)
print (arr_re)

arr_flat = arr2d.flatten()
print (arr_flat)

arr_rav = arr2d.ravel()
print (arr_rav)

arr_rav[1] = 11
print (arr2d)

arr_flat[1] = 0

# Sequence, repeat
np.arange(1,5)
np.arange(1,5, dtype = float)
np.arange(1,5, dtype = int)
np.arange(1,5,2)

np.zeros([2,2])
np.ones([2,2])

# repeat
list2 =[1,2,3]
np.tile(list2, 3)
np.repeat(list2, 3)
np.repeat(arr2d, 5)

# random number
np.random.rand(1,10)
np.random.rand(5,5)

np.random.randint(1,5,[4,4])

# unique and count
unique, count = np.unique(arr2d, return_counts = True)
print (unique)
print (count)

#####
ind = np.where(arr2d > 3)
print (ind)

[0,1]

lis = np.array([1,2,3,5,6])
ind1 = np.where(lis > 3)
print (ind1)

np.genfromtxt('Numpy.csv', delimiter = ',', skip_header = True)

# to change nan values
data = np.genfromtxt('Numpy.csv', delimiter = ',', skip_header = 1,
              filling_values = 100, dtype = int)

data.ndim
data.shape

data[:5]  # by default, it is a row
data [:5, ]
data [:5, 0]

#++++ for text data
data1 = np.genfromtxt('Numpy.csv', delimiter = ',', 
                      skip_header = 1, dtype = None)
print (data1)

# to save a file
np.savetxt('data.csv', data, delimiter = ',')
np.savetxt('data1.csv', data1, delimiter = ',', encoding = None)

# 
np.save ('data.npy', data, data1)
y = np.load('data.npy')
print (y)

np.savez ('data2.npz', data, data1)
y1 = np.load('data2.npz')
print (y1)
y1.files
y1['arr_0']
y1['arr_1']

###### concatenate
arr1 = np.zeros([4,4])
arr2 = np.ones([4,4])

np.concatenate([arr1, arr2])
np.concatenate([arr1, arr2], axis = 0)
np.concatenate([arr1, arr2], axis = 1)

# stacking
np.vstack([arr1, arr2])
np.hstack([arr1, arr2])

# row and columm wise
np.r_[arr1, arr2]
np.c_[arr1, arr2]

############ Sorting
arr3 = np.random.randint(1, 10, size = [10,5])
print (arr3)
np.sort(arr3)
row1 = np.sort(arr3, axis = 0)
col1 = np.sort(arr3, axis = 1)


# 3d array
x = [[[1,2,3], [1,2,3], [1,2,3]], [[1,2,3], [1,2,3], [1,2,3]], [[1,2,3],[1,2,3],[1,2,3]]]
arr3d = np.array(x)
print (arr3d)

arr3d.ndim

ran = np.arange(0, 24)
ran2d = ran.reshape(12,2)
ran3d = ran.reshape(12,2,1)
print (ran.reshape(3,2,4))

#
print (np.__version__)
np.show_config()

z = np.zeros(10)
z[4] = 1

###### Reg ex
import re

str1 = 'We are leArning,"" python... '
print (re.sub('[A-Z]', '', str1))
print (re.sub('[a-z]', '', str1))
print (re.sub('[~!@#$%^&*(),. \"_+\']', '', str1))

print (re.sub('[^a-zA-Z ]', '', str1))

str2 = 'eat walk talk eat'
x1 = re.compile ('eat')
str2 = x1.sub('sleep3', str2)
print (str2)

str3 = ''' we
are
learning
python'''
line = re.compile('\n')
str_copy = line.sub(' ', str3)
print (str_copy)

#### Numbers
\d = numbers
\D = non-numbers

num1 = '123456 working'
print (len(re.findall('\d', num1)))
print (len(re.findall('\D', num1)))

111-222-3333

num2 = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
str4 = 'My contact number is 111-222-3333'
str5 = num2.search(str4)
print (str5)

#web scrapping
a = 'yes'
print ('yes' in a)

import re
if re.search('python', 'we are learning Python'):
    print ('available')

# Finding a pattern
str7 = 'set, wet, jet, met'
x = re.findall('[a-p]et', str7)
print (x)
for i in x:
    print (i)


str7 = 'set, wet, jet, met'
x = re.findall('[a-z]et', str7)
print (x)
for i in x:
    print (i)


# Finding a pattern
str7 = 'set, wet, jet, met'
x = re.findall('[^a-p]et', str7)
print (x)
for i in x:
    print (i)

txt1 = 'hi, hi i am am sandeep'
repeat = re.findall(r'\b\w[a-zA-Z]\b', txt1, re.I)
print (repeat)
















