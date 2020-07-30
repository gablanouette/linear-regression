# This project is coding my own version of least squares linear regression
# To do so, we will:
#    Calculate the least squares weights
#    Select data by column  
#    Implement column cut-offs 


# first import modules
import pandas as pd
import numpy as np


# EDA

### import the necessary modules and sets a few plotting parameters for display

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# load training and testing datasets
training_path = './train.csv'
test_path = './test.csv'
data = pd.read_csv(tr_path)

# explore the data
data.head()

# list data columns 
data.columns


### We can plot the Price v. Living Area to find a correlation (note: there will be a positive correlation between the variables)

Y = data['SalePrice']
X = data['GrLivArea']

plt.scatter(X, Y, marker = "x")
plt.title("Sales Price vs. Living Area (excl. basement)")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice");

# we can plot the Price v. Year Built to find a correlation:

data.plot('YearBuilt', 'SalePrice', kind = 'scatter', marker = 'x')


# find the inverse matrix
def inverse_of_matrix(mat):
    """Calculate and return the multiplicative inverse of a matrix
    """
    matrix_inverse = np.linalg.inv(mat)
    return matrix_inverse

# create a function to read in data
def read_to_df(file_path):
    """Read on-disk data and return a dataframe
    """
    return pd.read_csv(file_path)
    
# create function to select a subset dataframe consisting of only the specified columns
def select_columns(data_frame, column_names):
    return data_frame[column_names]

# create function to create column cutoff which includes rows that have a value that exceeds the "max_value" or is less than "min_value"
def column_cutoff(data_frame, cutoffs):
    """Subset data frame by cutting off limits on column values
    """
    sub_df = data_frame
    for limits in cutoffs:
        sub_df = sub_df.loc[sub_df[limits[0]] >= limits[1],:]
        sub_df = sub_df.loc[sub_df[limits[0]] <= limits[2],:]
    return sub_df



# it is time to implement the Least Squares equation using the inverse matrix:  
# w_{LS} = (X^T X)^{−1}*X^T*y

# create a function to:
#    1. ensure that the number of rows of each matrix is greater than or equal to the number of columns. If not, transpose the matricies. 
        In particular, the y input should end up as a $n\times1$ matrix, and the x input as a $n\times p$ matrix
#    2. *prepend* an n*1 column of ones to the x input matrix
#    3. Use the above equation to find the least squares weights

def least_squares_weights(input_x, target_y):

    if input_x.shape[0] < input_x.shape[1]:
        input_x = input_x.transpose()
    if target_y.shape[0] < target_y.shape[1]:
        target_y = target_y.transpose()
                
    #assert input_x.shape[0] == target_y.shape[0]
    
    ones = np.ones((len(target_y), 1), dtype=int)
    
    input_x = np.concatenate((ones, input_x), axis=1)
    
    wLS = np.linalg.inv(input_x.transpose().dot(input_x)).dot(input_x.transpose()).dot(target_y)
    
    return wLS
    
# note: in the above function, it is necessary to prepend a column of ones to create the intercept term


#### now it is time to test on real data:

def column_cutoff(data_frame, cutoffs):
    data_subset = data_frame
    for column_limits in cutoffs:
        data_subset = data_subset.loc[data_subset[column_limits[0]] >= column_limits[1],:]
        data_subset = data_subset.loc[data_subset[column_limits[0]] <= column_limits[2],:]
    return data_subset
def least_square_weights(input_x, target_y):
    if input_x.shape[0] < input_x.shape[1]:
        input_x = np.transpose(input_x)
        
    if target_y.shape[0] < target_y.shape[1]:
        target_y = np.transpose(target_y)
        
        
    ones = np.ones((len(target_y), 1), dtype=int)
    
    augmented_x = np.concatenate((ones, input_x), axis=1)
    
    left_multiplier = np.matmul(np.linalg.inv(np.matmul(np.transpose(augmented_x), 
                                                        augmented_x)),
                                np.transpose(augmented_x))
    w_ls = np.matmul(left_multiplier, target_y)   
    
    return w_ls

df = read_to_df(tr_path)
df_sub = select_columns(df, ['SalePrice', 'GrLivArea', 'YearBuilt'])

cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
df_sub_cutoff = column_cutoff(df_sub, cutoffs)

X = df_sub_cutoff['GrLivArea'].values
Y = df_sub_cutoff['SalePrice'].values

### reshaping for input into function
training_y = np.array([Y])
training_x = np.array([X])

weights = least_squares_weights(training_x, training_y)
print(weights)

max_X = np.max(X) + 500
min_X = np.min(X) - 500

### Choose points evenly spaced between min_x in max_x
reg_x = np.linspace(min_X, max_X, 1000)

### Use the equation for our line to calculate y values
reg_y = weights[0][0] + weights[1][0] * reg_x

plt.plot(reg_x, reg_y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='k', label='Data')

plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.show()


#### Calculating RMSE
rmse = 0

b0 = weights[0][0]
b1 = weights[1][0]

for i in range(len(Y)):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/len(Y))
print(rmse)


#### Calculating R^2

ss_t = 0
ss_r = 0

mean_y = np.mean(Y)

for i in range(len(Y)):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print(r2)



# it is important to understand how to build algorithms from scratch
# sklearn implementation below:

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# sklearn requires a 2-dimensional X and 1 dimensional y. The below has shapes of:
# skl_X = (n,1); skl_Y = (n,)
skl_X = df_sub_cutoff[['GrLivArea']]
skl_Y = df_sub_cutoff['SalePrice']

lr.fit(skl_X,skl_Y)
print("Intercept:", lr.intercept_)
print("Coefficient:", lr.coef_)
