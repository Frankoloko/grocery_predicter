import numpy
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression

########################################################
# PREP THE DATA

data = [
    [0, 1],
    [1, 8],
    [2, 13],
    [3, 16],
    [4, 20],
]

# array() coonverts the nested array into it's own array
# Reshape converts [0 1 2 3] to [[0] [1] [2] [3]] i.e. from shape(1, n) to shape(n, 1).
X = numpy.array(data)[:, 0].reshape(-1, 1)
Y = numpy.array(data)[:, 1].reshape(-1, 1)

to_predict_x = [5, 6, 7]
to_predict_x = numpy.array(to_predict_x).reshape(-1, 1)

########################################################
# TRAIN AND TEST THE MODEL

regressor = LinearRegression() # I think this is equal to "y = m * x + c"
regressor.fit(X, Y) # fit() is the "train" part

predicted_y = regressor.predict(to_predict_x)
m = regressor.coef_[0][0]
c = regressor.intercept_[0]

print('X \n', X)
print('Y \n', Y)
print('to_predict_x \n', to_predict_x)
print("Predicted y: \n", predicted_y)
print("slope (m): ", m)
print("y-intercept (c): ", c)

########################################################
# GRAPH THE DATA

# Set titles
pyplot.title('Predict the next numbers in a given sequence')  
pyplot.xlabel('X')
pyplot.ylabel('Numbers')

# Scatter on the graph X and Y values
pyplot.scatter(X, Y, color="blue")
pyplot.scatter(to_predict_x, predicted_y, color="green")

# numpy.append just joins [[0],[1]] with [[2],[3]] to create [[0],[1],[2],[3]]
x_values = numpy.append(X, to_predict_x) # result: [0 1 2 3 4 5 6 7]

# complicated_array = [ m*i+c for i in x_values ] # Creates arrays inside arrays of [[ 2.4], [ 7. ], [11.6], [16.2], [20.8], [25.4], [30. ], [34.6]]
# y_values = numpy.array(complicated_array).reshape(-1, 1) # Creates [[ 2.4], [ 7. ], [11.6], [16.2], [20.8], [25.4], [30. ], [34.6]]
# y_values = numpy.append(y_values, [[]]) # Just turns [[0],[1]] into [0, 1]

# Plot the predicter line
y_values = []
for item in x_values:
    y_values.append(m * item + c)
pyplot.plot(x_values, numpy.append(y_values, [[]]), color="red")

pyplot.show()