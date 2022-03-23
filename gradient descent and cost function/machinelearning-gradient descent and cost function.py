#having a set of data and trying to output an equation/function
#example -> x = [1,2,3,4,5], y = [5,7,9,11,13]
#machine learning outputs -> y = 2x + 3 to predict future values
#example, the area and price example
#finding the best fit line, but we might have many lines
#calculate the errors, or mean squared errors (cost function)
#use the line with least error -> best fit
#gradient descent is an algorithm that finds the best fit line for a given training data set
#3D graph of z axis MSE, x axis m, y axis c
#find the lowest point in the 3D graph (minimum) -> error is minimum
#move a fixed step from highest point to lowest -> if closer to minimum (slope = 0) -> lower steps taken
#use derivatives
#partial derivatives -> keep one variable 0 and derivatize (with respect to x, y, ...)
#learning rate -> b1 = first step, b2 = b1 - learning rate * partial derivative with respect to b = second step

import numpy as np

def gradient_descent(x,y):
    m_curr = c_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.001

    for i in range(iterations):
        y_predicted = (m_curr * x) + c_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        cd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        c_curr = c_curr - learning_rate * cd
        print('m {}, b {}, iterations {}, cost {}'.format(m_curr,c_curr,i,cost))

    #cd = partial derivative of intercept, md = partial derivative of slope (slopes of the graph)
    #m_curr, c_curr = point at which the arrow starts
    #iterations = how many times we repeat
    #n = length of training data
    #learning rate = how much we increase per step
    #cost = MSE = get as low as possible (reduce the error as much as possible -> approach 0)

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)
