import csv
import numpy as np
import math

STOPPING_CRITERIA = 0.5
ETA = 0.1
ITERATION_LIMIT = 5000


def partialDerivative(X, Y, w, i):
    result = 0
    for n in range(len(X)):
        factor = -1 * Y[n] * X[n][i]
        exponent = -1 * Y[n] * (w.T @ X[n])
        if exponent > 40:
            result += factor
        else: 
            result += (factor * math.exp(exponent)) / (math.exp(exponent) + 1)
    return result / len(X)

    # result = 0
    # for n in range(len(X)):
    #     insideExp = -1*Y[n]*w[i]*X[n][i]
    #     if insideExp > 40:
    #         result += insideExp
    #     else:
    #         result += math.log(1 + math.exp(insideExp))
    # return result/len(X)


# def gradient(X, Y, w):
#     gradient = []
#     for i in range(len(X[0])):
#         gradient.append(partialDerivative(X, Y, w, i))
#     return np.array(gradient)

def gradient(X, Y, w):
    result = 0
    for n in range(len(X)):
        top = Y[n] * X[n]
        exponent = Y[n] * (w.T @ X[n])
        if exponent <= 200:
            bottom = 1 + math.exp(Y[n] * (w.T @ X[n]))
            result += top/bottom
    return -1 * result / len(X)


def isItTimeToStop(gradient):
    total = 0
    for i in gradient:
        total += i*i
    total = math.sqrt(total)
    print(total)
    return total < STOPPING_CRITERIA


def fullBatchGD(X, Y):
    iterations = 0
    currentW =np.array([0] * len(X[0]))
    while(iterations < ITERATION_LIMIT):
        grad = gradient(X, Y, currentW)
        # print(grad)
        currentW = currentW - ETA * grad
        iterations += 1
        if(isItTimeToStop(grad)):
            break
    print(iterations)
    return currentW


def main():
    # Read the csv file
    X = []
    Y = []
    with open('vehicle.csv') as file:
        reader = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in reader:
            if lineCount == 0:
                lineCount += 1
                continue

            newRow = []
            for i in range(len(row)):
                if i == len(row)-1:
                    if row[i]=='saab' or row[i]=='van':
                        X.append(newRow)
                        if row[i]=='van':
                            Y.append(1)
                        else:
                            Y.append(-1)
                else:
                    newRow.append(int(row[i]))
            lineCount += 1

    X = np.array(X)
    Y = np.array(Y)
    # To do: calculate gradient function    
    print("Final W: ", fullBatchGD(X, Y))
#     a = np.array([94,36,66,151,61,8,133,50,18,135,154,265,119,62,9,3,201,208])
#     c = np.array([97,44,96,195,63,9,185,36,22,144,202,512,165,66,4,8,191,199])
#     b = np.array([ 1.74495236e+01,  8.22235754e+00 , 2.17668652e+00, -1.15744331e+00,
#   1.80156282e+01 , 4.51802923e+00, -8.16826186e+00 , 2.57632232e+01,
#   1.63581822e+00 , 3.93774100e+01 , 4.48558554e+00 ,-1.64846137e+02,
#   1.82596223e+01 , 2.46562533e+01 , 2.76445179e-02, -5.27403803e+00,
#   5.25829418e+01,  5.36105863e+01])
    
    # print(a @ b)
    # print(1/(1 + math.exp(-1*(a @ b))))
    # print(math.log(1 + math.exp(2)))

if __name__ == "__main__":
    main()