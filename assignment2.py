import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys

STOPPING_CRITERIA = 0.5
ITERATION_LIMIT = 10000
STOCHASTIC_BATCH_COUNT = 10


def gradient(X, Y, w):
    result = 0
    for n in range(len(X)):
        top = Y[n] * X[n]
        exponent = Y[n] * (w.T @ X[n])
        if exponent <= 200:
            bottom = 1 + math.exp(Y[n] * (w.T @ X[n]))
            result += top/bottom
    return -1 * (result / len(X))


def isItTimeToStop(gradient):
    if type(gradient) is float:
        return False

    total = 0
    for i in gradient:
        total += i*i
    total = math.sqrt(total)
    return total < STOPPING_CRITERIA


def computeLoss(X, Y, w):
    total = 0
    for n in range(len(X)):
        exponent = -1 * Y[n] * (w.T @ X[n])
        if exponent > 40:
            total += exponent
        else:
            total += math.log(1 + math.exp(exponent))
    return total / len(X)

def fullBatchGD(X, Y, eta, shouldPlot = True):
    iterations = 0
    lossOverIterations = []
    currentW =np.array([0] * len(X[0]))

    while(iterations < ITERATION_LIMIT):
        grad = gradient(X, Y, currentW)
        currentW = currentW - eta * grad

        iterations += 1
        lossOverIterations.append(computeLoss(X, Y, currentW))
        if(isItTimeToStop(grad)):
            break

    if shouldPlot:
        plt.clf()
        plt.plot(lossOverIterations)
        label = "Full Batch Gradient Descent With Eta: " + str(eta)
        plt.ylabel(label)
        filename = "full_batch_gd_eta_" + str(eta) + ".png"
        plt.savefig(filename)
    
    return (currentW, iterations, computeLoss(X, Y, currentW))


def stochasticGD(X, Y, eta, shouldPlot = False):
    iterations = 0
    lossOverIterations = []
    currentW = np.array([0] * len(X[0]))
    converged = False

    while (iterations < ITERATION_LIMIT and not converged):
        for i in range(STOCHASTIC_BATCH_COUNT):
            batchSize = int(len(X) / STOCHASTIC_BATCH_COUNT)
            currentX = X[batchSize*i : batchSize*(i+1)]
            currentY = Y[batchSize*i : batchSize*(i+1)]
            
            grad = gradient(currentX, currentY, currentW)
            currentW = currentW - eta * grad

            lossOverIterations.append(computeLoss(X, Y, currentW))
            if(isItTimeToStop(grad)):
                converged = True
                break
        iterations += 1

    if shouldPlot:
        plt.clf()
        plt.plot(lossOverIterations)
        label = "Stochastic Gradient Descent With Eta: " + str(eta) 
        plt.ylabel(label)
        filename = "stochastic_gd_eta_" + str(eta) + ".png"
        plt.savefig(filename)

    return (currentW, iterations, computeLoss(X, Y, currentW))
    

def computeAccuracy(X, Y, w):
    truePred = 0
    for i in range(len(X)):
        if Y[i] * (w.T @ X[i]) > 0:
            truePred += 1

    return truePred / len(Y)


def crossValidationFullBatch(X, Y, n_fold, eta):
    batchSize = int(len(X) / n_fold)

    print('Cross validation on full batch gradient descent with eta ' + str(eta) + ':')
    iterationTotal = 0
    timeTotal = 0
    lossTotal = 0
    accuracyTotal = 0

    for i in range(n_fold):
        trainX = np.concatenate(( X[ : i*batchSize], X[(i+1) * batchSize : ] ))
        valX = X[i*batchSize : (i+1)*batchSize]
        trainY =np.concatenate(( Y[ : i*batchSize], Y[(i+1) * batchSize : ] ))
        valY = Y[i*batchSize : (i+1)*batchSize]
        
        startTime = time.time()
        w, iterations, loss = fullBatchGD(trainX, trainY, eta, shouldPlot=False)
        timeTaken = time.time() - startTime
        lossOnValidation = computeLoss(valX, valY, w)
        accuracy = computeAccuracy(valX, valY, w)
        print('Run ' + str(i) + ': time taken ' + str(timeTaken) + ', iterations: ' + str(iterations) + 
                ', loss on validation: ' + str(lossOnValidation) + ', accuracy on validation: ' + str(accuracy))

        iterationTotal += iterations
        lossTotal += lossOnValidation
        timeTotal += timeTaken
        accuracyTotal += accuracy

    print('For eta ' + str(eta) + ', average time taken = ' + str(timeTotal/n_fold) + 
            ', average iteration count = ' + str(iterationTotal/n_fold) + 
            ', average loss = ' + str(lossTotal/n_fold) + ', average accuracy = ' + str(accuracyTotal/n_fold) + '\n\n')


def crossValidationMiniBatch(X, Y, n_fold, eta):
    batchSize = int(len(X) / n_fold)

    print('Cross validation on stochastic gradient descent with eta ' + str(eta) + ':')
    iterationTotal = 0
    timeTotal = 0
    lossTotal = 0
    accuracyTotal = 0

    for i in range(n_fold):
        trainX = np.concatenate(( X[ : i*batchSize], X[(i+1) * batchSize : ] ))
        valX = X[i*batchSize : (i+1)*batchSize]
        trainY = np.concatenate(( Y[ : i*batchSize], Y[(i+1) * batchSize : ] ))
        valY = Y[i*batchSize : (i+1)*batchSize]
        
        startTime = time.time()
        w, iterations, loss = stochasticGD(trainX, trainY, eta, shouldPlot=False)
        timeTaken = time.time() - startTime
        lossOnValidation = computeLoss(valX, valY, w)
        accuracy = computeAccuracy(valX, valY, w)
        print('Run ' + str(i) + ': time taken: ' + str(timeTaken) + ', iterations: ' + str(iterations) + 
                ', loss on validation: ' + str(lossOnValidation) + ', accuracy on validation:' + str(accuracy))

        iterationTotal += iterations
        lossTotal += lossOnValidation
        timeTotal += timeTaken
        accuracyTotal += accuracy

    print('For eta ' + str(eta) + ', average time taken = ' + str(timeTotal/n_fold) + 
            ', average iteration count = ' + str(iterationTotal/n_fold) + 
            ', average loss = ' + str(lossTotal/n_fold) + ', average accuracy = ' + str(accuracyTotal/n_fold) + '\n\n')



def crossValidation(X, Y, n_fold):
    batchSize = int(len(X) / n_fold)
    etas = [0.05, 0.2, 0.8]

    for eta in etas:
        print('Cross validation on full batch gradient descent with eta ' + str(eta) + ':')
        iterationTotal = 0
        timeTotal = 0
        lossTotal = 0

        for i in range(n_fold):
            trainX = np.concatenate(( X[ : i*batchSize], X[(i+1) * batchSize : ] ))
            valX = X[i*batchSize : (i+1)*batchSize]
            trainY =np.concatenate(( Y[ : i*batchSize], Y[(i+1) * batchSize : ] ))
            valY = Y[i*batchSize : (i+1)*batchSize]
            
            startTime = time.time()
            w, iterations, loss = fullBatchGD(trainX, trainY, eta, shouldPlot=False)
            timeTaken = time.time() - startTime
            lossOnValidation = computeLoss(valX, valY, w)
            print('Run ' + str(i) + ': time taken = ' + str(timeTaken) + ', iterations = ' + str(iterations) + ', loss on validation: ' + str(lossOnValidation))

            iterationTotal += iterations
            lossTotal += lossOnValidation
            timeTotal += timeTaken

        print('For eta ' + str(eta) + ', average time taken = ' + str(timeTotal/n_fold) + 
                ', average iteration count = ' + str(iterationTotal/n_fold) + 
                ', average loss = ' + str(lossTotal/n_fold) + '\n\n')
        
    for eta in etas:
        print('Cross validation on stochastic gradient descent with eta ' + str(eta) + ':')
        iterationTotal = 0
        timeTotal = 0
        lossTotal = 0

        for i in range(n_fold):
            trainX = np.concatenate(( X[ : i*batchSize], X[(i+1) * batchSize : ] ))
            valX = X[i*batchSize : (i+1)*batchSize]
            trainY = np.concatenate(( Y[ : i*batchSize], Y[(i+1) * batchSize : ] ))
            valY = Y[i*batchSize : (i+1)*batchSize]
            
            startTime = time.time()
            w, iterations, loss = stochasticGD(trainX, trainY, eta, shouldPlot=False)
            timeTaken = time.time() - startTime
            lossOnValidation = computeLoss(valX, valY, w)
            print('Run ' + str(i) + ': time taken = ' + str(timeTaken) + ', iterations = ' + str(iterations) + ', loss on validation: ' + str(lossOnValidation))

            iterationTotal += iterations
            lossTotal += lossOnValidation
            timeTotal += timeTaken

        print('For eta ' + str(eta) + ', average time taken = ' + str(timeTotal/n_fold) + 
                ', average iteration count = ' + str(iterationTotal/n_fold) + 
                ', average loss = ' + str(lossTotal/n_fold) + '\n\n')


def main(step):
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
     
    if step == 'step1':
        crossValidationFullBatch(X, Y, 5, 0.05)
    elif step == 'step2':
        crossValidationMiniBatch(X, Y, 5, 0.05)


    # For plotting graphs   
    # w = fullBatchGD(X, Y, 0.05, shouldPlot=True)
    # w = fullBatchGD(X, Y, 0.2, shouldPlot=True)
    # w = fullBatchGD(X, Y, 0.8, shouldPlot=True)
    # w = stochasticGD(X, Y, 0.05, shouldPlot=True)
    # w = stochasticGD(X, Y, 0.2, shouldPlot=True)
    # w = stochasticGD(X, Y, 0.8, shouldPlot=True)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Check your inputs')
    else:
        step = sys.argv[2]
        main(step)