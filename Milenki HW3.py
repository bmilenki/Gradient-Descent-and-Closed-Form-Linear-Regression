import csv
import numpy as np
from matplotlib import pyplot as plt


def main():
    Question1()
    Question2()
    Question3()


def Question1():
    print("Question 1 Start -------------------")
    N = 10
    X = np.array([[-2],
                  [-5],
                  [-3],
                  [0],
                  [-8],
                  [-2],
                  [1],
                  [5],
                  [-1],
                  [6]])
    Y = np.array([[1],
                  [-4],
                  [1],
                  [3],
                  [11],
                  [5],
                  [0],
                  [-1],
                  [-3],
                  [1]])

    # fileName = "inclass example.csv"
    # data_class= readCSVfile(fileName)

    # X = data_class[:, :-1].astype(np.float64)
    # Y = data_class[:, -1].astype(np.float64)

    dummyVar = []
    for j in range(len(X)):
        dummyVar.append([1])

    print("XOrg:", X)
    print("YOrg:", Y)

    Xb = np.concatenate((dummyVar, X), axis=1)
    print("Xb: ", Xb)

    XtX1 = np.linalg.inv(np.matmul(Xb.transpose(), Xb))
    print("XtX1: ", XtX1)

    XtX1Xt = np.matmul(XtX1, Xb.transpose())
    print("XtX1Xt: ", XtX1Xt.round(4))

    XtX1XtY = np.matmul(XtX1Xt, Y)
    print("XtX1XtY:", XtX1XtY.round(4))

    W = XtX1XtY
    print("W: ", W)

    b = W[0]
    w = W[1:]

    print("b: ", b)
    print("w: ", w)

    Yhat = np.matmul(X, w) + b
    print("Yhat: ", Yhat)

    RMSE = myRMSE(Y, Yhat)
    print("RMSE: ", RMSE)

    print("Question 1 End -------------------")

def Question2():
    print("Question 2 Start -------------------")
    learnRate = .01
    epsilon = 2 ** -32

    prevJ = 100
    Jhist = []
    w1hist = []
    w2hist = []
    epochHist = []
    epochCount = 0

    # initalize
    w = [0, 0]
    x = [[1, 1]]
    J = calcJ(x[0], w)
    N = len(x)

    Jhist.append(J)
    w1hist.append(w[0])
    w2hist.append(w[1])
    epochHist.append(0)

    # actual gradient descent algorithm
    while abs(J - prevJ) > epsilon:
        epochCount += 1
        dJdw1Accum = 0
        dJdw2Accum = 0

        for i in range(N):
            dJdw1Accum += calcdJdw1(x[i], w)
            dJdw2Accum += calcdJdw2(x[i], w)

        prevJ = J

        w[0] = w[0] - learnRate * (1 / N) * dJdw1Accum
        w[1] = w[1] - learnRate * (1 / N) * dJdw2Accum

        J = calcJ(x[0], w)

        Jhist.append(J)
        w1hist.append(w[0])
        w2hist.append(w[1])
        epochHist.append(epochCount)

    # plotting

    # plot J vs Epoch
    plt.plot(epochHist, Jhist)
    plt.title("J vs Epoch")
    plt.xlabel("Epoch")
    plt.xticks(epochHist)
    plt.ylabel("J")
    plt.show()

    # plotting w1vs w2 vs J
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(w1hist, w2hist, Jhist, 'blue')
    plt.title("w1 vs w2 vs J")
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('J')
    fig.show()

    # Printing Final Results
    print("Final w1:", w[0])
    print("Final w2:", w[1])
    print("Final J:", J)
    print("Epochs:", epochCount)

    print("Question 2 End -------------------")


def Question3():
    print("Question 3 Start -------------------")
    fileName = "insurance.csv"
    data = readCSVfile(fileName)

    np.random.seed(0)
    np.random.shuffle(data)

    # Age = x[i][0]
    # Sex = x[i][1]
    # bmi = x[i][2]
    # children = x[i][3]
    # smoker = x[i][4]
    # region = x[i][5]
    X = data[:, :-1]
    Y = data[:, -1]

    # break into train and testing sets
    observations = len(X)
    trainNum = int(observations * (2 / 3))
    testNum = observations - trainNum

    train = X[0:trainNum, :]
    trainAns = Y[0:trainNum].astype(np.float64)

    test = X[trainNum:observations, :]
    testAns = Y[trainNum:observations].astype(np.float64)

    # Model 1
    M1_trainX = convertAllCatToEnum(train)
    M1_trainX = M1_trainX.astype(np.float64)
    M1_testX = convertAllCatToEnum(test)
    M1_testX = M1_testX.astype(np.float64)
    M1_W = trainClosedLinRegModel(M1_trainX, trainAns, includeBias=False)

    M1_YhatTrain = np.matmul(M1_trainX, M1_W)
    M1_trainingRMSE = myRMSE(trainAns, M1_YhatTrain)

    M1_YhatTest = np.matmul(M1_testX, M1_W)
    M1_testRMSE = myRMSE(testAns, M1_YhatTest)

    print("Model 1 RMSE For Training Set: ", M1_trainingRMSE)
    print("Model 1 RMSE For Testing Set: ", M1_testRMSE)

    # Model 2
    M2_trainX = M1_trainX.copy()
    M2_B, M2_W = trainClosedLinRegModel(M2_trainX, trainAns, includeBias=True)
    M2_testX = M1_testX.copy()

    M2_YhatTrain = np.matmul(M2_trainX, M2_W) + M2_B
    M2_trainingRMSE = myRMSE(trainAns, M2_YhatTrain)

    M2_YhatTest = np.matmul(M2_testX, M2_W) + M2_B
    M2_testRMSE = myRMSE(testAns, M2_YhatTest)

    print("Model 2 RMSE For Training Set: ", M2_trainingRMSE)
    print("Model 2 RMSE For Testing Set: ", M2_testRMSE)

    # Model 3
    M3_trainX = convertRegionToBinary(train)
    M3_trainX = M3_trainX.astype(np.float64)
    M3_testX = convertRegionToBinary(test)
    M3_testX = M3_testX.astype(np.float64)

    M3_W = trainClosedLinRegModel(M3_trainX, trainAns, includeBias=False)

    M3_YhatTrain = np.matmul(M3_trainX, M3_W)
    M3_trainingRMSE = myRMSE(trainAns, M3_YhatTrain)

    M3_YhatTest = np.matmul(M3_testX, M3_W)
    M3_testRMSE = myRMSE(testAns, M3_YhatTest)

    print("Model 3 RMSE For Training Set: ", M3_trainingRMSE)
    print("Model 3 RMSE For Testing Set: ", M3_testRMSE)

    # Model 4
    M4_trainX = convertRegionToBinary(train)
    M4_trainX = M4_trainX.astype(np.float64)

    M4_testX = convertRegionToBinary(test)
    M4_testX = M4_testX.astype(np.float64)

    M4_B, M4_W = trainClosedLinRegModel(M4_trainX, trainAns, includeBias=True)

    M4_YhatTrain = np.matmul(M4_trainX, M4_W) + M4_B
    M4_trainingRMSE = myRMSE(trainAns, M4_YhatTrain)

    M4_YhatTest = np.matmul(M4_testX, M4_W) + M4_B
    M4_testRMSE = myRMSE(testAns, M4_YhatTest)

    print("Model 4 RMSE For Training Set: ", M4_trainingRMSE)
    print("Model 4 RMSE For Testing Set: ", M4_testRMSE)

    print("Question 3 End -------------------")

def convertRegionToBinary(origX):
    # converts region to binary vars and other categorical vars to enumerated
    newX = origX.copy()

    # order isSouthWest -> isSouthEast -> isNorthWest -> isNorthEast
    binarization = []

    for i in range(len(newX)):
        binArg = []
        for j in range(len(newX[0])):
            # need to convert sex,smoker,region
            if j == 1:  # convert sex
                if newX[i][j] == "female":
                    newX[i][j] = 0
                elif newX[i][j] == "male":
                    newX[i][j] = 1

            elif j == 4:  # convert smoker
                if newX[i][j] == "no":
                    newX[i][j] = 0
                elif newX[i][j] == "yes":
                    newX[i][j] = 1

            elif j == 5:  # convert region
                if newX[i][j] == "southwest":
                    binArg = [1, 0, 0, 0]
                elif newX[i][j] == "southeast":
                    binArg = [0, 1, 0, 0]
                elif newX[i][j] == "northwest":
                    binArg = [0, 0, 1, 0]
                elif newX[i][j] == "northeast":
                    binArg = [0, 0, 0, 1]

        binarization.append(binArg)

    noRegX = np.delete(newX, len(newX[0]) - 1, 1)

    binX = np.concatenate((noRegX, binarization), axis=1)

    return binX


def convertAllCatToEnum(origX):
    newX = origX.copy()

    for i in range(len(newX)):
        for j in range(len(newX[0])):
            # need to convert sex,smoker,region
            if j == 1:  # convert sex
                if newX[i][j] == "female":
                    newX[i][j] = 0
                elif newX[i][j] == "male":
                    newX[i][j] = 1

            elif j == 4:  # convert smoker
                if newX[i][j] == "no":
                    newX[i][j] = 0
                elif newX[i][j] == "yes":
                    newX[i][j] = 1

            elif j == 5:  # convert region
                if newX[i][j] == "southwest":
                    newX[i][j] = 0
                elif newX[i][j] == "southeast":
                    newX[i][j] = 1
                elif newX[i][j] == "northwest":
                    newX[i][j] = 2
                elif newX[i][j] == "northeast":
                    newX[i][j] = 3

    return newX


def trainClosedLinRegModel(X, Y, includeBias=False):
    if includeBias:
        dummyVar = []
        for j in range(len(X)):
            dummyVar.append([1])

        # print("XOrg:", X)
        # print("YOrg:", Y)

        Xb = np.concatenate((dummyVar, X), axis=1)
        # print("Xb: ",Xb)

    else:
        Xb = X

    XtX1 = np.linalg.pinv(np.matmul(Xb.transpose(), Xb))
    # print("XtX1: ", XtX1)

    XtX1Xt = np.matmul(XtX1, Xb.transpose())
    # print("XtX1Xt: ", XtX1Xt)

    XtX1XtY = np.matmul(XtX1Xt, Y)
    # print("XtX1XtY:", XtX1XtY)

    W = XtX1XtY
    # print("W: ", W)

    if includeBias:
        b = W[0]
        w = W[1:]

        # print("b: ", b)
        # print("w: ", w)

        return b, w

    else:
        # print("W: ", W)
        return W


def calcJ(x, w):
    return (x[0] * w[0] - 5 * x[1] * w[1] - 2) ** 2


def calcdJdw1(x, w):
    return 2 * w[0] * (x[0] ** 2) - 10 * x[0] * x[1] * w[1] - 4 * x[0]


def calcdJdw2(x, w):
    return 50 * w[1] * (x[1] ** 2) - 10 * x[0] * x[1] * w[0] + 20 * x[1]


def readCSVfile(name):
    with open(name, newline='', encoding="utf-8-sig") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        data = np.asarray(list(csvreader))
    return data

def myRMSE(Y, Yhat):
    SE = 0
    N = len(Y)
    for i in range(N):
        SE += (Yhat[i] - Y[i]) ** 2

    MSE = SE / N

    RMSE = np.sqrt(MSE)

    return RMSE


if __name__ == "__main__":
    main()
