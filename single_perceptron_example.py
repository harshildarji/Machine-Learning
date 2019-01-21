# Single Perceptron Example (ATD E10.01)
'''
Definition: For a single perceptron, find an assignment to the parameters w0, w1, w2 such that the
perceptron implements the boolean function y(x1, x2) = x1 ^ ¬x2 for binary variables x1
and x2. (Use the Heaviside step function φ(x) = max(sign(x), 0) as threshold function.)
'''

# signum function
def sign(y):
    if y < 0:
        return -1
    elif y == 0:
        return 0
    else:
        return 1

# Heaviside step function
def heaviside(y):
    return max(sign(y), 0)

# learning function
def learningFunction(x, t, w, n):
    for i in range(4):
        y = (w[0] * x[0][i]) + (w[1] * x[1][i]) + (w[2] * x[2][i])
        error = t[i] - heaviside(y)

        for j in range(3):
            dW = n * error * x[j][i]
            w[j] += dW
    return w

if __name__ == '__main__':
    '''
    Intialization:
    x1: 0 0 1 1
    x2: 0 1 0 1
    t: 0 0 1 0 (target values)
    learning rate: n = 0.5
    initial weights: w0 = 0.5, w1 = 0.5, w2 = 0.5
    '''
    x0 = [1, 1, 1, 1]
    x1 = [0, 0, 1, 1]
    x2 = [0, 1, 0, 1]
    t = [0, 0, 1, 0]
    n = 0.4
    w = learningFunction([x0, x1, x2], t, [0.5, 1, -1], n)
    print('w0 = {:.2f}, w1 = {:.2f}, w2 = {:.2f}'.format(w[0], w[1], w[2]))
    for i in range(4):
        y = heaviside((w[0]) + (w[1] * x1[i]) + (w[2] * x2[i]))
        print('For X1 = {}, X2 = {} | y = {:.2f}, t = {}'.format(x1[i], x2[i], y, t[i]))