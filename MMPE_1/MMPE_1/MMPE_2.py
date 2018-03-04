import numpy as np
import MMPE_1 as lr1
from scipy.optimize import minimize 
s = 2
# максимизация мю (1 шаг двойственного градиента)
def calc_mu_A_op(D2, alpha):
    M_sing = lr1.single_point_FIM(alpha[0], alpha[1], 1)
    #домножим на -1, т.к. максимизируем
    return (-1) * np.trace(np.matmul(D2, M_sing))
def maximize_mu(D2):
    fun = lambda x: calc_mu_A_op(D2, x)
    x0 = np.array([np.random.uniform(-1., 1., 1), np.random.uniform(-1., 1., 1)])
    res = minimize(fun, x0, method='TNC', bounds=[[-1., 1.]] * 2)
    return res.fun, res.x

def make_ksi_tau(alpha_k, p_k, alpha, tau):
    import copy
    alpha_tau = copy.deepcopy(alpha_k)
    p_tau = copy.deepcopy(p_k)
    p_tau = np.array([p_tau[i]*(1.- tau) for i in range(len(p_tau))]).reshape((len(p_tau),))
    alpha_tau = np.vstack((alpha_tau , alpha))
    p_tau = np.hstack((p_tau, 1. * tau))
    return [alpha_tau, p_tau]
# матрица для минимизации по тау
def tau_minimize(ksi):
    M = np.zeros((s, s), dtype=float)
    for i in range(0, len(ksi[0])):
        M += lr1.single_point_FIM(ksi[0][i][0], ksi[0][i][1], ksi[1][i])
    return M
# минимизация по тау (двойственный градиент)
def minimize_TAU(alpha_k, p_k, alpha):
    fun = lambda tau: lr1.criterion_A_optimality(tau_minimize(make_ksi_tau(alpha_k, p_k, alpha, tau)))
    tau0 = np.random.uniform(0, 1, 1)
    res = minimize(fun, tau0, method='TNC', bounds=[[0, 1]])
    return res.x

# нахождение в массиве X индекса элемента, который близок к элементу x
def find_close(x, X):
    delta = 1e-2
    for i in range(len(X)):
        vec = np.array([x[0] - X[i][0], x[1] - X[i][1]])
        scal = np.dot(vec, vec)
        if np.sqrt(scal) < delta:
            return i
    return -1
# объединение близких точек
def union_сlose_point(alpha, p):
    newAlpha = [alpha[0]]
    newP = [p[0]]
    for i in range(1, len(alpha)):
        index = find_close(alpha[i], newAlpha)
        if index == -1:
            newAlpha.append(alpha[i])
            newP.append(p[i])
        else:
            newP[index] += p[i]
    alpha = newAlpha
    p = newP
    return np.array(alpha), np.array(p)


# удаление точек с малыми весами
def remove_point(alpha, p):
    delta = 0.15
    sum = 0
    index = 0
    for i in range(len(p)):
        if p[i] < delta:
            sum += p[i]
            p[i] = 0
            alpha[i] = [0, 0]
            index += 1
    for i in range(index):
        p = np.delete(p, np.where(p == 0)[0], 0)
        alpha = np.delete(alpha, np.where(alpha== [0,0])[0], 0)
    sum /= len(p)
    p = np.array([p[i] + sum for i in range(len(p))])
    return alpha, p

def dual_gradient_proc(alpha_0, p_0):
    delta = 1e-3
    iter = 1
    maxiter = 20
    flag = True
    # начальный план
    alpha_k = alpha_0
    p_k = p_0

    while iter <= maxiter and flag:
        M = lr1.the_normalized_FIM(alpha_k, p_k)
        D = np.linalg.inv(M)
        D2 = np.matmul(D, D)

        # Поиск локального максимума мю
        mu, alpha = maximize_mu(D2)
        eta = lr1.criterion_A_optimality(M)
        print("Condition", abs(abs(mu) - eta))
        if abs(abs(mu) - eta) <= delta:
            print("Plane was found")
            flag = False
        elif abs(mu) > eta:
            #Вычисляем тау_к
            tau_k = minimize_TAU(alpha_k, p_k, alpha)
            alpha_k, p_k = make_ksi_tau(alpha_k, p_k, alpha, tau_k[0])
            # поиск близких точек
            alpha_k, p_k = union_сlose_point(alpha_k, p_k)
            # удаление точек с малыми весами
            alpha_k, p_k = remove_point(alpha_k, p_k)
            iter += 1
        alpha_k, p_k = np.round(alpha_k, 5), np.round(p_k, 5)
    return alpha_k, p_k

def dual_gradient(start_x, start_p):
    import matplotlib.pyplot as plt
    fout = open("dual_results.txt", "w")
    n = int(round(float((s + 1) * s) / 2 + 1)) 
    reseach_count = 1
    for i in range(0, reseach_count):
        fout.write("Начальный план:\n" + str(start_x) + "\n" + str(start_p) + "\n")
        plot_x = [start_x[i][0] for i in range(0, n)]
        plot_y = [start_x[i][1] for i in range(0, n)]
        plt.plot(plot_x, plot_y, 'ro')
        plt.show()

        M1 = lr1.the_normalized_FIM(start_x, start_p)
        fout.write("Марица Фишера:\n" + str(M1) + "\n")
        A1 = lr1.criterion_A_optimality(M1)
        fout.write("Значение критерия А-оптимальности: " + str(A1) + "\n")

        x, p = dual_gradient_proc(start_x, start_p)

        fout.write("Конечный план:\n[")
        for i in range(0, len(x)):
            fout.write(str(x[i]))
            if i != len(x) - 1:
                fout.write(", ")

        fout.write("]\n" + str(p) + "\n")

        plot_x = [x[i][0] for i in range(0, len(x))]
        plot_y = [x[i][1] for i in range(0, len(x))]
        plt.plot(plot_x, plot_y, 'ro')
        plt.show()

        M2 = lr1.the_normalized_FIM(x, p)
        fout.write("Марица Фишера:\n" + str(M2) + "\n")
        A2 = lr1.criterion_A_optimality(M2)
        fout.write("Значение критерия А-оптимальности: " + str(A2) + "\n\n\n")
    fout.close()
