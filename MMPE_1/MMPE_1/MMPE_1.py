import numpy as np
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
#количество неизв параметров
s = 2
#количество точек плана
q = int(round(float((s + 1) * s) / 2 + 1)) 

#генерация начального плана
def create_start_plan(filename="ksi_0.txt"):
    alpha_0 = np.array([np.round(np.random.uniform(-1, 1, 2), 5) for i in range(q)])
    p_0 = np.array([np.round(1.0 / q, 5)] * q)
    np.savetxt(filename, np.hstack((alpha_0.ravel(), p_0)), fmt='%.5f', delimiter=' ')
# Загрузка начального плана из файла
def get_start_plan(filename="ksi_0.txt"):
    ksi_0 = np.loadtxt(filename, delimiter=' ')
    alpha_0, p_0 = ksi_0[:-q].reshape((q, s)), ksi_0[-q:]
    return alpha_0, p_0

#регрессионная модель 
def f(x1, x2):
    return np.array([[x1], [x2**2]])

#ИМФ для одноточечного плана
def single_point_FIM(x1, x2, p):
    return p * np.dot(f(x1, x2), f(x1, x2).T)
#ИМФ
def the_normalized_FIM(alpha, p):
    F = np.array([f(x[0], x[1]) for x in alpha])
    M = np.zeros((s, s), dtype=float)
    for fun, p_ in zip(F, p):
        M += p_ * np.dot(fun, fun.T)
    return M

# матрица для минимизации 
def minimize_FIM(tmp, fix_tmp, flag):
    M = np.zeros((s, s), dtype=float)
    for i in range(q):
        if flag == 0:
            M += single_point_FIM(tmp[2*i], tmp[2*i + 1], fix_tmp[i])
        else:
            M += single_point_FIM(tmp[i][0], tmp[i][1], fix_tmp[i])
    return M
# минимизация по альфа (спектру плана)
def minimize_alpha(x0, p, fix_p):
    func = lambda x: criterion_D_optimality(minimize_FIM(x, fix_p, 0))
    fix_p = p
    res = minimize(func, x0.ravel(), method='TNC', bounds=[[-1, 1]] * (2 * q))
    return res.x.reshape((q, s))
 
# минимизация по р
def minimize_p(x, p0, fix_x):
    func = lambda p: criterion_D_optimality(minimize_FIM(fix_x, p, 1))
    fix_x = x
    res = minimize(func, p0,method='TNC', bounds=[[0, 1]]*q)
    sum = np.sum(res.x)
    #нормируем веса
    return res.x * (1./sum)

# Вычисления критерия А-оптимальности
def criterion_A_optimality(M):
    M_inv = np.linalg.inv(M)
    return np.trace(M_inv)
# Вычисления критерия D-оптимальности
def criterion_D_optimality(M):
    M_det = np.linalg.det(M)
    return - np.log(M_det)

# прямая градиентная процедура
def direct_gradient_proc(alpha_0, p_0):
    fix_alpha = []
    fix_p = []
    delta_e = 1e-3 
    flag = False
    iter = 0
    maxiter = 5 
    while not flag and iter <= maxiter:
        iter = 1
        print("Iter: ", iter)
        eps = 1.0
        # шаг 1. Выбор начального плана
        alpha_k = alpha_0
        p_k = p_0
        while eps > delta_e and iter <= maxiter:
            # Минимизация по спектру плана
            alpha_k1 = minimize_alpha(alpha_k, p_k, fix_p) 
            # Минимизация по весам плана
            p_k1 = minimize_p(alpha_k1, p_k, fix_alpha)  
            # Вычисление суммы квадратов отклонений спектра и весов
            # плана от предыдущей итерации
            dif_a = (p_k1 - p_k)**2
            dif_p = np.linalg.norm(alpha_k1 - alpha_k)
            eps = np.sum(dif_a + dif_p)
            alpha_k = alpha_k1
            p_k = p_k1
            iter += 1 
        print("alpha_new: ", alpha_k)
        print("P_new: ", p_k)
        M_new = the_normalized_FIM(alpha_k, p_k)
        #обратная ИМФ
        D_new = np.linalg.inv(M_new)
        #D2 = np.matmul(D_new, D_new)
        #необходимое условие оптимальности
        #для А-оптимального плана вычисление эты
        #eta = criterion_A_optimality(M_new) 
        #необходимое условие оптимальности
        #для D-оптимального плана вычисление эты
        eta = s
        flag = True
        for a_k in alpha_k:
            mu = np.trace(np.dot(D_new, single_point_FIM(a_k[0], a_k[1], 1.)))
            if abs(mu - eta) > delta_e:
               flag = False
               print("Discrepancy: ", abs(mu - eta))
               print(alpha_k, mu, eta)
               print(M_new)
               print(D_new) 
        if flag:
            print("Plan was found")
        else:
            print("Plan is not found!")
            return 0, 0
        iter += 1
 
    return alpha_k, p_k
 
def direct_gradient(alpha_0, p_0):
    with open("results.txt", "w") as file:
        for i in range(0, 1):
            file.write("Начальное приближение:\n" + str(alpha_0) + "\n" + str(p_0) + "\n\n")
            plot_x = [alpha_0[i][0] for i in range(q)]
            plot_y = [alpha_0[i][1] for i in range(q)]
            plt.plot(plot_x, plot_y, 'ro')
            plt.show()
 
            M1 = the_normalized_FIM(alpha_0, p_0)
            file.write("ИМФ:\n" + str(M1) + "\n\n")
            D1 = criterion_D_optimality(M1)
            file.write("Значение критерия D-оптимальности: " + str(D1) + "\n\n")
 
            alpha, p = direct_gradient_proc(alpha_0, p_0)
            file.write("Оптимальный план:\n" + str(alpha) + "\n" + str(p) + "\n\n")
            plot_x = [alpha[i][0] for i in range(q)]
            plot_y = [alpha[i][1] for i in range(q)]
            plt.plot(plot_x, plot_y, 'ro')
            plt.show()
 
            M2 = the_normalized_FIM(alpha, p)
            file.write("ИМФ:\n" + str(M2) + "\n\n")
            D2 = criterion_D_optimality(M2)
            file.write("Значение критерия D-оптимальности: " + str(D2) + "\n\n\n")
if __name__ == "__main__":
    #create_start_plan()
    alpha_0, p_0 = get_start_plan()
    direct_gradient(alpha_0, p_0)

