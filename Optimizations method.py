import numpy as np
import matplotlib.pyplot as plot
from scipy.linalg import solve

def f1(point):
    x, y = point 
    return 100 * (y - x) ** 2 + 5*(1 - x) ** 2

def f2(point):
    x, y = point
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def gradient1(point):
    x, y = point
    return np.array([210 * x - 200 * y - 10, 200 * y - 200 * x]) 

def gradient2(point):
    x, y = point
    return np.array([4 * x ** 3 + 4 * x * y - 42 * x + 2 * y ** 2 - 14, 4 * y ** 3 + 4 * x * y - 26 * y + 2 * x ** 2 - 22])  # Градиент функции f2 

def hessian1(point):
    x, y = point
    return [[210, -200], [-200, 200]]

def hessian2(point):
    x, y = point
    return [[12 * x ** 2 + 4 * y - 42, 4 * x + 4 * y],
            [4 * x + 4 * y, 12 * y ** 2 + 4 * x - 26]]
        

def nelder_mead(f, initial_simplex, alpha=1, beta=0.5, gamma=2, max_iter=1000, tol=1e-6):
    n = len(initial_simplex[0])
    simplex = initial_simplex

    for _ in range(max_iter):
        # Исходные вершины симплекса сортируются на основе значений их функций в порядке возрастания
        simplex = sorted(simplex, key=lambda x: f(x))

        # Вычисляется центр тяжести лучших (исключая наихудшие) вершин.
        centroid = [sum(point) / n for point in zip(*simplex[:-1])]

        # Точка отражения вычисляется на основе центроида, и если она улучшает значение функции, но не лучше второй наилучшей точки, она принимается.
        reflection = [centroid[i] + alpha * (centroid[i] - simplex[-1][i]) for i in range(n)]
        if f(simplex[0]) <= f(reflection) < f(simplex[-2]):
            simplex[-1] = reflection
        # Если отраженная точка является новой наилучшей, вычисляется точка расширения. Если точка расширения лучше, чем отраженная точка, она принимается; в противном случае принимается отраженная точка.
        elif f(reflection) < f(simplex[0]):
            expansion = [centroid[i] + gamma * (reflection[i] - centroid[i]) for i in range(n)]
            if f(expansion) < f(reflection):
                simplex[-1] = expansion
            else:
                simplex[-1] = reflection
        # Если отраженная точка хуже, чем вторая наихудшая точка, вычисляется точка сжатия. Если точка сжатия лучше, чем отраженная точка, она принимается; в противном случае весь симплекс сжимается в направлении наилучшей точки.
        else:
            contraction = [centroid[i] + beta * (simplex[-1][i] - centroid[i]) for i in range(n)]
            if f(contraction) < f(simplex[-1]):
                simplex[-1] = contraction
            else:
                # Шаг 6: Уменьшение
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]

        #  Сходимость проверяется путем сравнения значений функций вершин.
        if max([abs(f(x) - f(simplex[0])) for x in simplex]) <= tol:
            break

    return simplex[0]


initial_simplex = [[0, 0], [1, 0], [0, 1]]  # Начальный симплекс

print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Метод Нельдера - Мида:") 
print(f"Для функции f1: {nelder_mead(f1, initial_simplex)}, для функции f2: {nelder_mead(f2, initial_simplex)}")

def gradient_descent(f, gradient, initial_point, learning_rate=0.001, max_iterations=1000, tolerance=1e-6):
    current_point = np.array(initial_point)
    for iteration in range(max_iterations):
        grad = gradient(current_point)
        new_point = current_point - learning_rate * grad
        if np.abs(f(new_point) - f(current_point)) < tolerance:
            break

        current_point = new_point
    return current_point

initial_point = [1000, 1000]

print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Метод градиентного спуска:")
print(f"Для функции f1: {gradient_descent(f1, gradient1, initial_point)}, для функции f2: {gradient_descent(f2, gradient2, initial_point)}")


def conjugate_gradient_descent(f, grad_f, x0, tol=1e-6, max_iter=2000): # двигаемся в направлении, ортогональном всем предыдущим направлениям
    x = np.array(x0)
    gradient = grad_f(x)
    napravleniye = -gradient
    iteration = 0
    
    while np.linalg.norm(gradient) > tol and iteration < max_iter: # выполняет итерации до тех пор, пока норма градиента не станет меньше заданной точности
        alpha = line_search(f, grad_f, x, napravleniye) # На каждой итерации выполняется линейный поиск для определения оптимального шага alpha в направлении direction
        x = x + alpha * napravleniye
        new_gradient = grad_f(x)
        beta = np.dot(new_gradient, new_gradient) / np.dot(gradient, gradient)
        napravleniye = -new_gradient + beta * napravleniye
        gradient = new_gradient
        iteration += 1
    
    return x

def line_search(f, grad_f, x, direction):
    alpha = 1.0 # начальное значение шага
    c = 0.5 # константа в условии Армихо
    rho = 0.5 # коэффициент уменьшения шага
    
    while f(x + alpha * direction) > f(x) + c * alpha * np.dot(grad_f(x), direction): # проверка условия Армихо: новое значение функции вблизи ожидаемого шага alpha уменьшится по сравнению с текущим значением функции, учитывая градиент и константу c.
        alpha *= rho
    
    return alpha


print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Метод сопряженных градиентов:")
print(f"Для функции f1: {conjugate_gradient_descent(f1, gradient1, np.array([-10, -10]))}, для функции f2: {conjugate_gradient_descent(f2, gradient2, np.array([-10, -10]))}")

def newton_method(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):


    x_min = x0
    for _ in range(max_iter):
        grad = grad_f(x_min)
        hess = hess_f(x_min)
        direction = np.linalg.solve(hess, -grad) # Решаем систему линейных уравнений для нахождения направления спуска
        x_min = x_min + direction # Обновляем текущую точку
        if np.linalg.norm(grad) < tol: # Если норма градиента становится меньше заданной точности, алгоритм завершается
            break
    
    return x_min
print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Метод Ньютона второго порядка:")
print(f"Для функции f1: {newton_method(f1, gradient1, hessian1, np.array([1.0, 1.0]))}, для функции f2: {newton_method(f2, gradient2, hessian2, np.array([1.5, 1.5]))}")
print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")





