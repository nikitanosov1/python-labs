import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Union
import numpy as np
import random


class Regression:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Regression class is static class")

    @staticmethod
    def rand_in_range(rand_range: Union[float, Tuple[float, float]] = 1.0) -> float:
        if isinstance(rand_range, float):
            return random.uniform(-0.5 * rand_range, 0.5 * rand_range)
        if isinstance(rand_range, tuple):
            return random.uniform(rand_range[0], rand_range[1])
        return random.uniform(-0.5, 0.5)

    @staticmethod
    def test_data_along_line(k: float = 1.0, b: float = 0.1, arg_range: float = 1.0,
                             rand_range: float = 0.05, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует линию вида y = k * x + b + dy, где dy - аддитивный шум с амплитудой half_disp
        :param k: наклон линии
        :param b: смещение по y
        :param arg_range: диапазон аргумента от 0 до arg_range
        :param rand_range: диапазон шума данных
        :param n_points: количество точек
        :return: кортеж значений по x и y
        """
        x_step = arg_range / (n_points - 1)
        return np.array([i * x_step for i in range(n_points)]),\
            np.array([i * x_step * k + b + Regression.rand_in_range(rand_range)
                     for i in range(n_points)])

    @staticmethod
    def second_order_surface_2d(surf_params:
                                Tuple[float, float, float, float, float, float] = (
                                    1.0, -2.0, 3.0, 1.0, 2.0, -3.0),
                                args_range: float = 1.0, rand_range: float = .1, n_points: int = 1000) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует набор тестовых данных около поверхности второго порядка.
        Уравнение поверхности:
        z(x,y) = a * x^2 + x * y * b + c * y^2 + d * x + e * y + f
        :param surf_params: 
        :param surf_params [a, b, c, d, e, f]:
        :param args_range x in [x0, x1], y in [y0, y1]:
        :param rand_range:
        :param n_points:
        :return:
        """
        x = np.array([Regression.rand_in_range(args_range)
                     for _ in range(n_points)])
        y = np.array([Regression.rand_in_range(args_range)
                     for _ in range(n_points)])
        dz = np.array([surf_params[5] + Regression.rand_in_range(rand_range)
                      for _ in range(n_points)])
        return x, y, surf_params[0] * x * x + surf_params[1] * y * x + surf_params[2] * y * y + \
            surf_params[3] * x + surf_params[4] * y + dz

    @staticmethod
    def test_data_2d(kx: float = -2.0, ky: float = 2.0, b: float = 12.0, args_range: float = 1.0,
                     rand_range: float = 1.0, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует плоскость вида z = kx*x + ky*x + b + dz, где dz - аддитивный шум в диапазоне rand_range
        :param kx: наклон плоскости по x
        :param ky: наклон плоскости по y
        :param b: смещение по z
        :param args_range: диапазон аргументов по кажой из осей от 0 до args_range
        :param rand_range: диапазон шума данных
        :param n_points: количество точек
        :returns: кортеж значенией по x, y и z
        """
        x = np.array([Regression.rand_in_range(args_range)
                     for _ in range(n_points)])
        y = np.array([Regression.rand_in_range(args_range)
                     for _ in range(n_points)])
        dz = np.array([b + Regression.rand_in_range(rand_range)
                      for _ in range(n_points)])
        return x, y, x * kx + y * ky + dz

    def test_data_nd(k: np.ndarray = [1, 2, 3], b: float = 12, dim=3, half_disp: float = 1.01, n: int = 100):
        points = np.asarray([np.random.randn(n) for _ in range(dim)])

        f = []
        for row in range(n):
            res = 0
            for i in range(dim):
                res += k[i] * points[i, row]
            f.append(res + b + np.random.normal(scale=half_disp))

        data_rows = []
        for row in range(n):
            curr = []
            for i in range(dim):
                curr.append(points[i, row])
            curr.append(f[row])
            data_rows.append(curr)

        return np.asarray(data_rows)

    @staticmethod
    def distance_sum(x: np.ndarray, y: np.ndarray, k: float, b: float) -> float:
        """
        Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b при фиксированных k и b
        по формуле: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
        :param x: массив значений по x
        :param y: массив значений по y
        :param k: значение параметра k (наклон)
        :param b: значение параметра b (смещение)
        :returns: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5
        """
        return np.sqrt(np.power((y - x * k - b), 2.0).sum())

    @staticmethod
    def distance_field(x: np.ndarray, y: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b, где k и b являются диапазонами
        значений. Формула расстояния для j-ого значения из набора k и k-ого значения из набора b:
        F(k_j, b_k) = (Σ(yi -(k_j * xi + b_k))^2)^0.5 (суммирование по i)
        :param x: массив значений по x
        :param y: массив значений по y
        :param k: массив значений параметра k (наклоны)
        :param b: массив значений параметра b (смещения)
        :returns: поле расстояний вида F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
        """
        return np.array([[Regression.distance_sum(x, y, k_i, b_i) for k_i in k.flat] for b_i in b.flat])

    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Линейная регрессия.\n
        Основные формулы:\n
        yi - xi*k - b = ei\n
        yi - (xi*k + b) = ei\n
        (yi - (xi*k + b))^2 = yi^2 - 2*yi*(xi*k + b) + (xi*k + b)^2 = ei^2\n
        yi^2 - 2*(yi*xi*k + yi*b) + (xi^2 * k^2 + 2 * xi * k * b + b^2) = ei^2\n
        yi^2 - 2*yi*xi*k - 2*yi*b + xi^2 * k^2 + 2 * xi * k * b + b^2 = ei^2\n
        d ei^2 /dk = - 2*yi*xi + 2 * xi^2 * k + 2 * xi * b = 0\n
        d ei^2 /db = - 2*yi + 2 * xi * k + 2 * b = 0\n
        ====================================================================================================================\n
        d ei^2 /dk = (yi - xi * k - b) * xi = 0\n
        d ei^2 /db =  yi - xi * k - b = 0\n
        ====================================================================================================================\n
        Σ(yi - xi * k - b) * xi = 0\n
        Σ yi - xi * k - b = 0\n
        ====================================================================================================================\n
        Σ(yi - xi * k - b) * xi = 0\n
        Σ(yi - xi * k) = n * b\n
        ====================================================================================================================\n
        Σyi - k * Σxi = n * b\n
        Σxi*yi - xi^2 * k - xi*b = 0\n
        Σxi*yi - Σxi^2 * k - Σxi*b = 0\n
        Σxi*yi - Σxi^2 * k - Σxi*(Σyi - k * Σxi) / n = 0\n
        Σxi*yi - Σxi^2 * k - Σxi*Σyi / n + k * (Σxi)^2 / n = 0\n
        Σxi*yi - Σxi*Σyi / n + k * ((Σxi)^2 / n - Σxi^2)  = 0\n
        Σxi*yi - Σxi*Σyi / n = -k * ((Σxi)^2 / n - Σxi^2)\n
        (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n) = k\n
        окончательно:\n
        k = (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n)\n
        b = (Σyi - k * Σxi) /n\n
        :param x: массив значений по x
        :param y: массив значений по y
        :returns: возвращает пару (k, b), которая является решением задачи (Σ(yi -(k * xi + b))^2)->min
        """
        x_sum = x.sum()
        y_sum = y.sum()
        xy_sum = (x * y).sum()
        xx_sum = (x * x).sum()
        inv_n = 1.0 / x.size

        # sum_xi_multiply_yi = 0
        # sum_xi = 0
        # sum_yi = 0
        # sum_squared_xi = 0
        # for i in range(n):
        #     sum_xi_multiply_yi += x[i] * y[i]
        #     sum_xi += x[i]
        #     sum_yi += y[i]
        #     sum_squared_xi += x[i] * x[i]
        # k = (sum_xi_multiply_yi - sum_xi * sum_yi / n) / \
        #     (sum_squared_xi - (sum_xi) ** 2 / n)
        # b = (sum_yi - k * sum_xi) / n
        k = (xy_sum - x_sum * y_sum * inv_n) / (xx_sum - x_sum * x_sum * inv_n)
        b = (y_sum - k * x_sum) * inv_n
        return k, b

    @staticmethod
    def bi_linear_regression(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
        """
        Билинейная регрессия.\n
        Основные формулы:\n
        zi - (yi * ky + xi * kx + b) = ei\n
        zi^2 - 2*zi*(yi * ky + xi * kx + b) + (yi * ky + xi * kx + b)^2 = ei^2\n
        ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + ((yi*ky)^2 + 2 * (xi*kx*yi*ky + b*yi*ky) + (xi*kx + b)^2)\n
        ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx + b)^2\n
        ei^2 =\n
        zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
        ei^2 =\n
        zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
        ei^2 =\n
        zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b + b^2\n
        ====================================================================================================================\n
        d Σei^2 /dkx = Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b = 0\n
        d Σei^2 /dky = Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi = 0\n
        d Σei^2 /db  = Σ-zi + yi*ky + xi*kx = 0\n
        ====================================================================================================================\n
        d Σei^2 /dkx / dkx = Σ xi^2\n
        d Σei^2 /dkx / dky = Σ xi*yi\n
        d Σei^2 /dkx / db  = Σ xi\n
        ====================================================================================================================\n
        d Σei^2 /dky / dkx = Σ xi*yi\n
        d Σei^2 /dky / dky = Σ yi^2\n
        d Σei^2 /dky / db  = Σ yi\n
        ====================================================================================================================\n
        d Σei^2 /db / dkx = Σ xi\n
        d Σei^2 /db / dky = Σ yi\n
        d Σei^2 /db / db  = n\n
        ====================================================================================================================\n
        Hesse matrix:\n
        || d Σei^2 /dkx / dkx;  d Σei^2 /dkx / dky;  d Σei^2 /dkx / db ||\n
        || d Σei^2 /dky / dkx;  d Σei^2 /dky / dky;  d Σei^2 /dky / db ||\n
        || d Σei^2 /db  / dkx;  d Σei^2 /db  / dky;  d Σei^2 /db  / db ||\n
        ====================================================================================================================\n
        Hesse matrix:\n
                       | Σ xi^2;  Σ xi*yi; Σ xi |\n
        H(kx, ky, b) = | Σ xi*yi; Σ yi^2;  Σ yi |\n
                       | Σ xi;    Σ yi;    n    |\n
        ====================================================================================================================\n
                          | Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b |\n
        grad(kx, ky, b) = | Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi |\n
                          | Σ-zi + yi*ky + xi*kx                |\n
        ====================================================================================================================\n
        Окончательно решение:\n
        |kx|   |1|\n
        |ky| = |1| -  H(1, 1, 0)^-1 * grad(1, 1, 0)\n
        | b|   |0|\n

        :param x: массив значений по x
        :param y: массив значений по y
        :param z: массив значений по z
        :returns: возвращает тройку (kx, ky, b), которая является решением задачи (Σ(zi - (yi * ky + xi * kx + b))^2)->min
        """
        n = x.size
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_z = np.sum(z)
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_xy = np.sum(x*y)
        sum_xz = np.sum(x*z)
        sum_yz = np.sum(y*z)

        A = np.array([[sum_x2, sum_xy, sum_x],
                      [sum_xy, sum_y2, sum_y],
                      [sum_x, sum_y, n]])

        B = np.array([sum_xz, sum_yz, sum_z])

        return tuple((np.linalg.inv(A) @ B).flat)

        # kx, ky, b = np.linalg.solve(A, B)

        # return kx, ky, b

    @staticmethod
    def n_linear_regression(data_rows: np.ndarray) -> np.ndarray:
        """
        H_ij = Σx_i * x_j, i in [0, rows - 1] , j in [0, rows - 1]
        H_ij = Σx_i, j = rows i in [rows, :]
        H_ij = Σx_j, j in [:, rows], i = rows

               | Σkx * xi^2    + Σky * xi * yi + b * Σxi - Σzi * xi|\n
        grad = | Σkx * xi * yi + Σky * yi^2    + b * Σyi - Σzi * yi|\n
               | Σyi * ky      + Σxi * kx                - Σzi     |\n

        x_0 = [1,...1, 0] =>

               | Σ xi^2    + Σ xi * yi - Σzi * xi|\n
        grad = | Σ xi * yi + Σ yi^2    - Σzi * yi|\n
               | Σxi       + Σ yi      - Σzi     |\n

        :param data_rows:  состоит из строк вида: [x_0,x_1,...,x_n, f(x_0,x_1,...,x_n)]
        :return:
        """
        s_rows, s_cols = data_rows.shape

        hessian = np.zeros((s_cols, s_cols,), dtype=float)

        grad = np.zeros((s_cols,), dtype=float)

        x_0 = np.zeros((s_cols,), dtype=float)

        for row in range(s_cols - 1):
            x_0[row] = 1.0
            for col in range(row + 1):
                value = np.sum(data_rows[:, row] @ data_rows[:, col])
                hessian[row, col] = value
                hessian[col, row] = value

        for i in range(s_cols):
            value = np.sum(data_rows[:, i])
            hessian[i, s_cols - 1] = value
            hessian[s_cols - 1, i] = value

        hessian[s_cols - 1, s_cols - 1] = data_rows.shape[0]

        for row in range(s_cols - 1):
            grad[row] = np.sum(hessian[row, 0: s_cols - 1]) - \
                np.dot(data_rows[:, s_cols - 1], data_rows[:, row])

        grad[s_cols - 1] = np.sum(hessian[s_cols - 1, 0: s_cols - 1]
                                  ) - np.sum(data_rows[:, s_cols - 1])

        return x_0 - np.linalg.inv(hessian) @ grad

    @staticmethod
    def poly_regression(x: np.ndarray, y: np.ndarray, order: int = 5) -> np.ndarray:
        """
        Полином: y = Σ_j x^j * bj\n
        Отклонение: ei =  yi - Σ_j xi^j * bj\n
        Минимизируем: Σ_i(yi - Σ_j xi^j * bj)^2 -> min\n
        Σ_i(yi - Σ_j xi^j * bj)^2 = Σ_iyi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2\n
        условие минимума:\n d/dbj Σ_i ei = d/dbj (Σ_i yi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2) = 0\n
        :param x: массив значений по x
        :param y: массив значений по y
        :param order: порядок полинома
        :return: набор коэффициентов bi полинома y = Σx^i*bi
        """
        n = len(x)
        X = np.zeros((n, order + 1))
        Y = np.zeros(n)

        for i in range(n):
            for j in range(order + 1):
                X[i][j] = x[i] ** j
            Y[i] = y[i]

        A = np.dot(X.T, X)
        B = np.dot(X.T, Y)

        b = np.linalg.solve(A, B)

        return b

    @staticmethod
    def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        :param x: массив значений по x\n
        :param b: массив коэффициентов полинома\n
        :returns: возвращает полином yi = Σxi^j*bj\n
        """
        n = len(x)
        order = len(b) - 1
        y = np.zeros(n)

        for i in range(n):
            for j in range(order + 1):
                y[i] += x[i] ** j * b[j]
        return y

    @staticmethod
    def quadratic_regression_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray, order: int = 5) -> np.ndarray:
        # Первый способ (Не использую, криво работает)
        """
        https://math.stackexchange.com/questions/2572460/2d-polynomial-regression-with-condition 

        A = np.asarray([[xi ** (power - i) * yi ** i for power in range(order)
                       for i in range(power + 1)] for xi, yi in zip(x, y)])
        values = np.linalg.inv(A.T @ A) @ A.T @ z
        """
        # ВТОРОЙ СПОСОБ
        b = [x * x, x * y, y * y, x, y, np.array([1])]
        m = np.zeros((len(b), len(b)), dtype=float)
        d = np.zeros((6,), dtype=float)
        for rows in range(6):
            d[rows] = (b[rows] * z).sum()
            for cols in range(rows + 1):
                m[rows, cols] = (b[rows] * b[cols]).sum()
                m[cols, rows] = m[rows, cols]
        m[5, 5] = x.size
        values = np.linalg.inv(m) @ d
        return values

    @staticmethod
    def distance_field_example():
        """
        Функция проверки поля расстояний:\n
        1) Посчитать тестовыe x и y используя функцию test_data\n
        2) Задать интересующие нас диапазоны k и b (np.linspace...)\n
        3) Рассчитать поле расстояний (distance_field) и вывести в виде изображения.\n
        4) Проанализировать результат (смысл этой картинки в чём...)\n
        :return:
        """
        print("distance field test:")
        x, y = Regression.test_data_along_line()
        k_, b_ = Regression.linear_regression(x, y)
        print(f"y(x) = {k_:1.5} * x + {b_:1.5}\n")
        k = np.linspace(k_ - 2.0, k_ + 2.0, 128, dtype=float)
        b = np.linspace(b_ - 2.0, b_ + 2.0, 128, dtype=float)
        z = Regression.distance_field(x, y, k, b)
        plt.imshow(z, extent=[k.min(), k.max(), b.min(), b.max()])
        plt.plot(k_, b_, 'r*')
        plt.xlabel("k")
        plt.ylabel("b")
        plt.grid(True)
        plt.show()

    @staticmethod
    def linear_reg_example():
        """
        Функция проверки работы метода линейной регрессии:\n
        1) Посчитать тестовыe x и y используя функцию test_data\n
        2) Получить с помошью linear_regression значения k и b\n
        3) Вывести на графике x и y в виде массива точек и построить\n
           регрессионную прямую вида: y = k*x + b\n
        :return:
        """
        print("linear reg test:")
        x, y = Regression.test_data_along_line()
        print(x, y)
        k, b = Regression.linear_regression(x, y)

        plt.plot(x, y, '.g', label='Data Points')
        plt.plot(x, k * x + b, color='red', label='Regression Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    @staticmethod
    def bi_linear_reg_example():
        """
        Функция проверки работы метода билинейной регрессии:\n
        1) Посчитать тестовыe x, y и z используя функцию test_data_2d\n
        2) Получить с помошью bi_linear_regression значения kx, ky и b\n
        3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить\n
           регрессионную плоскость вида:\n z = kx*x + ky*y + b\n
        :return:
        """
        from matplotlib import cm
        x, y, z = Regression.test_data_2d()
        kx, ky, b = Regression.bi_linear_regression(x, y, z)
        # Create a meshgrid of x and y values
        x_grid, y_grid = np.meshgrid(x, y)
        # Compute the predicted z values using the regression equation
        z_pred = kx * x_grid + ky * y_grid + b
        # Create a 3D plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the data points
        ax.plot(x, y, z, '.r')  # , marker='o')
        # Plot the regression plane
        ax.plot_surface(x_grid, y_grid, z_pred,
                        cmap=cm.coolwarm)  # , alpha=0.5)
        # Set labels for x, y, and z axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # Show the plot
        plt.show()

    @staticmethod
    def poly_reg_example():
        """
        Функция проверки работы метода полиномиальной регрессии:\n
        1) Посчитать тестовыe x, y используя функцию test_data\n
        2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression\n
        3) Вывести на графике x и y в виде массива точек и построить\n
           регрессионную кривую. Для построения кривой использовать метод polynom\n
        :return:
        """
        print('\npoly regression test:')
        x, y = Regression.test_data_along_line()
        coefficients = Regression.poly_regression(x, y)
        y_ = Regression.polynom(x, coefficients)

        k, b = Regression.linear_regression(x, y)
        plt.plot(x, y, 'og', label='Data Points')
        plt.plot(x, y_, color='red', label='PolyRegression Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    @staticmethod
    def test_date_nd(surf_params: np.ndarray = np.array([1, 2, 3, 4, 5, 6, 10000]),
                     arg_range: float = 10, rand_range: float = 0.05, n_points: int = 100) -> np.ndarray:
        data = np.zeros((n_points, surf_params.size))
        import random
        for i in range(surf_params.size-1):
            data[:, i] = np.array(
                [random.uniform(-0.5*arg_range, 0.5*arg_range) for _ in range(n_points)])
            data[:, surf_params.size-1] += data[:, i] * surf_params[i]
        data[:, surf_params.size-1] += \
            np.array([surf_params[surf_params.size-1] + random.uniform(-0.5 *
                     rand_range, 0.5*rand_range) for _ in range(n_points)])
        return data

    @staticmethod
    def n_linear_reg_example():
        """
        Функция проверки работы метода регрессии произвольного размера:
        """
        k = [2, -3, 5, 6, 8]
        b = 5
        dim = len(k)
        data_rows = Regression.test_data_nd(k=k, dim=dim, b=b)
        pred = Regression.n_linear_regression(data_rows)
        print(
            f"f\t = {''.join([f'{ki:.2f} * x_{i} + ' for i, ki in enumerate(k)])}{b}")
        print(
            f"f_pred   = {''.join([f'{ki:.2f} * x_{i} + ' for i, ki in enumerate(pred[:-1])])}{pred[-1]}")

    @staticmethod
    def quadratic_reg_example():
        x, y, z = Regression.second_order_surface_2d(n_points=16)
        order = 5
        b = Regression.quadratic_regression_2d(x, y, z, order=order)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.plot(x, y, z, "*r")

        X, Y = np.meshgrid(x, y)

        Z_pred = b[0] * X * X + b[1] * X * Y + \
            b[2] * Y * Y + b[3] * X + b[4] * Y + b[5]

        ax.plot_surface(X, Y, Z_pred, cmap="Reds")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()


if __name__ == "__main__":
    Regression.distance_field_example()
    Regression.linear_reg_example()
    Regression.bi_linear_reg_example()
    Regression.n_linear_reg_example()
    Regression.poly_reg_example()
    Regression.quadratic_reg_example()
    pass
