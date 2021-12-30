import itertools
import numpy as np
import glob
import os
import time

from random import randint, random, choice
from functools import wraps
from numba import njit
from tqdm import tqdm


EXP = np.exp(1)

COLORS = [
    '#000080', '#00008B', '#0000CD', '#0000FF', '#00BFFF', '#00CED1',
    '#7B68EE', '#800000', '#A52A2A', '#BC8F8F', '#CD853F', '#D2691E',
    '#DAA520', '#F4A460', '#000000', '#808080', '#006400', '#00FA9A',
    '#00FF00', '#00FF7F', '#228B22', '#32CD32', '#66CDAA', '#90EE90',
    '#FF4500', '#FF8C00', '#FFA500', '#C71585', '#FF1493', '#FF69B4',
    '#FFC0CB', '#8B0000', '#DC143C', '#FA8072', '#FF0000', '#4B0082',
    '#800080', '#9400D3', '#BA55D3', '#DDA0DD', '#FFD700', '#FFFF00',
]
colors_list = COLORS.copy()


def choice_color():
    global colors_list

    if not colors_list:
        colors_list = COLORS.copy()

    color = choice(colors_list)
    colors_list.remove(color)
    return color


def timeit(method):
    """Декоратор для измерения времени работы функций"""
    @wraps(method)
    def timed(*args, **kw):
        ts = time.monotonic()
        result = method(*args, **kw)
        te = time.monotonic()
        s = te - ts
        print(f'{method.__name__}: {s:2.3f} s')
        return result
    return timed


def pi_monte_carlo(n_points):
    """Функция, считающая число Пи методом Монте-Карло"""
    internal_points = 0

    for i in range(n_points):
        x, y = random(), random()
        if x ** 2 + y ** 2 <= 1:
            internal_points += 1

    return 4 * internal_points / n_points


def create_matrix(n):
    """Функция которая возвращает двумерный массив со значениями +1"""
    result = np.array([[+1 for x in range(n)] for x in range(n)])
    return np.array(result)


def create_random_matrix(n):
    """Функция которая возвращает случайный двумерный массив"""
    values_spin = (+1, -1)
    result = np.array([[choice(values_spin) for x in range(n)] for x in range(n)])
    return np.array(result)


def get_matrix_indexes(matrix):
    """Функция, возвращающая все i и j индексы для итерации"""
    matrix_len = len(matrix)
    iter_obj = list(range(matrix_len))
    return list(itertools.product(iter_obj, repeat=2))


def matrix_to_file(matrix, path):
    """Сохраняет конфигурацию спина в текстовый файл"""
    index = get_matrix_indexes(matrix)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'w') as file:
        for item in index:
            file.write(f'{item[0]} {item[1]} 0 {matrix[item[0]][item[1]]}\n')


def str_to_print_matrix(matrix):
    """Функция которая возвращает строку для последующей печати конфигурации"""
    result = ''
    for j in matrix:
        for i in j:
            result += '+1' if i == 1 else str(i)
            result += ' '
        result += '\n'
    return result


def get_string_permutations(n):
    """Функция, возвращающая всевозможные конфигурации строки длины n из значений ('+1', '-1')"""
    data = (+1, -1)
    result = itertools.product(data, repeat=n)
    return list(result)


def get_matrix_permutations(n, numpy=True):
    """"Функция, возвращающая массив перебранных конфигураций спинов"""
    data_str = get_string_permutations(n)
    result = list(itertools.product(data_str, repeat=n))

    if not numpy:
        return result

    return np.array(result)


@njit
def get_list_energy(data, j_index=1):
    """Функция, возвращающая лист энергий для всех перебранных конфигураций"""
    n = len(data)
    data_energy = np.zeros(n)

    for i in range(n):
        result = system_energy(data[i], j_index)
        data_energy[i] = result
    return data_energy


def matrix_max_and_min_energy(data, j_index=1):
    """Функция, возвращающая 2 листа конфигураций, с минимальной и максимальной энергией соответсвенно,
    принимает всевозможные конфигурации"""

    data_energy = list(map(lambda x: system_energy(x, j_index), data))

    data_result_max = []
    data_result_min = []

    max_en, min_en = max(data_energy), min(data_energy)
    for index, item in enumerate(data_energy):
        if item == max_en:
            data_result_max.append(data[index])

        if item == min_en:
            data_result_min.append(data[index])

    return data_result_max, data_result_min


@njit
def get_spin_neighbors_value(i, j, matrix):
    """Функция, возвращающая кортеж значений соседей спина"""
    n = len(matrix)

    return matrix[(i + 1) % n][j], matrix[(i - 1) % n][j], matrix[i][(j + 1) % n], matrix[i][(j - 1) % n]


def get_spin_neighbors(i, j, matrix):
    """Функция, возвращающая кортеж соседей спина"""
    n = len(matrix)

    return ((i + 1) % n, j), ((i - 1) % n, j), (i, (j + 1) % n), (i, (j - 1) % n)


@njit
def system_energy(matrix, j_index=1):
    """Функция, возвращающая энергию системы"""
    n = len(matrix)
    spins_energy = 0
    for i in range(n):
        for j in range(n):
            spins_energy += spin_energy(i, j, matrix, j_index)

    return spins_energy / 2


@njit
def spin_energy(i, j, matrix, j_index=1):
    """"Функция, которая возвращает энергию спина"""
    neighbors = get_spin_neighbors_value(i, j, matrix)
    sum_neighbors = 0
    for item in neighbors:
        sum_neighbors += item * matrix[i][j]

    return -j_index * sum_neighbors


def random_spin_energy(matrix):
    """Функция, возвращающая энергию рандомного спина"""
    size_matrix = len(matrix) - 1
    i, j = randint(0, size_matrix), randint(0, size_matrix)
    return spin_energy(i, j, matrix)


@njit
def exp_therm(energy, min_energy, temperature, k_ratio=1):
    """Промежуточная функция, вычисляющая экспоненту для вычисления термодинамических величин"""
    return EXP ** (-1 * (energy - min_energy) / (temperature * k_ratio))


@njit
def get_list_exp_therm(data_energy, temperature, k_ratio=1):
    """Функция, возвращающая лист экспонент для вычисления термодинамических величин
     для всех перебранных конфигураций"""
    n = len(data_energy)
    min_energy = min(data_energy)
    list_therm = np.zeros(n)

    for i in range(n):
        result = exp_therm(data_energy[i], min_energy, temperature, k_ratio)
        list_therm[i] = result

    return list_therm


@njit
def statistical_sum(data_exp):
    """Функция, возвращает значение статистической суммы"""
    return sum(data_exp)


@njit
def get_list_probabilities(data_exp):
    """Функция, возвращяющая лист вероятностей, в котором может находится i-тая система"""
    z = statistical_sum(data_exp)
    n = len(data_exp)
    data_p = np.zeros(n)
    for i in range(n):
        result = data_exp[i] / z
        data_p[i] = result
    return data_p


@njit
def average_energy(data_energy, data_p):
    """Функция, возвращающая среднее значение энергии"""
    n = len(data_energy)
    result = 0
    for i in range(n):
        result += (data_energy[i] * data_p[i])

    return result


@njit
def average_energy_quad(data_energy, data_p):
    """Функция, возвращающая среднеквадратичное значение энергии"""
    n = len(data_energy)
    result = 0
    for i in range(n):
        result += (data_energy[i] * data_energy[i] * data_p[i])

    return result


@njit
def matrix_magnetization(matrix):
    """Функция, возвращающая значение намагниченности системы"""
    n = len(matrix)
    result = 0
    for i in range(n):
        for j in range(n):
            result += matrix[i][j]

    return result


@njit
def get_list_magnetization(data):
    """Функция, возвращяющая лист намагниченности конфигураций по модулю"""
    n = len(data)
    data_mg = np.zeros(n)
    for i in range(n):
        result = matrix_magnetization(data[i])
        data_mg[i] = abs(result)

    return data_mg


@njit
def average_magnetization(data_mg, data_p):
    """Функция, возвращающая среднее значение намагниченности"""
    n = len(data_mg)
    result = 0
    for i in range(n):
        result += (data_mg[i] * data_p[i])

    return result


@njit
def average_magnetization_quad(data_mg, data_p):
    """Функция, возвращающая среднеквадратичное значение намагниченности"""
    n = len(data_mg)
    result = 0
    for i in range(n):
        result += (data_mg[i] * data_mg[i] * data_p[i])

    return result


@njit
def specific_heat(n, energy, energy_quad, temperature, k_ratio=1):
    """Функция, возвращающая значение удельной теплоемкости"""
    result = (energy_quad - (energy ** 2)) / (k_ratio * n * n * temperature * temperature)
    return result


@njit
def magnetic_susceptibility(n, mg, mg_quad, temperature, k_ratio=1):
    """Функция, возвращающая значение магнитной восприимчивости"""
    result = (mg_quad - (mg ** 2)) / (k_ratio * n * n * temperature)
    return result


def thermodynamic_iteration(n, path, k_ratio=1, j_index=1):
    """Функция, выполняющая термодинамический перебор и сохраняющая данные в файл"""
    print(f"\033[31mStart for {n}x{n} \033[0m")

    path += f'/{n}x{n}.txt'
    data = get_matrix_permutations(n)
    print('get_matrix_permutations - ok')
    data_en = get_list_energy(data, j_index)
    print('get_list_energy - ok')
    data_mg = get_list_magnetization(data)
    print('get_list_magnetization - ok')
    n_ratio = n * n

    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'w') as file:
        for temperature in tqdm(np.arange(0.0001, 5.01, 0.01)):

            data_exp = get_list_exp_therm(data_en, temperature, k_ratio)
            data_p = get_list_probabilities(data_exp)

            en = average_energy(data_en, data_p)
            en2 = average_energy_quad(data_en, data_p)

            mg = average_magnetization(data_mg, data_p)
            mg2 = average_magnetization_quad(data_mg, data_p)

            c = specific_heat(n, en, en2, temperature, k_ratio)
            x = magnetic_susceptibility(n, mg, mg2, temperature, k_ratio)

            file.write(f'{temperature} {en / n_ratio} {mg / n_ratio} {c} {x}\n')
    print()


@njit
def exp_metropolis(old_energy, new_energy, temperature, k_ratio=1):
    """Функция вычисляющая экспоненту, в алгоритме Метрополиса"""
    delta = new_energy - old_energy
    return EXP ** (- delta / (k_ratio * temperature))


def metropolis_algorithm(n, steps, path, k_ratio=1):
    """Выполняет алгоритм Метрополиса"""

    print(f"\033[31mStart for {n}x{n} \033[0m")
    path += f'/{n}x{n}.txt'

    data = create_random_matrix(n)
    n_ratio = n * n

    # steps = steps * n * n

    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'w') as file:

        for temperature in tqdm(np.arange(5, 0, -0.01)):
            en = 0
            en2 = 0
            mg = 0
            mg2 = 0

            old_en = system_energy(data)

            for step in range(1, steps):

                old_en = system_energy(data)
                save_data = data

                i, j = randint(0, n - 1), randint(0, n - 1)
                data[i][j] *= -1

                new_en = system_energy(data)

                if new_en >= old_en:
                    f = exp_metropolis(old_en, new_en, temperature, k_ratio)
                    r = random()
                    if r >= f:
                        data[i][j] = data[i][j] * -1
                        data = save_data
                        new_en = old_en

                old_en = new_en
                new_m = abs(matrix_magnetization(data))

                en += (new_en - en) / step
                en2 += (new_en * new_en - en2) / step
                mg += (new_m - mg) / step
                mg2 += (new_m * new_m - mg2) / step

            c = specific_heat(n, en, en2, temperature, k_ratio)
            x = magnetic_susceptibility(n, mg, mg2, temperature, k_ratio)

            file.write(f'{temperature} {en / n_ratio} {mg / n_ratio} {c} {x}\n')
        print()
    pass


def cluster(matrix, seed):
    """Базовый кластерный алгоритм, возвращет список спинов"""
    seed_value = matrix[seed[0]][seed[1]]

    queue = [seed]
    cluster_result = [seed]

    while queue:
        spin = queue.pop(0)
        spin_neighbors = get_spin_neighbors(spin[0], spin[1], matrix)

        for neighbor in spin_neighbors:
            x, y = neighbor
            value = matrix[x][y]

            if value == seed_value and neighbor not in cluster_result:
                queue.append(neighbor)
                cluster_result.append(neighbor)

    return cluster_result


def cluster_to_file(matrix, seed, cluster_list, path):
    """Запись кластера по файлам"""
    n = len(matrix)

    path1 = path + f'/seed_cluster.txt'
    path2 = path + f'/cluster.txt'
    path3 = path + f'/config.txt'

    try:
        os.remove(path1)
        os.remove(path2)
        os.remove(path3)
    except FileNotFoundError:
        pass

    with open(path1, 'w') as file:
        file.write(f'{seed[0]} {seed[1]} 0 {matrix[seed[0]][seed[1]]}\n')

    with open(path2, 'w') as file:
        for spin in cluster_list:
            file.write(f'{spin[0]} {spin[1]} 0 {matrix[spin[0]][spin[1]]}\n')

    matrix_to_file(matrix, path3)


def wolf_cluster(matrix, seed, f):
    """Кластерный алгоритм Вольфа, возвращет список спинов"""
    n = len(matrix)

    seed_value = matrix[seed[0]][seed[1]]

    queue = [seed]
    cluster_result = [seed]

    while queue:
        spin = queue.pop(0)
        spin_neighbors = get_spin_neighbors(spin[0], spin[1], matrix)

        for neighbor in spin_neighbors:
            x, y = neighbor
            value = matrix[x][y]

            if value == seed_value and neighbor not in cluster_result:
                r = random()
                if r < f:
                    queue.append(neighbor)
                    cluster_result.append(neighbor)

    return cluster_result


def wolf_algorithm(n, steps, path, k_ratio=1):
    """Выполняет алгоритм Вольфа"""

    print(f"\033[31mStart for {n}x{n} \033[0m")
    path += f'/{n}x{n}.txt'

    data = create_random_matrix(n)
    n_ratio = n * n
    new_en = system_energy(data)

    # steps = steps * n * n

    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'w') as file:

        for temperature in tqdm(np.arange(5, 0, -0.01)):
            en = 0
            en2 = 0
            mg = 0
            mg2 = 0

            f = 1 - EXP ** (-2/temperature)

            for step in range(1, steps):

                seed = randint(0, n - 1), randint(0, n - 1)
                cluster_list = wolf_cluster(data, seed, f)

                if randint(0, 1) == 0:
                    for spin in cluster_list:
                        x, y = spin
                        data[x][y] = data[x][y] * -1
                    new_en = system_energy(data)

                new_m = abs(matrix_magnetization(data))

                en += (new_en - en) / step
                en2 += (new_en * new_en - en2) / step
                mg += (new_m - mg) / step
                mg2 += (new_m * new_m - mg2) / step

            c = specific_heat(n, en, en2, temperature, k_ratio)
            x = magnetic_susceptibility(n, mg, mg2, temperature, k_ratio)

            file.write(f'{temperature} {en / n_ratio} {mg / n_ratio} {c} {x}\n')
        print()


def cluster_markup(matrix):
    """Выявляет в конфигурации все кластеры и возвращает словарем"""
    queue = get_matrix_indexes(matrix)
    result = dict()
    index = 0

    while queue:
        seed = queue[0]
        cluster_list = cluster(matrix, seed)
        result[index] = cluster_list

        for spin in cluster_list:
            queue.remove(spin)
        index += 1

    return result


def cluster_markup_to_file(markup_dict, matrix, path):
    """Записывает все кластеры конфигурации в отдельные файлы и создаеи скрипт"""
    dict_len = len(markup_dict)
    n = len(matrix)

    for file in glob.glob(path + "/*"):
        os.remove(file)

    for index, value in markup_dict.items():
        path_file = path + f'/{index}.txt'

        with open(path_file, 'w') as file:
            for spin in value:
                file.write(f'{spin[0]} {spin[1]} 0 {matrix[spin[0]][spin[1]]}\n')

    script_file = path + '/script.gp'
    script_text = f'set terminal png font "Verdana,14" size 1000, 1000\n' \
                  f'set output "Cluster.png"\n' \
                  f'plot [-1:{n}][-1:{n}] '

    for i in range(dict_len):
        script_text += f"'{i}.txt' using ($1-($3/4)):($2-($4/4)):($3/2):($4/2) with vectors lw 3 " \
                       f"lc rgb '{choice_color()}' notitle"
        if i != dict_len - 1:
            script_text += ", "

    with open(script_file, 'w') as file:
        file.write(script_text)


def sv_cluster(matrix, seed, f, available_values):
    """Кластерный алгоритм Свендсена-Ванга, возвращет список спинов"""
    n = len(matrix)

    seed_value = matrix[seed[0]][seed[1]]

    queue = [seed]
    cluster_result = [seed]

    while queue:
        spin = queue.pop(0)
        spin_neighbors = get_spin_neighbors(spin[0], spin[1], matrix)

        for neighbor in spin_neighbors:
            x, y = neighbor
            value = matrix[x][y]

            if neighbor not in available_values:
                continue

            if value == seed_value and neighbor not in cluster_result:
                r = random()
                if r < f:
                    queue.append(neighbor)
                    cluster_result.append(neighbor)

    return cluster_result


def sv_cluster_markup(matrix, f):
    """Выявляет в конфигурации все кластеры с определенной вероятностью и возвращает словарем"""
    queue = get_matrix_indexes(matrix)
    result = dict()
    index = 0

    while queue:
        seed = queue[0]
        cluster_list = sv_cluster(matrix, seed, f, queue)
        result[index] = cluster_list

        for spin in cluster_list:
            queue.remove(spin)
        index += 1

    return result


def sv_algorithm(n, steps, path, k_ratio=1):
    """Выполняет алгоритм Свендсена-Ванга"""

    print(f"\033[31mStart for {n}x{n} \033[0m")
    path += f'/{n}x{n}.txt'

    data = create_random_matrix(n)
    n_ratio = n * n
    new_en = system_energy(data)

    # steps = steps * n * n
    # Отключено из-за коллебаний в графиках и большой разницы времени

    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'w') as file:

        for temperature in tqdm(np.arange(5, 0, -0.01)):
            en = 0
            en2 = 0
            mg = 0
            mg2 = 0

            f = 1 - EXP ** (-2/temperature)

            for step in range(1, steps):

                cluster_dict = sv_cluster_markup(data, f)

                for cluster_list in cluster_dict.values():
                    if randint(0, 1) == 0:
                        for spin in cluster_list:
                            x, y = spin
                            data[x][y] = data[x][y] * -1
                new_en = system_energy(data)
                new_m = abs(matrix_magnetization(data))

                en += (new_en - en) / step
                en2 += (new_en * new_en - en2) / step
                mg += (new_m - mg) / step
                mg2 += (new_m * new_m - mg2) / step

            c = specific_heat(n, en, en2, temperature, k_ratio)
            x = magnetic_susceptibility(n, mg, mg2, temperature, k_ratio)

            file.write(f'{temperature} {en / n_ratio} {mg / n_ratio} {c} {x}\n')
        print()


# Тестовые функции

def test_other():
    pass


def test_lab_1():
    test_data = [1000, 10000, 100000, 1000000, 10000000, 20000000, 50000000, 100000000]

    pi_monte_carlo_fast = njit()(pi_monte_carlo)
    for x in test_data:

        print(f'{x} - {pi_monte_carlo_fast(x)}')


def test_lab_2():
    n = 3
    data = get_matrix_permutations(n)

    str_out = ''
    if n > 2:
        str_out = f"Конфигурация 1:\n" \
                  f"{str_to_print_matrix(data[0])}" \
                  f"Конфигурация {len(data)}:\n" \
                  f"{str_to_print_matrix(data[-1])}"

        str_out += f"Колличество конфигураций: {len(data)} \n\n"

    else:
        for index, item in enumerate(data, 1):
            str_out += f"Конфигурация {index}: \n" \
                       f"{str_to_print_matrix(item)}"

        str_out += f"Колличество конфигураций: {len(data)} \n\n"

    print(str_out)

    try:
        os.remove('C:/Users/user/Desktop/mm/2/mat_model_test.txt')
    except FileNotFoundError:
        pass

    with open('C:/Users/user/Desktop/mm/2/mat_model_test.txt', 'w') as file:
        file.write(str_out)


def test_lab_3():
    test_data_1 = create_matrix(4)
    test_data_2 = create_matrix(5)

    print(f"Энергия случайного спина системы 4x4: {random_spin_energy(test_data_1)} \n"
          f"Энергия всей системы 4x4: {system_energy(test_data_1)}\n")

    print(f"Энергия случайного спина системы 5x5: {random_spin_energy(test_data_2)} \n"
          f"Энергия всей системы 5x5: {system_energy(test_data_2)}")
    pass


def test_lab_4():
    data_matrix = get_matrix_permutations(4, False)

    min_max_1 = matrix_max_and_min_energy(data_matrix, 1)
    test_data_max_1 = min_max_1[0][0]
    test_data_min_1 = min_max_1[1][0]

    min_max_2 = matrix_max_and_min_energy(data_matrix, -1)
    test_data_max_2 = min_max_2[0][0]
    test_data_min_2 = min_max_2[1][0]

    iterable = (test_data_max_1, test_data_min_1, test_data_max_2, test_data_min_2)
    name = ('1_max', '1_min', '-1_max', '-1_min')

    for index, item in enumerate(iterable):
        matrix_to_file(item, f'C:/Users/user/Desktop/mm/4/{name[index]}.txt')
    pass


def test_lab_5():
    for i in range(2, 6):
        thermodynamic_iteration(i, 'C:/Users/user/Desktop/mm/5')
    pass


def test_lab_6():
    for i in range(2, 6):
        metropolis_algorithm(i, 10000, 'C:/Users/user/Desktop/mm/6')
    pass


def test_lab_7():
    n = 10
    matrix = create_random_matrix(n)
    seed = randint(0, n - 1), randint(0, n - 1)
    cluster_list = cluster(matrix, seed)
    cluster_to_file(matrix, seed, cluster_list, 'C:/Users/user/Desktop/mm/7')
    pass


def test_lab_8():
    for i in range(2, 6):
        wolf_algorithm(i, 10000, 'C:/Users/user/Desktop/mm/8')


def test_lab_9():
    data = create_random_matrix(20)
    markup = cluster_markup(data)
    cluster_markup_to_file(markup, data, 'C:/Users/user/Desktop/mm/9')
    pass


def test_lab_10():
    for i in range(2, 6):
        sv_algorithm(i, 10000, 'C:/Users/user/Desktop/mm/10')
    pass

# Главная функция


def main():
    # test_other()

    # test_lab_1()
    # test_lab_2()
    # test_lab_3()
    # test_lab_4()
    # test_lab_5()
    # test_lab_6()
    # test_lab_7()
    # test_lab_8()
    test_lab_9()
    # test_lab_10()

    pass


if __name__ == "__main__":
    main()
