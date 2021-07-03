import numpy as np
from Thruster_class_0_1_1  import *
import csv
import json
import os
# ------------------------------------------------------------------------------------
#             Main program
# ------------------------------------------------------------------------------------

#------------------------------Чтение файла-------------------------------------------
print(os.path.abspath(os.getcwd()))
print(os.path.dirname(os.path.abspath(__file__)))
input_filename = input("Введите входной файл: ")
print("Start")
# output_filename = input("Введите выходной файл: ")

def readJsonFile(filename):
    f_json = open(str(os.path.abspath(os.getcwd())) + "\\" + "balance_model\\" + filename, "r", encoding='utf8')
    return f_json.read()
data = readJsonFile(input_filename)
data_from_json = json.loads(data)
'''
Массовый расход
'''
mass_flow = np.arange(float(data_from_json["mass_flow"][0]), float(data_from_json["mass_flow"][1]), float(data_from_json["mass_flow"][2]))
# mass_flow = np.arange(0.05, 0.2, 0.05)  # массовый расход мг/сек
# mass_flow = np.arange(0.05, 0.2, 0.05)  # массовый расход мг/сек
print(mass_flow)
'''
Параметры разрядной камеры
'''
geometry_param = {}

if data_from_json["gtrk"]["type_id"] == "cylinder":
    diameter = float(data_from_json["gtrk"]["diameter"])  # внутренний диаметр разрядной камеры
    length = float(data_from_json["gtrk"]["length"])  # длина разрядной камеры
    geometry_param = {'Diameter': diameter, 'Length': length}

elif data_from_json["gtrk"]["type_id"] == "cone":
    diameter = float(data_from_json["gtrk"]["diameter"])  # внутренний диаметр разрядной камеры
    length = float(data_from_json["gtrk"]["length"]) # длина разрядной камеры
    min_diameter = float(data_from_json["gtrk"]["min_diameter"]) # минимальный диаметр конуса
    geometry_param = {'Diameter': diameter, 'Length': length, 'Min diameter': min_diameter}

elif data_from_json["gtrk"]["type_id"] == "hemisphere":
    diameter = float(data_from_json["gtrk"]["diameter"])  # внутренний диаметр разрядной камеры
    length = float(data_from_json["gtrk"]["length"]) # длина разрядной камеры
    radius_sphere = float(data_from_json["gtrk"]["radius"]) # радиус сферы
    geometry_param = {'Diameter': diameter, 'Length': length, 'Sphere Radius': radius_sphere}

elif data_from_json["gtrk"]["type_id"] == "hemisephere_cylinder":
    diameter = float(data_from_json["gtrk"]["diameter"])  # внутренний диаметр разрядной камеры
    length = float(data_from_json["gtrk"]["length"]) # длина разрядной камеры
    cylinder_part = float(data_from_json["gtrk"]["part"]) # цилиндрическая часть сферы
    geometry_param = {'Diameter': diameter, 'Length': length, 'Cylinder part': cylinder_part}
# radius_sphere = 35e-3  # радиус сферы 'Sphere Radius': radius_sphere
# min_diameter = 40.6E-3  # минимальный диаметр конуса'Min diameter': min_diameter # in geometry_param
# cylinder_part = 5E-3  # цилиндрическая часть сферы 'Cylinder part': cylinder_part # in geometry_param
# geometry_param = {'Diameter': diameter,\
#                   'Length': length}

print(geometry_param)
'''
Параметры электродов
'''

diameter_out = float(data_from_json["ios"]["output_diameter"]) # выходной диаметр эмиссионного электрода
scr_el_hole_diam = float(data_from_json["ios"]["scr_el_hole_diam"]) # диаметр отверстия эмиссионного электрода
acel_el_hole_diam = float(data_from_json["ios"]["acel_el_hole_diam"]) # диаметр отверстия ускоряющего электрода
scr_th = float(data_from_json["ios"]["scr_th"]) # толщина эмиссионного электрода
acel_th = float(data_from_json["ios"]["acel_th"]) # толщина ускоряющего электрода
grid_gap = float(data_from_json["ios"]["grid_gap"]) # межэлектродный зазор
bridge_len = float(data_from_json["ios"]["bridge_len"]) # перемычка эмиссионного электрода
jumper_th = float(data_from_json["ios"]["jumper_th"]) # толщина перемычки добавлено ИЛЬНУРОМ

grid_parameters = {'Hole diameter of screen electrode': scr_el_hole_diam, \
                   'Hole diameter of accel electrode': acel_el_hole_diam, \
                   'Total grid diameter': diameter_out, \
                   'Screen grid thickness': scr_th, \
                   'Accel grid thickness': acel_th, \
                   'Screen grid bridge length': bridge_len, \
                   'Intergrid gap': grid_gap, \
                   # 'Ion transparency': ion_transp,\
                   # 'Neutrals transparency': neutrals_transp,\
                   # 'Geometric transparency': transp_geom
                   }
'''
Параметры индуктора
'''
Rc = float(data_from_json["indk"]["thread_radius"])  # радиус катушки
num_turns = int(data_from_json["indk"]["number_of_turns"])  # число витков
Lcoil = float(data_from_json["indk"]["inductance"])  # индуктивность
freq = float(data_from_json["vchg"]["frequency"])  # частота
Rcoil = float(data_from_json["indk"]["resistance"])  # сопротивление сборки
Icoil = np.arange(float(data_from_json["vchg"]["coil_currents"][0]), float(data_from_json["vchg"]["coil_currents"][1]), float(data_from_json["vchg"]["coil_currents"][2]))  # ток
print('sizes', np.size(Icoil), np.size(mass_flow))
""" Массив расчетных мощностей."""
power_limit = [None]

""" Массив температур стенки ГРК."""
Tg1 = 273 + float(data_from_json["temp"]["Tg1"]) # для тока Icoil[0] из массива
Tg2 = 273 + float(data_from_json["temp"]["Tg2"]) # для тока Icoil[-1] из массива
dTg = np.ceil((Tg2 - Tg1) / (np.size(Icoil)))
Tg = np.arange(Tg1, Tg2, dTg) # массив температур
antenna_param = {'Frequency': freq, \
                 'Rcoil': Rcoil, \
                 'Lcoil': Lcoil, \
                 'Icoil': Icoil,\
                 'Num of turns': num_turns,\
                 'Coil radius': Rc}

thruster_GTNN = Thruster(geometry_param, grid_parameters, antenna_param, mass_flow, power_limit, Tg)

print('thruster', thruster_GTNN.filename)

# Столбец данных
columns = ("Electron density (1/m3);Neutral density (1/m3);Neutral temperature (K);Electron temperature (eV);Absorbed Power (W);Power losses (W);Ion current density (A/m2);Child-Langmuir limit (A/m2);Plasma resistance (Ohm);Plasma inductance (H);Elastic collision frequency;Total Power (W);RF Power (W);Beam current (mA);Uee (V);Thrust (mN);Thrust cost (W/mN);Gas efficiency;Specific impulse (sec);Mass Flow (mg/s);Coil current (A);Ionization losses (W);Excitation losses (W);Thermal losses (W);Wall particles losses (W);Area (m2);Volume (m3);Ion cost (W/mA);Total efficiency;Energy efficiency;Electron walls flux (W);Electron EE flux (W);Ion SE flux (W);Ion walls flux (W);Ions charge exchange flux (W);AE radiant flux (W);SE radiant flux (W);Walls radiant flux (W);Floating potential SE (V);Floating potential W (V);1Eq-1;1Eq-2;2Eq-1;2Eq-2;2Eq-3;2Eq-4;3Eq-1;3Eq-2;3Eq-3;Beam current (mA) (test);Screen ions current (mA);Walls ions current (mA);Walls electron current (mA);Screen electron current (mA);i-n mean free path (m);Walls temperature (K);Radial coefficient;Length coefficient").split(';')

common_index = 0 # общее кол-во столбцов
column_index = 0 # индекс строки где названия столбцов
common_array = [] # общий массив строк
# Cylinder (L 27.5, D 45.0)_2.0 MHz_5 turns_PL None W.csv

# чтение csv
with open(os.path.dirname(os.path.abspath(__file__)) + "\\" + str(thruster_GTNN.filename) , newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in spamreader:
        for column in columns:
            if column in row:
                column_index = common_index
                continue
        common_array.append(row)
        common_index += 1
array_with_column = {} # словарь с массивами для каждого отдельного столбца
for length in range(len(common_array[column_index])):
    # array_with_column.append([])
    array_with_column.update({common_array[column_index][length]: []})
# заполнение данными для каждого показателя
for col_index in range(len(common_array)):
    if col_index > column_index:
        for i in range(len(common_array[col_index])):
            # print('values', common_array[col_index][i])
            # array_with_column[common_array[column_index][i]].append(common_array[col_index][i])
            array_with_column[common_array[column_index][i]].append(np.format_float_scientific(float(common_array[col_index][i]), 2))

json_filename = str(thruster_GTNN.filename).split('.csv')[0] + '.json'
print('json_filename', json_filename)

# заполнение словаря
with open(os.path.dirname(os.path.abspath(__file__)) + "\\" + 'test.json', 'w') as f:
    json.dump(array_with_column, f)

if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "\\" + 'old_files'):
    print("Существует")
    with open(os.path.dirname(os.path.abspath(__file__)) + "\\" + 'old_files\\' + json_filename, 'w') as f:
        json.dump(array_with_column, f)
else:
    os.mkdir(os.path.dirname(os.path.abspath(__file__)) + "\\" + 'old_files')
    with open(os.path.dirname(os.path.abspath(__file__)) + "\\" + 'old_files\\' + json_filename, 'w') as f:
        json.dump(array_with_column, f)