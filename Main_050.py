import numpy as np
from Thruster_class_0_1_1  import *


# ------------------------------------------------------------------------------------
#             Main program
# ------------------------------------------------------------------------------------

'''
Массовый расход
'''
mass_flow = np.arange(0.05, 0.4, 0.05)  # массовый расход мг/сек

'''
Параметры разрядной камеры
'''
diameter = 45E-3  # внутренний диаметр разрядной камеры
length = 27.5E-3  # длина разрядной камеры

radius_sphere = 35e-3  # радиус сферы 'Sphere Radius': radius_sphere
min_diameter = 40.6E-3  # минимальный диаметр конуса'Min diameter': min_diameter # in geometry_param
cylinder_part = 5E-3  # цилиндрическая часть сферы 'Cylinder part': cylinder_part # in geometry_param
geometry_param = {'Diameter': diameter,\
                  'Length': length}
'''
Параметры электродов
'''
diameter_out = 50e-3 # выходной диаметр эмиссионного электрода
scr_el_hole_diam = 2.4E-3 # диаметр отверстия эмиссионного электрода
acel_el_hole_diam = 1.6E-3 # диаметр отверстия ускоряющего электрода
scr_th = 0.5E-3 # толщина эмиссионного электрода
acel_th = 1.5E-3 # толщина ускоряющего электрода
grid_gap = 0.80E-3 # межэлектродный зазор
bridge_len = 0.5E-3 # перемычка эмиссионного электрода

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
Rc = 25E-3  # радиус катушки
num_turns = 5.0  # число витков
Lcoil = 1.2E-6  # индуктивность
freq = [2e6]  # частота
Rcoil = [0.14]  # сопротивление сборки
Icoil = np.arange(4.0, 15.0, 2)  # ток

""" Массив расчетных мощностей."""
power_limit = [50, 75, 100, 125, 150]

""" Массив температур стенки ГРК."""
Tg1 = 273 + 70 # для тока Icoil[0] из массива
Tg2 = 273 + 401 # для тока Icoil[-1] из массива
dTg = np.ceil((Tg2 - Tg1) / (np.size(Icoil)))
Tg = np.arange(Tg1, Tg2, dTg) # массив температур

antenna_param = {'Frequency': freq[0], \
                 'Rcoil': Rcoil[0], \
                 'Lcoil': Lcoil, \
                 'Icoil': Icoil,\
                 'Num of turns': num_turns,\
                 'Coil radius': Rc}

thruster_GTNN = Thruster(geometry_param, grid_parameters, antenna_param, mass_flow, power_limit, Tg)
