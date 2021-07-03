import numpy as np
import scipy.special as sp
from scipy.integrate import ode
from scipy import integrate
import csv
import datetime
import time
import re
import pandas as pd
import math
import itertools
import os
import time
import requests

class Thruster:
    """ Class Internal Constants.

    Parameters:
        m:  electron mass (kg)
        qe: electron charge ()
        kb: Boltzmann constant (J/K)
        c: light speed (m/s)
        aem: atomic mass unit
        mu_0: vacuum permeability (H/m)
        Eps0: permittivity of free space (m-3*kg-1*s4*A2 )
        alfa: average temperature set of thermal conductivity (Ti and SS)

    """
    m = 9.1E-31
    qe = 1.6E-19
    kb = 1.38e-23
    c = 299792458
    aem = 1.66E-27
    mu_0 = 4 * 3.14 * 1E-7
    Eps0 = 8.854E-12
    alfa = 4.15e-3

    # ------------------------------------------------------------------------------
    def __init__(self, geometry_param, grid_parameters, antenna_param, mass_flow, power_limit, Tg0):
        """ Исполняемый метод класса.

        Args:
            geometry_param {dict}: геомтерические парамтеры камеры
                'Diameter': (float) - диаметр перфорированной части ЭЭ (м)
                'Length': (float) - длина разрядной камеры (м)
                'Sphere Radius': (float) - радиус сферы разрядной камеры (м) (опционально, при наличии geometry_type = 'Sphere' )
                'Min diameter': (float) - меньший диаметр конуса (м) (опционально, при наличии geometry_type = 'Cone' )
                'Cylinder part': (float) - длина цилиндрической части разрядной камеры (м)(опционально, при наличии
                                                                                        geometry_type = 'SemiSphere' )

            grid_parameters {dict}: параметры электродов
                'Hole diameter of screen electrode': (float) - диаметр отверстий ЭЭ (м)
                'Hole diameter of  accel electrode': (float) - диаметр отверстий УЭ (м)
                'Total grid diameter': (float) - общий диаметр электрода ЭЭ (м)
                'Screen grid thickness': (float) - толщина ЭЭ (м)
                'Accel grid thickness': (float) - толщина УЭ (м)
                'Screen grid bridge length': (float) - расстояние между отверстиями ЭЭ (м)
                'Intergrid gap': (float) - межэлектродный зазор (м)

                'Ion transparency': (float) - прозрачность для ионов (опционально)
                'Neutrals transparency': (float) - прозрачность для нейтралов (УЭ) (опционально)
                'Geometric transparency': (float) - прозрачность геометрическая для ионов (ЭЭ) (опционально)


            antenna_param {dict}: параметры антенны
                'Frequency': (float) - частота ВЧ генератора
                'Rcoil': (float) - сопративление антенны и/или сборки двигателя
                'Lcoil': (float) - индуктивность антенны  и/или сборки двигателя
                'Icoil':(array_like) - массив токов антенны
                'Num of turns': (float) число витков антенны
                'Coil radius': (float) радиус витка антенны

            mass_flow (array_like): массив массовых расходов (мг/с)
            power_limit(array_like): предел по мощности (Вт)
            Tg0 (array_like): массив начальных температура газа равных температуре разрядной камеры (эксперимент)

        """

        """ Запись параметров из словарей в .self. """
        self.Tg_0 = Tg0
        self.mass_flow = mass_flow  # массовый расход (mg/c)
        self.D = geometry_param['Diameter']
        self.diameter_out = grid_parameters['Total grid diameter']
        self.grid_gap = grid_parameters['Hole diameter of screen electrode']
        self.scr_th = grid_parameters['Screen grid thickness']
        self.acel_th = grid_parameters['Accel grid thickness']
        self.freq = antenna_param['Frequency']
        self.Rcoil0 = antenna_param['Rcoil']
        self.Lcoil = antenna_param['Lcoil']
        self.I_coil = antenna_param['Icoil']
        self.num_turns = antenna_param['Num of turns']
        self.Rc = antenna_param['Coil radius']
        self.scr_el_hole_diam = grid_parameters['Hole diameter of screen electrode']
        self.acel_el_hole_diam = grid_parameters['Hole diameter of accel electrode']
        self.grid_parameters = grid_parameters
        self.geometry_param = geometry_param
        self.power_limit = power_limit
        self.antenna_param = antenna_param
        self.filename = ''
        print('\nStart calculation...')

        """ Вызов методов расчета параметров газа, геометрии и формирования имени файла.
         
        Parameters:
           self.clausing: коэффициент учета отражения нейтральных частиц обратно в разрядную камеру - clausing.py  
         
        """
        self.clausing = 0.17
        self.import_gas_data()
        self.geom_param()
        Q0 = mass_flow / (self.M * 1E6)  # удельный расход (1/c)

        """ Вызов метода расчета прозрачностей электродов 
       
         Conditions(if):
            Если grid_param{} уже содержит 'Ion transparency', 'Neutrals transparency', 'Geometry transparency':
                метод grid_parameters() - не вызывается
            
        """
        if self.grid_parameters.get('Ion transparency'):
            self.transp_ion = grid_parameters['Ion transparency']
            self.transp_neutral = grid_parameters['Neutrals transparency']
            self.transp_geom = grid_parameters['Geometric transparency']
        else:
            self.grid_param()
            self.grid_test()  # вызов метода вывода в консоль значений прозрачностей

        """ Вызов метода расчета параметров плазмы. """
        self.plasma_param(self.I_coil, Q0, self.Tg_0)

        """ Вызов метода расчета тепловых потоков.Невозможен без выозова метода plasma_param(). """
        self.heat_flux()

        """ Вызов метода расчета параметров двигателя. Невозможен без выозова метода plasma_param(). """
        for i in range(np.size(power_limit)):
            self.thruster_param(power_limit[i], mass_flow, self.I_coil)

        print('\nCalculation completed successfully!')  # Вывод в консоль сообщения об окончании расчета

    # ------------------------------------------------------------------------------
    def timing(f):
        """ Функция расчета времени исполнения метода. """
        def wrap(*args):
            time1 = time.time()
            ret = f(*args)
            time2 = time.time()
            print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
            return ret

        return wrap

    # ------------------------------------------------------------------------------
    def parse_file(self, filename = os.path.dirname(os.path.abspath(__file__)) + "\\" + 'Cross Section.txt'):
        """ Парсер файла со значеними сечений взаимодествия и константами рабочего газа

        Args:
            filename: имя импортируемого файла (по умолчанию 'Cross Section.txt')
            Файл должен содержать:
                Сечения взаимодействия
                - упругого взаимодействия электрона и нейтрала (таблица) - ELASTIC
                - ионизации (таблица, потенциал ионизации) - IONIZATION
                - возбуждения нейтрального атома (таблица, потенциал возбуждения) - EXCITATION
                                        если существует несколько уровней возбуждения, то берутся усредненное значения
                - упрогое ион-нейтрал взаимодействие (значение газокинетического сечения атома) - ELASTIC ION NEUTRAL
                К-т темплопроводности (W/K m) - THERMAL CONDUCTIVITY

            Файл должен иметь формат lxcat.net

        Parameters:
            FLOAT_REGEXP: регулярное выражение (РВ) для числа с плавующей точкой
            GAS_DEFINITION: РВ для импорта названия газа (формат lxcat.net)
            PROCESS: РВ для импорта процесса (таблица сечений или константы) - определяет строку с заглавнми буквами
            PARAM: константа для процесса (потенциал реакции либо сама константа)
            TABLE_DELIMITER: РВ определяющее конец таблицы
            TABLE_RAW: РВ определяющее формат таблицы данных (не используется)

        Returns:
            result: словарь с импортируемыми данными из файла

        """
        FLOAT_REGEXP = '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
        GAS_DEFINITION = re.compile(r'\*+ (\w+) \*+')
        PROCESS = re.compile(r'(?=^([A-Z0-9]+\s){0,3}[A-Z0-9]+$)(^(?:(?!FORMAT)(?![a-z]).)*?$)')
        PARAM = re.compile(fr'PARAM.:  (\S+) = ({FLOAT_REGEXP}).*')
        TABLE_DELIMITER = '-' * 29
        TABLE_RAW = re.compile(fr'\s+({FLOAT_REGEXP})\s+({FLOAT_REGEXP})')

        result = {}
        current_gas = None
        current_process = None
        current_data = None
        with open(filename, 'r') as f:
            for raw_line in f.readlines():
                line = raw_line.strip()
                if GAS_DEFINITION.match(line):
                    current_gas = {}
                    result[GAS_DEFINITION.match(line).group(1)] = current_gas
                elif PROCESS.match(line):
                    current_process = {}
                    current_gas[line] = current_process
                elif PARAM.match(line):
                    match = PARAM.match(line)
                    current_process['param'] = {match.group(1): float(match.group(2))}
                elif line == TABLE_DELIMITER:
                    if current_data:
                        current_process = None
                        current_data = None
                        continue
                    current_data = []
                    current_process['data'] = current_data
                elif current_data is not None:
                    values = line.split('\t')
                    current_data.append(list(map(float, values)))
        return result

    # ------------------------------------------------------------------------------
    def df_formation(self, raw_data):
        """ Метод создания pd.df из словаря с данными.

        Args:
            df_raw(dict): словарь со значениями сечений и констант из внешнего файла (Cross Section.txt)

        Returns:
            df: pd.df-таблица со значеними сечений взаимодействия выбранной реакции и энергии электрона

        """
        data = tuple(raw_data)
        df = pd.DataFrame(list(data))
        df.columns = ['Energy (eV)', 'Cross section (m2)']

        return df

    # ------------------------------------------------------------------------------
    def reaction_rate(self, data):
        """ Метод расчета констант сорости реакций.

        Args:
           data: pd.df-таблица со значениями сечений взаимодействия для выбранной реакции

        Parameters:
            Te_d: массив значений энергии электронов

        Returns:
             df(pd.DataFrame): таблица константант скорости реакции для массива Те - Te_d

        """
        Te_d = np.arange(0.001, 20, 0.1)
        Kr = np.zeros(np.size(Te_d))
        for i in range(np.size(Te_d)):
            Te = Te_d[i]
            k1 = np.sqrt(8 * self.qe * Te / (np.pi * self.m))
            func = lambda E: k1 * (E / Te) * np.exp(-(E / Te)) \
                             * np.interp((E), data['Energy (eV)'], data['Cross section (m2)']) / Te
            Kr[i] = integrate.quad(func, 0, np.inf)[0]
        df = pd.DataFrame({'Kin': Kr, 'Tev':Te_d})

        return df
    # ------------------------------------------------------------------------------
    @timing
    def import_gas_data(self):
        """ Импорт свойств рабочего газа.

        Метод записи констант и таблиц в .self

        Parameters:
            raw_data(dict) : словарь со значениями сечений и констант из внешнего файла (Cross Section.txt)
            self.M(float): масса атома (кг)
            self.Ui(float): потенциал ионизации (eВ)
            self.Uj(float): потенциал возбуждения атома (еВ)
            self.kt(float): температуропроводность газа (W/K*m)
            self.sigi(float): общее сечение соудараения нейтрала и иона, равно газокинетическому сечению атома
            self.exc_data(pd.DataFrame): таблица со значениями сейчений возбуждения и энергий электронов
            self.ion_data(pd.DataFrame): таблица со значениями сейчений ионизации и энергий электронов
            self.elc_data(pd.DataFrame): таблица со значениями сейчений упругого е-н взаимодействия и энергий электронов
            self.Kiz_data(pd.DataFrame): таблица со значеними констатн скорости ионизации и энергий электронов
            self.Kex_data(pd.DataFrame): таблица со значеними констатн скорости возбуждения и энергий электронов
            self.Kel_data(pd.DataFrame): таблица со значеними констатн скорости упр. удара и энергий электронов

        """
        raw_data = self.parse_file()
        self.gas_type = tuple(raw_data.keys())[0]
        self.Uj = tuple(raw_data[self.gas_type]['EXCITATION']['param'].values())[0]
        self.Ui = tuple(raw_data[self.gas_type]['IONIZATION']['param'].values())[0]
        self.M = self.m / tuple(raw_data[self.gas_type]['ELASTIC']['param'].values())[0]
        self.kt = tuple(raw_data[self.gas_type]['THERMAL CONDUCTIVITY']['param'].values())[0]
        self.sigi = tuple(raw_data[self.gas_type]['ELASTIC ION NEUTRAL']['param'].values())[0]

        self.exc_data = self.df_formation(raw_data[self.gas_type]['EXCITATION']['data'])
        self.ion_data = self.df_formation(raw_data[self.gas_type]['IONIZATION']['data'])
        self.elc_data = self.df_formation(raw_data[self.gas_type]['ELASTIC']['data'])

        self.Kiz_data = self.reaction_rate(self.ion_data)
        self.Kex_data= self.reaction_rate(self.exc_data)
        self.Kel_data = self.reaction_rate(self.elc_data)

    # ------------------------------------------------------------------------------
    def geom_param(self):
        """ Расчет геометрических параметров ГРК.

        Рассчитывается:

        Conditions(if):

            Если geometry_param{} содержит только ключи 'Length','Diameter','Total grid diameter' ():
                используется geometry_type = 'Cylinder'

                Parameters:
                    dop_area: дополнительная площадь ГРК (инжектор и т.д.)
                    self.A_r = Aw: площадь стенок цилиндра (для учета радиальных изменений)
                    self.As_total: площадь выходного отверстия цилиндра
                    self.V: полный объем цилиндра
                    self.A: полная площадь цилиндра
                    self.A_l: суммарная площадь торцевых поверхностей цилиндра (для учета осевых изменений)
                    self.As: суммарная площадь ЭЭ
                    self.A_dc: площадь поверхностей ГРК (без выходного отверстия)

            Если geometry_param{} содержит ключ 'Cylinder part':
                используется geometry_type = 'Semisphere'

                Parameters:
                    Aw_sphere: площадь стенок сферической части
                    Aw_cyl: площадь стенок цилиндрической части
                    self.A_l = self.As_total: площадь выходного отверстия цилиндра (для учета осевых изменений)
                    self.A: суммарная площадь цилиндра + полусферы
                    V_cyl: объем цилиндрической части
                    V_sphere: объем сферической части
                    self.V: суммарный объем ГРК
                    self.A_r: площадь стенок (для учета радиальных изменений)

            Если geometry_param{} содержит ключ 'Sphere Radius':
                 используется geometry_type = 'Sphere'
                  !!!дописать парамтеры!!!

            Если geometry_param{} содержит ключ 'Min diameter'(меньший диаметр конуса):
                используется geometry_type = 'Cone'
                     !!!дописать парамтеры!!!
                     :param geometry_param:
        """
        self.As = np.pi * ((self.diameter_out / 2) ** 2)
        self.R = self.D / 2

        if self.geometry_param.get('Length'):
            self.L = self.geometry_param['Length']
        if self.geometry_param.get('Diameter'):
            self.D = self.geometry_param['Diameter']
        self.geometry_type = 'Cylinder'

        if self.geometry_param.get('Min diameter'):
            self.d = self.geometry_param['Min diameter']
            self.geometry_type = 'Cone'
        else:
            self.d = False

        if self.geometry_param.get('Cylinder part'):
            self.cyl_l = self.geometry_param['Cylinder part']
            self.geometry_type = 'Semisphere + Cylinder'
        else:
            self.cyl_l = False

        if self.geometry_param.get('Sphere Radius'):
            self.R_sp = self.geometry_param['Sphere Radius']
            self.geometry_type = 'Semisphere'  # Запись типа геометрии в .self
        else:
            self.R_sp = False

        if self.geometry_type == 'Cylinder':
            dop_area = 0
            self.A_r = Aw = 2 * np.pi * self.R * self.L + dop_area
            self.As_total = np.pi * (self.R ** 2)
            self.V = np.pi * (self.R ** 2) * self.L
            self.A = Aw + 2 * self.As_total
            self.A_l = 2 * self.As_total
            self.A_dc = Aw + self.As_total

        elif self.geometry_type == 'Semisphere + Cylinder':
            Aw_sphere = np.pi * (self.D ** 2) / 2
            Aw_cyl = 2 * np.pi * self.R * self.cyl_l
            self.As_total = np.pi * (self.R ** 2)
            self.A_l = self.As_total = np.pi * (self.D ** 2) / 4
            self.A = self.A_dc = self.As_total + Aw_cyl + Aw_sphere
            V_cyl = np.pi * (self.R ** 2) * self.cyl_l
            V_sphere = np.pi * (self.D ** 3) / 12
            self.V = V_cyl + V_sphere
            self.A_r = Aw_cyl + Aw_sphere


        elif self.geometry_type == 'Cone':
            self.As_total = np.pi * (self.R ** 2)  # площадь выходного отверстия
            Aw = np.pi * np.sqrt((self.L ** 2) + ((self.R - (self.d / 2)) ** 2)) * (
                    self.R + (self.d / 2))  # площадь стенок
            self.V = (1 / 3) * np.pi * self.L * (
                    (self.R ** 2) + (self.R * (self.d / 2)) + ((self.d / 2) ** 2))  # полный объем камеры
            Ain = np.pi * ((self.d / 2) ** 2)
            self.A = self.A_dc = Aw + self.As_total + Ain
            self.A_r = Aw
            self.A_l = self.As + Ain
            self.As = np.pi * ((self.diameter_out / 2) ** 2)

        elif self.geometry_type == 'Semisphere':
            self.L = self.R_sp
            self.R = self.R_sp
            self.V = (4/6) * np.pi * (self.R ** 3)
            self.As_total = np.pi * (self.R ** 2)  # площадь выходного отверстия
            self.A = self.A_dc = 2 * np.pi * (self.R**2) + self.As_total

            self.A_r = 2 * np.pi * (self.R **2)
            self.A_l = self.As_total
            self.As = np.pi * ((self.diameter_out / 2) ** 2)

        elif self.geometry_type == 'truncated Semisphere':
            self.L = self.R_sp + np.sqrt((self.R_sp ** 2) - (self.R ** 2))
            self.V = np.pi * (self.L ** 2) * (self.R_sp - (self.L / 3))
            self.A = 2 * np.pi * self.R_sp * self.L + np.pi * (self.R_sp ** 2)
            self.As_total = np.pi * (self.R ** 2)  # площадь выходного отверстия
            self.A_r = 2 * np.pi * self.R_sp * self.L
            self.A_l = self.As_total
            self.As = np.pi * ((self.diameter_out / 2) ** 2)
    # ------------------------------------------------------------------------------
    def grid_param(self, scr_th=0.5E-3, acel_th=1.5E-3, grid_gap=0.80E-3, bridge_len=0.5E-3):
        """ Расчет параметров электродов.

        Рассчитывается:
            1. Максимальное число отверстий ЭЭ при заданных параметрах
            2. Прозрачности геометрическая, ионная и для нейтралов

        Args:
            scr_th: толщина ЭЭ
            acel_th: толщина УЭ
            grid_gap: межэлектродный зазор
            bridge_len: растояние между отверстиями ЭЭ

        Returns(запись в .self):
            self.num_of_holes: число отверстий электродов
            self.transp_geom: геометрическая прозрачность ЭЭ
            self.transp_neutral: прозрачность УЭ
            self.transp_ion: прозрачность для ионов

        """
    # ===============================================================================
    # The Electrode Holes Counter
    # author  : Ion Quiet
    # version : 1.003
    # 26.10.2018
    # ===============================================================================

        perf_diam = self.diameter_out
        s_a_gap = grid_gap
        perf_rad = (perf_diam / 2) * 1E3  # [mm] -> the radius of a perforated area
        scr_el_rad = (self.scr_el_hole_diam / 2) * 1E3  # [mm] -> the radius of a screen electrode hole
        bridge_len *= 1E3  # [mm] -> the length of bridge between holes
        step = scr_el_rad * 2 + bridge_len  # [mm] -> the distance between axes of screen electrode holes

        # -> the number of holes steps
        num_of_steps = int((perf_rad - scr_el_rad) / step)

        # -> [mm] the distance between rows of holes
        row_step = math.sqrt(step ** 2 - (step / 2) ** 2)

        # -> [mm] the length of rows
        row_length = [(math.sqrt(perf_rad ** 2 - (i * row_step) ** 2) - i * step / 2) for i in range(1, num_of_steps)]

        # -> the number of holes in a sector. There are six sectors
        sect_holes = 0

        # -> the cycle for calculation of holes number in a sector.
        #    On each iteration is calculated a number of holes per row in a sector,
        #    inside the perforation area radius
        for n in row_length:
            sect_holes = sect_holes + int((n - scr_el_rad) / step)

        # -> The total number of holes
        self.num_of_holes = 1 + num_of_steps * 2 * 3 + sect_holes * 6

        # ===============================================================================

        S_holes_sum = self.num_of_holes * np.pi * (self.scr_el_hole_diam ** 2) / 4
        S_ac_holes_sum = self.num_of_holes * math.pi * (self.acel_el_hole_diam ** 2) / 4
        Emiss_ohu = self.num_of_holes * math.pi * ((self.scr_el_hole_diam / 2) + (bridge_len / 2)) ** 2
        self.transp_geom = S_holes_sum / self.As
        self.transp_neutral = S_ac_holes_sum / self.As
        lg = self.grid_gap + self.scr_th + (self.acel_th / 2)
        le = np.sqrt((lg) ** 2 + (self.scr_el_hole_diam ** 2 / 4))

        Emiss_area = 2 * math.pi * le * (le - lg)
        self.transp_ion = (self.num_of_holes * Emiss_area) / self.As
        transp_oh = Emiss_ohu / self.As

    # ------------------------------------------------------------------------------
    def grid_test(self):
        """ Вывод значений параметров электродов в консоль. """
        print('Ion transparency: ', self.transp_ion, 'Neutrals transparency: ', self.transp_neutral, \
              'Number of holes: ', self.num_of_holes)

    # ------------------------------------------------------------------------------
    def interp_kr(self, Te, data, ind):
        """ Метод интерполяции таблицы для текущего значения Те.

        Args:
            Te: текущая температура электронов (эВ)
            data: pd.df - таблица со значениями искомой величины
            ind: метка выбора

        Alternatives:
            ind = True:
                Возвращается значение константы скорости реакции
            ind = False:
                Возвращает значение сечения взаимодействия

        Returns:

            kr: значение искомой величины для текущего значения Те

        """
        if ind:
            kr = np.interp(Te, data['Tev'], data['Kin'])
        else:
            kr = np.interp(Te, data['Energy (eV)'], data['Cross section (m2)'])

        return kr

    # ------------------------------------------------------------------------------
    def ode_equation(self, n, ng, Tg, Te, ind):
        """ Расчет дополнительных соотношений при решении ОДЕ.

        Args:
            n: концентрация плазмы (1/м3)
            ng: конецентрация нейтрального газа (1/м3)
            Tg: температура нейтрального газа (К)
            Te: температура электронов (К)
            ind: индекс запроса

        Parameters:
            Tev: температура электронов (эВ)

        Returns:
           Если ind = 1:
            Eq1, Eq2, Eq3, Ploss - основные уравнения ОДЕ (вызывается в цикле расчета величин)
           Если ind = 0:
            Prf, Pabs, Ploss, Ji, uB, Rind, etc  - необходимые параметры плазмы (вызывается после выхода на стационар
                                                    для расчета стационарных значений)

        """
        Tev = (self.kb * Te) / self.qe

        """ Расчет к-тов реакции и сечений соударений.
        
        Parameters:
            Kex: к-т реакции возбуждения
            Kiz: к-т реакции ионизации (аппроксимация из книги Гобеля (приложение))
            Kel: к-т реакции упругого взаимодействия электрона-нейтрала (старая версия Kel = 1E-13)
            Kin: к-т реакции ион-нейтрального взаимодействия
            
            uB: скорость Бома (м/с)
            ve: средняя скорость электра (м/с)
            vg: средняя тепловая скорость нейтралов (м/c)
            vi: средняя скорость ионов (м/с) - равна скорости нейтралов
            
            A0: кулоновский лагорифм (длина тепловой диффузии)
            lam_i: длина свобоного пробега нейтрала (м)
            Vm: частота упругого соударения электронов и нейтралов
            sig_elc: сечение упруго электрон-нейтрал взаимодействия 
            Vin: частота упругого соударения ионов и нейтралов
            B: амплитудное значение магнитного поля индуктора
            wci, wce: циклотронные частоты для ионов и электронов
            Da: коэффицент диффузии поперк магнитного поля
            hr: коэффициент учета изменения параметров плазмы по радиусу
            hl: коэффициент учета изменения параметров плазмы по длине
            
         Alternatives: 
            sigma_using: индекатор использования внешних сечений взаимодействия
            
            1. Расчет констант реакции по аналитическим формулам (книга Гобеля, для ксенона) - sigma_using = 0
                    if Tev <= 5.0:
                        Kiz = 1E-20 * (((8 * self.qe * Tev) / (np.pi * self.m)) ** 0.5) * \
                        (3.97 + 0.643 * Tev - 0.0368 * (Tev ** 2)) * (np.exp(-self.Ui / Tev))
                     elif Tev > 5.0:
                        Kiz = 1E-20 * (((8 * self.qe * Tev) / (np.pi * self.m)) ** 0.5) * \
                        (-1.031E-4 * (Tev ** 2) + 6.386 * (np.exp(-self.Ui / Tev)))
                            
                    Kex = 1.29E-13 * (np.exp(-(self.qe * self.Uj) / (self.kb * Te)))
                    sig_elc = 6.6E-19 * (((Tev / 4) - 0.1) / (1 + ((Tev / 4) ** 1.6)))
                    
            2. Расчет констант реакции через сечения взаимодействия (импорт файла) - sigma_using = 1
                        
        """
        uB = np.sqrt(self.kb * Te / self.M)
        ueT = np.sqrt(self.kb * Te / self.m)
        ve = np.sqrt(8 * self.kb * Te / (np.pi * self.m))
        vg = np.sqrt(8 * self.kb * Tg / (np.pi * self.M))
        vi = vg

        sigma_using = False
        if sigma_using == False:
            if Tev <= 5.0:
                Kiz = 1E-20 * (((8 * self.qe * Tev) / (np.pi * self.m)) ** 0.5) * \
                      (3.97 + 0.643 * Tev - 0.0368 * (Tev ** 2)) * (np.exp(-self.Ui / Tev))
            elif Tev > 5.0:
                Kiz = 1E-20 * (((8 * self.qe * Tev) / (np.pi * self.m)) ** 0.5) * \
                      (-1.031E-4 * (Tev ** 2) + 6.386 * (np.exp(-self.Ui / Tev)))
            Kex = 1.29E-13 * (np.exp(-(self.qe * self.Uj) / (self.kb * Te)))
            sig_elc = 6.6E-19 * (((Tev / 4) - 0.1) / (1 + ((Tev / 4) ** 1.6)))
            Kel = sig_elc * ve

        if sigma_using == True:
            Kiz = self.interp_kr(Tev,self.Kiz_data,ind=True)
            Kex = self.interp_kr(Tev,self.Kex_data,ind=True)
            Kel = self.interp_kr(Tev,self.Kel_data,ind=True)
            sig_elc = self.interp_kr(Tev,self.elc_data,ind=False)

        Kin = self.sigi * vi
        A0 = (self.R / 2.405) + (self.L / np.pi)
        lam_i = 1 / (self.sigi * ng)
        Vm = sig_elc * ve * ng # Ven

        Vin = self.sigi *vi * ng
        B = self.num_turns * self.Icoil * self.mu_0 / self.L
        wci = self.qe * B / self.M
        wce = self.qe * B / self.m
        tB = wci * wce / (Vm * Vin)

        Da = self.qe*Tev / (1 + tB) / (self.M * Vin)

        if self.geometry_type == 'Cone':
            Rd = (self.D - self.d) / 4
        elif self.geometry_type == 'Semisphere + Cylinder' or self.geometry_type == 'Semisphere':
            Rd = self.R_sp
        elif self.geometry_type == 'Cylinder':
            Rd = self.R

        mf_r = (0.8 * Rd * uB )/(2.405 * sp.jn(1, 2.405) * Da)
        mf_l = (0.86 * self.L * uB )/(np.pi * Da)
        hr = 0.8 / np.sqrt(4 + (Rd / lam_i) + (mf_r)**2)
        hl = 0.86 / np.sqrt(3 + (self.L / 2 / lam_i) + (mf_l) ** 2)

        """ Расчет к-тов неоднородности распределения плазмы от центра к стенкам ГРК.
        
        Parameters:
            lam_i: длина свободного пробега ион-нейтрал взаимодействия
            hl: осеваой к-т неоднородности распределения плазмы от центра к стенкам ГРК
            hr: радиальный к-т неоднородности распределения плазмы от центра к стенкам ГРК
            Aeff: эффективная площадь падения электронов (модель Чаберта)
            Aeff1: эффективная площадь падения нейтралов (модель Чаберта)
            
        Conditions(if):
            Если тип ГРК - конус:
                в качестве расстояния до стенки  используется среднее значение радиусов конуса
                
            Если типа ГРК - полусфера или сфера
                в качестве расстояния до стенки используется радиус сферы
                
            Если тип ГРК - цилиндр:
                в качестве расстояния до стенки используется радиус цилиндра
            
        """

        Aeff = hr * self.A_r + hl * self.A_l
        Aeff1 = hr * self.A_r + hl * self.A_l - self.transp_ion * hl * self.As

        """ Параметры эквивалентных нагрузок цепи.
        
        Parameters:
            
            w: круговая частота ВЧ генератора
            wpe: плазменная частота
            Eps_p: комплексная плазменная проводимость
            k0: волновой вектор
            k: волновой вектор (??? в чем разница физическая ???)
            Z1: к-т содержащий функции Бесселя (используется в Rind и Lind)
            Rind: эквивалентное сопративление плазмы
            Lind: эквивалентная индуктивность контура (в расчете не участвует)
            Zind: общий импеданс (в расчете не участвует)
        
        Notes:
            1. Только для цилиндрической ГРК для остальных нужно уточнение
            
        """
        w = 2 * np.pi * self.freq  # cycle frequency of RF
        wpe = np.sqrt((n * (self.qe ** 2)) / (self.m * self.Eps0))
        Eps_p = 1 - ((wpe ** 2) / (w * (w - 1j * Vm)))
        k0 = w / self.c
        k = k0 * (Eps_p ** 0.5)
        Z1 = (1j * k * self.R * sp.jn(1, k * self.R)) / (Eps_p * sp.jn(0, k * self.R))
        Rind = ((2 * np.pi * (self.num_turns ** 2)) / (self.L * w * self.Eps0)) * Z1.real
        Lind = self.Lcoil * (1 - ((self.R ** 2) / (self.Rc ** 2))) + ((2 * np.pi * (self.num_turns ** 2)) \
                                                                      / (self.L * (w ** 2) * self.Eps0)) * Z1.imag
        Zind = Rind + 1j * Lind * w

        """ Учет влияния магнитного поля антенны на диффузию частиц к стенкам.
                           
        Parameters:
            self.Fc: к-т учета магн. поля, участвет при определнии потоков на стенки
        
        Notes:
            1. Из стати Гобеля реализована модель учета магн. поля confinement_factor() - зависимая от парамтеров плазмы
            2. Ипользовались к-ты выбираемые в ручную
            3. Использовался постоянный к-т
        
        """

        """  Уравнение сохранения энергии для электронов. 

        Parameters:
            P1: потери энергии на ионизацию (Вт/м3)
            P2: потери энергии на возбуждение нейтральных атомов (Вт/м3)
            P3: потери на упругий нагрев нейтралов (Вт/м3)
            P4: потери частиц на стенках ГРК, электроде, и в ионном пучке (Вт/м3)
            Ploss: суммарные потери энергии(Вт/м3)
            Pabs: энергия поглащенная плазмой от ВЧ генератора (Вт/м3)

            Prf: мощность ВЧ генератора (Вт)

            phi_w: потенциал между диэлектрической стенкой и плазмой
            phi_e: потенциал между ЭЭ и плазмой
            phi: общий потенциал между плазмой и стенками
            Is: поток ионов на ЭЭ
            Iw: поток ионов на стенки ГРК
            Ies: поток электронов ЭЭ (self.As_total - учитывает не перфорированную часть электрода)
            Iew: поток электронов на стенки ГРК
            Ib: ионный поток выходящий из двигателя
            Ji: плотность ионного потока

        Alternatives:
            1. Постоянный коэффициент учета магнитного поля  magn_conf = False
            2. Расчет коэффициента учета магнитного поля magn_conf = True
        """
        # magn_conf = True
        magn_conf = False
        if magn_conf == False:
            self.Fc = 0.75
            phi_w = (self.kb * Te / self.qe) * np.log((5 / (6 * self.Fc)) * np.sqrt((2 * self.M) / (np.pi * self.m)))
            phi_e = (self.kb * Te / self.qe) * np.log((1 - self.transp_ion) * (5 / 6) * np.sqrt((2 * self.M) / (np.pi * self.m)))
            Is = 0.6 * (1 - self.transp_ion) * n * self.qe * uB * self.As_total
            Iw = 0.6 * n * self.qe * uB * self.A_dc * self.Fc
            Ib = 0.6 * n * self.qe * uB * self.As_total * self.transp_ion
            Ies = 0.25 * n * self.qe * ve * np.exp(-(self.qe * phi_e / (self.kb * Te))) \
                  * ((1 - self.transp_ion) * self.As_total)
            Iew = 0.25 * n * self.qe * ve * np.exp(-(self.qe * phi_w / (self.kb * Te))) * self.A_dc * self.Fc

        if magn_conf:
            phi_w = (self.kb * Te / self.qe) * np.log((1/(2*hl)) * np.sqrt(self.M / (2 * np.pi * self.m)))
            phi_e = (self.kb * Te / self.qe) * np.log((1 - self.transp_ion) * (1/(2*hl)) * np.sqrt(2*self.M / (np.pi * self.m)))
            Is = hl* (1 - self.transp_ion) * n * self.qe * uB * self.As_total
            Iw = hl* n * self.qe * uB * self.A_dc
            Ib = hl * n * self.qe * uB * self.As_total * self.transp_ion
            Ies = 0.25 * n * self.qe * ve * np.exp(-(self.qe * phi_e / (self.kb * Te))) \
                  * ((1 - self.transp_ion) * self.As_total)
            Iew = 0.25 * n * self.qe * ve * np.exp(-(self.qe * phi_w / (self.kb * Te))) * self.A_dc

        P1 = self.Ui * n * ng * Kiz * self.qe
        P2 = self.Uj * n * ng * Kex * self.qe
        P3 = 3 * (self.m / self.M) * self.kb * (Te - Tg) * n * ng * Kel
        P4 = (Ies * (2.0 * Tev + phi_e) + Iew * (2.0 * Tev + phi_w)) / self.V
        Ploss = -(P1 + P2 + P3 + P4)
        Pabs = (0.5 / self.V) * Rind * (self.Icoil ** 2)
        Prf = 0.5 * (Rind + self.Rcoil) * (self.Icoil ** 2)
        Ji = Ib / self.As_total / self.transp_ion

        """ Оценка двукратных ионов.

         Parameters:
            Udin: потенциал двукратной ионизации ксенона
            Kdin: скорость реакции двукратной ионизации
            Idi: ток двукратных ионов

        """
        Udin = 21.2
        Kdin = 2e-14 * np.exp(-Udin / Tev) * Tev
        Idi = n * n * self.qe * Kdin * self.V

        """ Уравнение неразрывности для электронов.
        
        Parameters:
            Eq11: источник частиц в результате ионизации
            Eq12: потери частиц на стенках камеры и в пучке
        
        Alternatives:
            1. Eq12 = n * uB * (Aeff / self.V) # модель Чаберта
            2. Eq12 = (Is + Iw + Ib) / (self.V * self.qe) # потери частиц через потоки на элементы
        
        """
        Eq11 = n * ng * Kiz
        Eq12 = - (Is + Iw + Ib) / (self.V * self.qe)

        """ Уравнение неразрывности для нейтралов.
        
        Parameters:
            Eq21: удельный расход РТ в ГРК
            Eq22: источник нейтралов в виде рекомбинации ионов на стенках
            Eq23: убыль нейтралов в виде ионизации
            Eq24: убыль нейтралов за срез двигателя
            
            Fg: поток нейтралов покидающих двигатель
            uW: тепловая скорость нейтралов в приближении равности их температуры Tg0 (темп стенки)
            
        Alternatives:
            1. Eq22 = n * uB * (Aeff1 / self.V) # модель Чаберта (поток нейтралов после рекомбинации со скростью Бома)
            2. Eq22 = ((Is + Iw) * uW) / (self.V * self.qe * uB) # поток нейтралов после рекомбинации со скростью стенки
            3. Eq22 = (Is + Iw) / (self.V * self.qe) # поток нейтралов после рекомбинации со скростью Бома
            
        """

        Fg = 0.25 * ng * vg * self.clausing
        uW = np.sqrt(8 * self.kb * self.Tg0 / (np.pi * self.M))

        Eq21 = self.Q0 / self.V
        Eq22 = (Is + Iw) / (self.V * self.qe)
        Eq23 = - n * ng * Kiz
        Eq24 = - Fg * (self.Ag / self.V)

        """  Уравнение сохранения энергии для нейтралов.
            
        Parameters:
            Eq31: нагрев нейтралов за счет упругих соударений с электронами
            Eq32: нагрев нейтралов за счет упругих соударений с ионами
            Eq33: нагрев нейтралов при соударении с нагретой стенкой
        
        """
        Eq31 = 3 * (self.m / self.M) * self.kb * (Te - Tg) * n * ng * Kel
        Eq32 = 0.25 * self.M * (uB ** 2) * n * ng * Kin
        Eq33 = - self.kt * (((Tg - self.Tg0) / A0) * (self.A / self.V))

        """ Основные уравнения ОДЕ.

         Parameters:
             Eq1: уравнение неразрывности для электронов (ур-е баланса частиц)
             Eq2: уравнение неразрывности для нейтралов (ур-е баланса частиц)
             Eq3: уравнение сохранения энергии для нейтралов (ур-е баланса энергий)
             Eq4 (Ploss): уравнение сохранения энергии для электронов (ур-е баланса энергий)

        """
        Eq1 = Eq11 + Eq12
        Eq2 = Eq21 + Eq22 + Eq23 + Eq24
        Eq3 = (2 / (3 * self.kb)) * (Eq31 + Eq32 + Eq33)
        Eq4 = (2 / (3 * self.kb)) * (Pabs + Ploss)

        if ind:
            return Eq1, Eq2, Eq3, Eq4
        else:
            print('Double ions',Idi, 'Currents', Ib, Ies-Is, 'RF power', Prf)
            print('hl', hl, 'hr', hr)
            print('Using import sigma:', sigma_using)
            return Prf, Pabs, Ploss, Ji, uB, Rind, Lind, Vm, P1, P2, P3, P4, Is, Iw, Ib, Ies, Iew, \
                   phi_e, phi_w, Eq11, Eq12, Eq21, Eq22, Eq23, Eq24, Eq31, Eq32, Eq33, lam_i, hr, hl

    # ------------------------------------------------------------------------------
    def f(self, t, y):
        """ Решаемые уравнения ОДЕ.

        Args:
            t: время расчета (не используется)
            y: значение величин в 0-ой момент

        Parameters:
            n, ng, Tg, Te - значение величин в 0-ой момнет времени
            Eq1, Eq2, Eq3, Eq4 - значение уравнений при дейтсвующих n, ng, Tg, Te
            f1: концентрация электронов
            f2: концентрация нейтралов
            f3: температура нейтралов
            f4: температура электронов

        Returns:
            n, ng, Tg, Te: значения после выхода на стационарное решение

        """
        n, ng, Tg, Te = y

        Eq1, Eq2, Eq3, Eq4 = self.ode_equation(n, ng, Tg, Te, ind=1)

        f1 = Eq1
        f2 = Eq2
        f3 = (1 / ng) * (Eq3 - Tg * Eq2)
        f4 = (1 / n) * (Eq4 - Te * Eq1)
        #print(n, ng, Tg, Te)
        return [f1, f2, f3, f4]

    # ------------------------------------------------------------------------------
    def solver_vode(self):
        """ Решатель ОДЕ методом Рунге-Кутта (VODE).

        Parameters:
            ODE: решеаемые уравнения
            y0, t0: начальные значения n, ng, Tg, Te
            tmax: время интегрирования
            dt: шаг интегрировния по времени

        Returns:
            sol: раситанные значения n, ng, Tg, Te

        """
        ODE = ode(self.f)
        y0, t0 = [self.n0, self.ng0, self.Tg0, self.Te0], 0
        r = ODE.set_integrator('vode', atol=[1E-04, 1E-04, 1E-04, 1E-04], max_step=0.000001, nsteps=100000)
        r = ODE.set_initial_value(y0, t0)
        tmax = 0.02
        dt = 0.000001

        while ODE.successful() and ODE.t <= tmax:
            sol = ODE.integrate(ODE.t + dt)

        # print(sol1)
        # print('%.2f \t%.3f \t%.3f \t%.3f' % (ODE.t, ODE.y[0],ODE.y[1],ODE.y[2]))
        print('-----------------------------\n'
              'Icoil:', self.Icoil, 'A', 'Mass flow:', self.Q0 * self.M * 1E6, 'mg/s Fc', self.Fc,'\nResults:', sol,
              '\n'

              '-----------------------------')
        return sol

    # ------------------------------------------------------------------------------
    def solver_dop853(self):
        """ Решатель ОДЕ методом Рунге-Кутта (dop853).

        Parameters:
            ODE: решеаемые уравнения
            y0, t0: начальные значения n, ng, Tg, Te
            tmax: время интегрирования
            dt: шаг интегрировния по времени

        Returns:
            sol: раситанные значения n, ng, Tg, Te

        """
        ts = []
        ys = []
        tmax = 0.01
        ODE = ode(self.f)
        y0, t0 = [self.n0, self.ng0, self.Tg0, self.Te0], 0
        #print('TEST!!!!!!!',self.n0, self.ng0, self.Tg0, self.Te0)
        # r = ODE.set_integrator('dop853', rtol=0.1, max_step=0.000000001, nsteps=1e9)  # метод Рунге – Кутта
        r = ODE.set_integrator('dop853', rtol=0.01, max_step=0.000001, nsteps=1e5)
        # r = ODE.set_integrator('dop853', rtol=0.1, max_step=0.01, nsteps=1e2)  # метод Рунге – Кутта
        # r.set_solout(fout)  # загрузка обработчика шага
        r = ODE.set_initial_value(y0, t0)  # задание начальных значений
        sol = r.integrate(tmax)  # решаем ОДУ

        Y = np.array(ys)
        tm = np.array(ts)

        print('-----------------------------\n'
              'Icoil:', self.Icoil, 'A', 'Mass flow:', self.Q0 * self.M * 1E6, 'mg/s','\nResults:', sol,
              '\n'

              '-----------------------------')

        return sol

    # ------------------------------------------------------------------------------
    def ode_initial_value(self, Q0, Tg0, n0=1E17, Te0=10000.0):

        """ Начальные условия ОДЕ.

        Args:
            Q0: удельный расход
            n0: начальная концентрация плазмы
            Te0: начальная температура электронов

        Parameters:
            self.Ag: открытая площадка для нейтралов
            self.Ai: открытая площадка для ионов
            vg0: скорость потока нейтралов (не импользуется)
            p0: давление в 0-ой момент (Pa)(не импользуется)
            p0_t: давление в 0-ой момент (torr) (не импользуется)
            self.ng0_m: начальная концентрация газа

        """
        self.n0 = n0
        self.Te0 = Te0
        #self.Ag = self.transp_neutral * np.pi * ((self.diameter_out / 2) ** 2)
        self.Ag = self.transp_geom * np.pi * ((self.diameter_out / 2) ** 2)
        self.Ai = self.transp_ion * np.pi * ((self.diameter_out / 2) ** 2)
        vg0 = np.sqrt(8 * self.kb * self.Tg0 / (np.pi * self.M))
        p0 = (4 * self.kb * Tg0 * Q0) / (vg0 * self.Ag)
        p0_t = p0 * 7.5E-3  # torr
        self.ng0_m = (4 * Q0) / (vg0 * self.Ag * self.clausing)

    # ------------------------------------------------------------------------------

    @timing
    def plasma_param(self, Icoil, mass_flow, Tg0):
        """ Расчет парамтеров плазмы.

        Args:
            Icoil: массив токов в антенне
            mass_flow: массив расходов
            Tg0: массив температур стенки

        Parameters:

        Notes:
            1. Сделать возможность выбора из списка величин для записи
            2. Есть ли возможность сократить формирование массивов в более компактном формате

        """

        """ Формирование массивов. """
        self.n_el = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.n_neutrals = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.T_neutrals = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.T_el = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Prf = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Pabs = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Ploss = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Ji = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.uB = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Rind = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Lind = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Vm = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.P1 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.P2 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.P3 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.P4 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.phi_w = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.phi_e = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Is = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Iw = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Ib = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Ies = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Iew = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq11 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq12 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq21 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq22 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq23 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq24 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq31 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq32 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.Eq33 = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.lami = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.hr = np.zeros((np.size(Icoil), np.size(mass_flow)))
        self.hl = np.zeros((np.size(Icoil), np.size(mass_flow)))

        """ Расчет параметров плазмы в цикле по Icoil и массовому расходу
        
        Parameters:
            n_el, n_neutrals, T_neutrals, T_el: концентрации и температуры электронов и нейтралов
            Prf,  Lind, Rind
            P1, P2, P3, P, Ploss, Pabs: составляющие уравнения сохраниения энергии для электронов, суммарные вложенная
                                        мощность и потери
            Is, Iw, Ib, Ies, Iew: потоки электронов и ионов на элекменты ГРК
            phi_e, phi_w: потенциалы между плазмой и стенкой и электродом
            Eq11, Eq12: составляющие уравнения неразрывности для электронов
            Eq21, Eq22, Eq23, Eq24: составляющие уравнения неразрывности для нейтралов
            Eq31, Eq32, Eq33, составляющие уравнения сохраниения энергии для нейтралов
            lami, Ji, uB: длина свобоного пробега нейтрала, плотность тока ионов в ГРК, скорость Бома
            Vm: частота упругого соударения электронов и ионов
        
        Alternatives:
            1. self.Rcoil = self.Rcoil0 * (1 + self.alfa * (self.Tg0 - 300)) - учет изменения сопр от темп
            2. self.Rcoil = self.Rcoil0 - без учета изменений
        
        Notes:
            1. На каждом шагу происходит перезапись в .self значений: Icoil, Tg0, ng0, Q0
            2. Вызывается метод solver_dop853() в котром вызвается метод
        
        """
        try:
            counter = 0
            for i in range(np.size(Icoil)):
                for j in range(np.size(mass_flow)):
                    counter += 1
                    start_time = time.time()
                    self.Icoil = Icoil[i]
                    self.Tg0 = Tg0[i]
                    self.ode_initial_value(mass_flow, self.Tg0)
                    self.ng0 = self.ng0_m[j]
                    self.Q0 = mass_flow[j]
                    #self.Rcoil = self.Rcoil0 * (1 + self.alfa * (self.Tg0 - 300))
                    self.Rcoil = self.Rcoil0
                    self.n_el[i][j], self.n_neutrals[i][j], self.T_neutrals[i][j], self.T_el[i][j] = self.solver_dop853()

                    self.Prf[i][j], self.Pabs[i][j], self.Ploss[i][j], self.Ji[i][j], self.uB[i][j], self.Rind[i][j], \
                    self.Lind[i][j], self.Vm[i][j], self.P1[i][j], self.P2[i][j], self.P3[i][j], self.P4[i][j], \
                    self.Is[i][j], self.Iw[i][j], self.Ib[i][j], self.Ies[i][j], self.Iew[i][j], \
                    self.phi_e[i][j], self.phi_w[i][j], self.Eq11[i][j], self.Eq12[i][j], self.Eq21[i][j], self.Eq22[i][j], \
                    self.Eq23[i][j], self.Eq24[i][j], self.Eq31[i][j], self.Eq32[i][j], self.Eq33[i][j], \
                    self.lami[i][j], self.hr[i][j], self.hl[i][j] = self.ode_equation(self.n_el[i][j], self.n_neutrals[i][j],\
                    self.T_neutrals[i][j], self.T_el[i][j], ind=0)
                    requests.post('http://localhost:9990/set_time', data={'key': round((np.size(Icoil) * np.size(mass_flow) - counter) * (time.time() - start_time))})
            # print("--- %s seconds ---" % (time.time() - start_time))
        except:
            print("Ошибка")


    # ------------------------------------------------------------------------------
    def heat_flux(self, Ae=3.95, Aa=3.95):
        """ Расчет тепловых потоков от плазмы.

        Args:
            Ae: работа выхода электрона из ЭЭ (Ti - 3.95 eV, Mo - 4.37 eV), по умолчанию - Ti
            Aa: работа выхода электрона из УЭ, по умолчанию - Ti

        Parameters:
            Tev: температура электронов в эВ
            P3: удельная мощность возбужденных атомов
            Icex: ток перезарядочных ионов (2% от тока пучка)

            Потоки на стенки ГРК (Вт)
                self.Pwi: поток ионов
                self.Pwe:  поток электронов
                self.Pxw: лучистый поток возбужденных атомов
            Потоки на ЭЭ (Вт)
                self.Pse: поток электронов
                self.Psi: поток ионов
                self.Pxs: лучистый поток
            Потоки на УЭ (Вт)
                self.Pcex: поток перезарядочниых ионов (для Uae = 500V)
                self.Pxa: лучистый поток возбужденных атомов
            Потоки покидающие двигатель (Вт)
                self.Pb: мощность ионного пучка
                self.Pxout: лучистый поток возбужденных атомов
            Проверка баланса мощностей:
                Psum: суммарная мощность
                Pion: мощность ионов

        """
        Tev = self.kb * self.T_el / self.qe
        Icex = 0.02 * self.Ib

        self.Pwi = self.Iw * (self.phi_w + 0.5 * Tev + self.Ui)
        self.Pwe = self.Iw * (2 * Tev)
        self.Pxw = (self.A_dc / self.A) * self.P2 * self.V
        self.Pse = self.Ies * (Ae + 2 * Tev)
        self.Psi = self.Is * (self.phi_e + 0.5 * Tev + self.Ui - Ae)
        self.Pxs = (1 - self.transp_geom) * self.As * self.P2 * self.V / self.A
        self.Pxa = (1 - self.transp_neutral) * self.transp_geom * self.As * self.P2 * self.V / self.A
        Uae = 500
        self.Pcex = Icex * (Uae + self.Ui - Aa)
        self.Pb = self.Ib * (self.phi_e + 0.5 * Tev + self.Ui)  # мощность ионного пучка
        self.Pxout = self.transp_neutral * self.transp_geom * self.As * self.P2 * self.V / self.A

        Psum = self.P3 * self.V + self.Pwi + self.Pwe + self.Pxw + self.Pse + self.Psi + self.Pxs + self.Pxa + self.Pcex + self.Pb + self.Pxout
        Pion = self.Pwe + self.Pse

    # ------------------------------------------------------------------------------
    def thruster_param(self, power_limit, mass_flow , Icoil, gamma=0.97, KN_flow=0.03, dJcl=0.7, dUae=0.15):
        """ Расчет параметров двигателя.

        Args:
            power_limit: предел по общей мощности двигателя
            mass_flow: массив расходов
            Icoil: массив токов индуктора
            gamma: к-т учета угла расхождения пучка (по умолчанию 0.97)
            KN_flow: массовый расход в катоде-нейтралезаторе (мг/с)
            dJcl: к-т отношения допустимой плотности ионов к предельной по з-ну Чайлда-Ленгмюра
            dUae: отношение потенциалов на УЭ и ЭЭ

        Parameters:
            self.I_beam: ток ионного пучка
            self.Uee: потенциал ЭЭ
            self.Uae: потенциал УЭ
            self.ios: мощность ИОС
            self.total_power: общая мощность двигателя (рассчитывается, если не задана)
            le: расстояние от центра УЭ до поверхности мениска
            self.Jcl: предельный ионный ток
            self.thrust: тяга двигателя (Н) - для записи используется self.thrust*1е3 (мН)
            self.thrust_cost: цена тяги (Вт/Н) - для записи используется self.thrust_cost*1е-3 (Вт/мН)
            self.ion_flow: поток ионов (мг/с)
            self.gas_efficiency: газовая эффективность
            self.specific_impulse: удельный импульс
            self.Vi_av: скорость ускоренных ионов
            self.power_thrust: мощность пучка
            self.energy_efficiency: энергетическая эффективность
            self.full_efficiency: полная эффективность
            self.ion_cost: цена иона (Вт/мА)

            self.file_saver(): вызов метода записи полученных данных в файл для текущего значения общей мощности

        Conditions:
            Если задан self.power_limit = [...]:
                Потенциал ЭЭ рассчитывается из условия остатка мощности на ИОС при заданном токе пучка
            Если self.power_limit = [None]:
                Потенциал ЭЭ рассчитывается из условия, что плотность тока ионов равна 70%(dJcl) от предельного значения
                          по з-ну Чайлда-Ленгмюра, при этом считается, что потенциал УЭ равен 15%(dUae) от потенциала ЭЭ

        Alternatives:
            1. self.I_beam = self.Ib - ток пучка из потоков частиц
               self.I_beam = self.Ji * self.As * self.transp_ion - по модели Чаберта

        """
        total_power = np.zeros((np.size(Icoil), np.size(mass_flow)))
        Uee = np.zeros((np.size(Icoil), np.size(mass_flow)))
        Jcl = np.zeros((np.size(Icoil), np.size(mass_flow)))
        Uae = np.zeros((np.size(Icoil), np.size(mass_flow)))
        thrust = np.zeros((np.size(Icoil), np.size(mass_flow)))
        thrust_cost = np.zeros((np.size(Icoil), np.size(mass_flow)))

        self.I_beam = self.Ib
        le = (self.grid_gap + self.scr_th) ** 2 + (self.scr_el_hole_diam ** 2 / 4)
        if power_limit!=None:
            Uee = (power_limit - self.Prf) / self.I_beam
            Uae[:][:] = Uee * dUae
            total_power[:][:] = power_limit
        if power_limit == None:
            for i in range(np.size(Icoil)):
                for j in range(np.size(mass_flow)):
                    Pr = (4 * self.Eps0 / 9) * (np.sqrt(2 * self.qe / self.M))
                    Uee[:][:] = (self.Ji * le / (Pr * dJcl)) ** (2 / 3) / (1 + dUae)
                    Uae[:][:] = Uee * dUae

        ios = Uee * self.I_beam
        total_power = ios + self.Prf

        Jcl = (4 / 9) * self.Eps0 * np.sqrt(2 * self.qe / self.M) * (((Uee + abs(Uae)) ** 1.5) / (le))
        thrust = gamma * self.I_beam * np.sqrt(2 * self.M * (Uee) / self.qe)
        thrust_cost = total_power / thrust
        ion_flow = self.I_beam * self.M / self.qe
        gas_efficiency = ion_flow / ((mass_flow + KN_flow) * 1E-6)
        specific_impulse = thrust / ((mass_flow + KN_flow) * 1E-6) / 9.8
        Vi_av = np.sqrt(2 * self.qe * Uee / self.M)
        power_thrust = ion_flow * (Vi_av ** 2) / 2
        energy_efficiency = power_thrust / total_power
        full_efficiency = energy_efficiency * gas_efficiency
        ion_cost = self.Prf / self.I_beam

        th_param_title = ['Electron density (1/m3)', 'Neutral density (1/m3)', 'Neutral temperature (K)',
                               'Electron temperature (eV)', \
                               'Absorbed Power (W)', 'Power losses (W)', 'Ion current density (A/m2)',
                               'Child-Langmuir limit (A/m2)', \
                               'Plasma resistance (Ohm)', 'Plasma inductance (H)', 'Elastic collision frequency', \
                               'Total Power (W)', 'RF Power (W)', 'Beam current (mA)', 'Uee (V)', 'Thrust (mN)', \
                               'Thrust cost (W/mN)', 'Gas efficiency', \
                               'Specific impulse (sec)', 'Mass Flow (mg/s)', 'Coil current (A)', 'Ionization losses (W)',
                               'Excitation losses (W)', \
                               'Thermal losses (W)', 'Wall particles losses (W)', 'Area (m2)', 'Volume (m3)',
                               'Ion cost (W/mA)', \
                               'Total efficiency', 'Energy efficiency', 'Electron walls flux (W)', \
                               'Electron EE flux (W)', 'Ion SE flux (W)', 'Ion walls flux (W)',
                               'Ions charge exchange flux (W)', \
                               'AE radiant flux (W)', 'SE radiant flux (W)', 'Walls radiant flux (W)', \
                               'Floating potential SE (V)', 'Floating potential W (V)', \
                               '1Eq-1', '1Eq-2', '2Eq-1', '2Eq-2', '2Eq-3', \
                               '2Eq-4', '3Eq-1', '3Eq-2', '3Eq-3', \
                               'Beam current (mA) (test)', 'Screen ions current (mA)', 'Walls ions current (mA)', \
                               'Walls electron current (mA)', 'Screen electron current (mA)', 'i-n mean free path (m)',
                               'Walls temperature (K)', 'Radial coefficient', 'Length coefficient']

        th_param = np.zeros((np.size(Icoil) * np.size(mass_flow), len(th_param_title)))

        k = 0
        for i in range(np.size(Icoil)):
            for j in range(np.size(mass_flow)):
                th_param[k][:] = self.n_el[i][j], self.n_neutrals[i][j], self.T_neutrals[i][j], (
                        self.kb * self.T_el[i][j]) / self.qe, \
                                      self.Pabs[i][j]*self.V, self.V * self.Ploss[i][j], self.Ji[i][j], Jcl[i][j], \
                                      self.Rind[i][j], self.Lind[i][j], self.Vm[i][j], \
                                      total_power[i][j], self.Prf[i][j], 1e3 * self.I_beam[i][j], \
                                      Uee[i][j], 1e3 * thrust[i][j], \
                                      1e-3 * thrust_cost[i][j], gas_efficiency[i][j], \
                                      specific_impulse[i][j], mass_flow[j], Icoil[i], self.P1[i][j], \
                                      self.P2[i][j], \
                                      self.P3[i][j], self.P4[i][j], self.A, self.V, ion_cost[i][j], \
                                      full_efficiency[i][j], energy_efficiency[i][j], self.Pwe[i][j], \
                                      self.Pse[i][j], self.Psi[i][j], self.Pwi[i][j], self.Pcex[i][j], \
                                      self.Pxa[i][j], self.Pxs[i][j], self.Pxw[i][j], \
                                      self.phi_e[i][j], self.phi_w[i][j], \
                                      self.Eq11[i][j], self.Eq12[i][j], self.Eq21[i][j], self.Eq22[i][j], \
                                      self.Eq23[i][j], \
                                      self.Eq24[i][j], self.Eq31[i][j], self.Eq32[i][j], self.Eq33[i][j], \
                                      self.Ib[i][j], self.Is[i][j], self.Iw[i][j], \
                                      self.Iew[i][j], self.Ies[i][j], self.lami[i][j], self.Tg_0[i], self.hr[i][j],\
                                      self.hl[i][j]
                k += 1

        self.file_saver(th_param, th_param_title, power_limit)

    # ------------------------------------------------------------------------------
    def file_saver(self, th_param, th_param_title, power_limit):
        """ Запись полученных данных в .csv-файл.
        Args:
            th_param: выбранные для записи данные
            th_param_title: шапка файла
            power_limit: текущее значение предела по общей мощности двигателя

        Parameters:
            self.filename_creator(power_limit): вызов метода формирования имени файла
            self.file_title_creator(power_limit): вызов метода формирования шапки файла

        """
        with open(str(os.path.abspath(os.getcwd())) + "\\" + "balance_model\\" + self.filename_creator(power_limit), 'w', newline='') as csvfile:
            titlewriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')
            titlewriter.writerow(self.file_title_creator(power_limit))
            datawriter = csv.writer(csvfile, delimiter=';', lineterminator='\n')
            datawriter.writerow(th_param_title)
            # prt = np.transpose(self.th_param)
            prt = th_param
            for i in range(0, len(prt)):
                datawriter.writerow(np.round(prt[i], 8))

    # ------------------------------------------------------------------------------
    def file_title_creator(self, power_limit):
        """ Формирование шапки файла.

        Args:
            power_limit: текущее значение предела по общей мощности двигателя

        """
        text = [f"Thruster class v.0.1: {self.date}\n--------------------------------------------------\n \
        INPUT\n--------------------------------------------------\nGas type: {self.gas_type}; Mass flow:{self.mass_flow} \
        \nGeometry:{self.geometry_type}; {self.geometry_param}\
        \nAntenna parameters: Frequency: {self.freq*1e-6} MHz, Rcoil: {self.Rcoil0} Ohm,Lcoil: {self.Lcoil*1e6} uH,\
         Num of turns: {self.num_turns}, Coil radius: {self.Rc}m\nGrid parameters: {self.grid_parameters} \
         \nPower limit: {power_limit} W\n--------------------------------------------------\n\
          OUTPUT\n--------------------------------------------------\n"]
        return text

    # ------------------------------------------------------------------------------
    def filename_creator(self, power_limit):
        """ Формирование имени файла расчета .csv

        Args:
            power_limit: текущее значение предела по общей мощности двигателя

        """
        today = datetime.datetime.today()
        self.date = today.strftime("%Y.%m.%d-h%Hm%Ms%S")
        if self.d:
            filename = f"{self.geometry_type} (L {round(self.L * 1e3, 2)}, Dmax {round(self.D * 1e3, 2)},\
Dmin {round((self.d * 1e3),2)})_{round(1e-6 * self.freq, 2)} MHz_\
{int(self.num_turns)} turns_PL {round(power_limit,1)} W.csv"
        elif self.cyl_l:
            filename = f'{self.geometry_type} (L {round(self.L * 1e3, 2)}, D {round(self.D * 1e3, 2)},\
 Cyl_length {round(self.cyl_l * 1e3, 2)})_{round(1e-6 * self.freq,2)} MHz_\
{int(self.num_turns)} turns_PL {round(power_limit,1)} W.csv'
        elif self.R_sp:
            filename = f'{self.geometry_type} (L {self.L * 1e3}, D {self.D * 1e3},\
Rad_sphere {self.R_sp * 1e3})_{1e-6 * self.freq} MHz_\
{int(self.num_turns)} turns_PL {round(power_limit,1)} W.csv'
        else:
            filename = f'{self.geometry_type} (L {round(self.L * 1e3, 2)}, \
D {round(self.D * 1e3, 2)})_{1e-6 * self.freq} MHz_\
{int(self.num_turns)} turns_PL {power_limit} W.csv'

        print(filename)  # вывод названия файла в консоли
        self.filename = filename
        return filename

    # -----------------------------------------------------------------------------
    def file_reader(self, file_name='supper_egg.csv'):
        """ Функция расчета времени исполнения метода.

        Notes:
            Не используется в расчете

        """
        with open(file_name, newline='') as csvfile:
            for line in itertools.islice(csvfile, 2, None):
                datareader = csv.reader(csvfile)
                for row in datareader:
                    print(', '.join(row))
    # ------------------------------------------------------------------------------