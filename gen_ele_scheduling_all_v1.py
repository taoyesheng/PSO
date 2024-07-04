# 求可供水量与核算后发电水量

"""调用第三方库"""
import copy
import time

import numpy as np
from matplotlib import pyplot as plt

from PSO import PSO
from read_data import ReadData, loss_water, newton, ca_elec_water
from tools import set_run_mode


# 计算每月的可供水量、核算后发电水量、发电计划 供水量-可供水量，发电量-核算后发电水量
class GenEleScheduling:
    __instance = None
    __initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(GenEleScheduling, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        """
           Initialize the class.
           只调用一次
        """
        # if GenEleScheduling.__initialized:
        #     return
        # GenEleScheduling.__initialized = True

        read_data = ReadData()
        # 初始库容（上年年末库容）
        self.last_month_capacity = read_data.v0
        # 月末库容
        self.monthly_capacities = []
        # 发电前可供水量
        self.supply_waters = []
        # 核算后发电水量
        self.elec_waters = []
        # 发电计划
        self.elec_plan = copy.deepcopy(read_data.elec_plan)

    # 上个月的发电量
    def ret_last_month_capacity(self, nMonth):
        idx = nMonth - 1
        if idx >= 0:
            return self.monthly_capacities[idx % len(self.monthly_capacities)]
        read_data = ReadData()
        return read_data.v0

    # 计算某个月的可供水量与核算后发电水量
    def cal_supply_elec_water(self, nIndex):
        read_data = ReadData()
        leakage = loss_water(self.ret_last_month_capacity(nIndex), read_data.B, read_data.min_v)  # 水库漏损量
        after_leakage_c = self.ret_last_month_capacity(nIndex) + read_data.inflow[nIndex] - leakage  # 扣损后库容
        supply_water = after_leakage_c - read_data.min_v  # 可供水量

        # 计算发电水量
        hu1 = newton(after_leakage_c, read_data.x0, read_data.eps)
        elec_w1 = ca_elec_water(self.elec_plan[nIndex], read_data.A, hu1, read_data.hd)
        elec_w1_a = np.minimum(supply_water, elec_w1)
        v_elec_w1_a = after_leakage_c - elec_w1_a
        hu2 = newton(v_elec_w1_a, read_data.x0, read_data.eps)
        hu = (hu1 + hu2) / 2
        elec_water = ca_elec_water(self.elec_plan[nIndex], read_data.A, hu, read_data.hd)  # 核算后发电水量

        # 计算弃水量
        elec_water_a = np.minimum(supply_water, elec_water)  # 实际发电水量
        before_abondw = supply_water - elec_water_a + read_data.min_v
        abondw = np.maximum(before_abondw - read_data.max_v[nIndex], 0)

        # 计算月末库容
        monthly_capacity = before_abondw - abondw

        return monthly_capacity, supply_water, elec_water

    # 计算全年

    def cal_supply_elec_waters_all(self):

        read_data = ReadData()
        for k2 in range(len(read_data.inflow)):
            monthly_capacity, supply_water, elec_water = self.cal_supply_elec_water(k2)
            self.monthly_capacities.append(monthly_capacity)
            self.supply_waters.append(supply_water)
            self.elec_waters.append(elec_water)

        # print('月未库存：', monthly_capacities)
        # print('可供发电水量：', supply_waters)
        # print('核算后发电用水量：', elec_waters)
        # print('发电计划：\n', elec_plan)

    def cal_supply_elec_waters(self):
        for k2 in range(len(self.elec_plan)):
            monthly_capacity, supply_water, elec_water = self.cal_supply_elec_water(k2)
            self.monthly_capacities.append(monthly_capacity)
            self.supply_waters.append(supply_water)
            self.elec_waters.append(elec_water)

        # print('月未库存：', monthly_capacities)
        # print('可供发电水量：', supply_waters)
        # print('核算后发电用水量：', elec_waters)
        # print('发电计划：\n', elec_plan)

    # 更新发电计划
    def update_elec_plan(self, x):
        self.elec_plan = copy.deepcopy(x)


# 对全年发电计划用粒子群算法求解
# 目标函数 计算出与原定发电计划总距离最小 为目标函数  欧式距离判断曲线相
def a_target_function_distance(x):
    read_data = ReadData()
    fSum = 0
    for i in range(len(x)):
        fSum += (x[i] - read_data.elec_plan[i]) ** 2
    return fSum


def a_target_function_distance_v1(x):
    read_data = ReadData()
    sumRate = 0
    for i in range(len(x)):
        sumRate += ((x[i] - read_data.elec_plan[i]) ** 2) / (read_data.elec_plan[i] ** 2)
    return sumRate


# set_run_mode(a_target_function_distance_v1, "multithreading")


# 加入非线性约束  是否违反约束条件 供水量>发电量>需水量
def a_against_constraint_ueq(x):
    ges = GenEleScheduling()
    ges.update_elec_plan(x)
    ges.cal_supply_elec_waters()
    read_data = ReadData()

    for i in range(len(x)):
        if not (ges.supply_waters[i] > ges.elec_waters[i] > read_data.need_water[i]):
            return True

    return False


# 加入非线性约束  是否违反约束条件 供水量>发电量>需水量
t_l_a_constraint_ueq_v1 = (a_against_constraint_ueq,)


def cal_all_elec_plan_v1(i):

    idx = i + 1

    # 开始时间
    startTime = time.time()
    pso = PSO(func=a_target_function_distance_v1, n_dim=idx, pop=100 * idx, max_iter=50 * idx, lb=[0] * idx,
              ub=[450] * idx,
              constraint_ueq=t_l_a_constraint_ueq_v1, n_processes=0)
    pso.record_mode = True
    pso.verbose = True
    pso.run()
    # 结束时间
    endTime = time.time()
    print("耗时：", endTime - startTime)
    print("月份: ", idx, ' best_x is ', pso.gbest_x, ' best_y is', pso.gbest_y)
    return pso


# 用粒子群算法求解全年发电计划
def iter_cal_all_elec_plan_v1(nStart=0, nEnd=12):
    # 计算全年发电计划
    for i in range(nStart, nEnd):
        pso = cal_all_elec_plan_v1(i)
        if not np.any(np.isinf(pso.gbest_y)):
            rd = ReadData()
            rd.reset_real_elec_plan(pso.gbest_x)
        else:
            print(" 错误！！！")

        Animation(pso, nStart + 1)


def main():
    rd = ReadData()
    iter_cal_all_elec_plan_v1(11, 12)
    print('原始发电计算：\n', rd.elec_plan)
    print('最终发电计划：\n', rd.real_elec_plan)

    # Set the default font to SimHei
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 画图
    # 创建一个新图形
    fig, ax = plt.subplots()

    # 绘制 elec_plan_origin
    ax.plot(rd.elec_plan, label='发电计划', color='blue')
    # 绘制elec_plan
    ax.plot(rd.real_elec_plan, label='最终发电计划', color='red')
    # 设置x轴的刻度为1到len(elec_plan_origin)的所有整数
    mIndex = [i for i in (range(len(rd.elec_plan) + 1))]
    mValue = [str(i + 1) for i in (range(len(rd.elec_plan) + 1))]
    # plt.xticks(range(0, len(rd.elec_plan) + 1))
    plt.xticks(mIndex, mValue)

    # 添加图例
    plt.legend()

    # 设置x轴和y轴标签
    ax.set_xlabel("月份")
    ax.set_ylabel("发电计划")

    # 显示图形
    plt.show()


def Animation(pso, idx=1):
    y_hist = []
    infLen = 0
    for data in pso.gbest_y_hist:
        if data != np.inf:
            y_hist.append(data[0])
        else:
            infLen += 1

    # Set the default font to SimHei
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 画图
    plt.plot(y_hist)

    xIdx = [i for i in range(len(y_hist))]
    xValue = [str(i + infLen - 1) for i in xIdx]
    plt.xticks(xIdx, xValue)
    plt.xlabel("迭代次数")
    plt.ylabel("目标函数值")

    plt.show()

    # # 画动画
    record_value = pso.record_value
    X_list, V_list = pso.record_value['X'], pso.record_value['Y']

    fig, ax = plt.subplots(1, 1)
    ax.set_title('title', loc='center')
    line = ax.plot([], [], 'b.')

    X_grid, Y_grid = np.meshgrid(np.linspace(0, 450, 100 * idx), np.linspace(0, 450, 100 * idx))


# 主函数调用
if __name__ == '__main__':
    for i in range(1):
        main()
