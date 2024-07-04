import copy

import numpy as np
import pandas as pd


def add_lists(lists):
    # 确保所有列表长度相同
    if len(set(len(l) for l in lists)) != 1:
        raise ValueError("All input lists must have the same length.")
    # 使用zip函数组合列表，然后元素对应相加
    summed_list = [a + b + c for a, b, c in zip(*lists)]
    return summed_list


# 水库漏损水量      v0-库容，B-损失系数，min_v-死库容
def loss_water(v0, B, min_v):
    loss = v0 * (1 - B)
    if np.less(loss, min_v).any():
        return 0
    else:
        lw = v0 * B
        return lw


# 水位库容曲线（牛顿插值法）     y-库容，x0-初始水位，eps-迭代精度
def newton(y, x0, eps):
    while True:
        f = 13.192233 * x0 * x0 - 429.036137 * x0 + 3873.971233 - y
        fp = 26.384466 * x0 - 429.036137
        x1 = x0 - f / fp
        if np.linalg.norm(x1 - x0) < eps:
            return x1
        x0 = x1


# 发电水量     elec_p-发电计划（发电量），A-发电机参数，hu-下水位，hd-上水位
def ca_elec_water(elec_p, A, hu, hd):
    elec_w = (3600 * elec_p) / (A * (hu - hd))
    return elec_w


class ReadData:
    __instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(ReadData, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        """
           Initialize the class.
           只调用一次
        """
        if ReadData._initialized:
            return
        ReadData._initialized = True

        """准备所需数据--data.csv包含以下列：month, eco_water, town_water, agr_water, inflow, max_v, elec_plan"""
        # 依次为月份，生态需水，城镇需水，农业需水，来水，最大蓄水库容，发电计划
        self.data = pd.read_csv('数据-50%.csv')
        self.inflow = self.data['inflow']
        self.max_v = self.data['max_v']
        self.elec_plan_origin = self.data['elec_plan']
        self.elec_plan = copy.deepcopy(self.elec_plan_origin)  # 保存原始发电计划
        self.elec_plan = self.elec_plan.astype(float)
        # 生态需水、城镇需水、农业需水的和（总需水）
        self.need_water = add_lists([self.data['eco_water'], self.data['town_water'], self.data['agr_water']])

        '''相关参数'''
        # 水库相关参数，包括死库容，初始库容，损失系数，水位库容曲线系数
        self.min_v = 4300  # 死库容-定值
        self.v0 = 21978  # 初始库容-定值
        self.B = 0.008  # 水库漏损系数-定值
        self.x0 = 76.8  # 牛顿插值-初始值
        self.eps = 0.0001  # 牛顿插值-迭代精度
        self.A = 8.5  # 发电水量参数-定值（发电机参数）
        self.hd = 29.5515591397849  # 逆推发电水量用到的下水位-定值

        self.real_elec_plan = []

    def reset_real_elec_plan(self, x):
        self.real_elec_plan = copy.deepcopy(x)
