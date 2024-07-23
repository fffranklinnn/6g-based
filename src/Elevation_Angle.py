import win32com.client
import datetime
import os
import time
import pandas as pd


def Elevation(satellite_num, starttime, stoptime):
    stk_app = win32com.client.Dispatch('STK11.Application')#连接到已经安装并运行的STK应用程序
    stk_app.Visible = True#使STK应用程序窗口可见
    stk_root = stk_app.Personality2#获取stk根对象
    stk_scenario = stk_root.CurrentScenario#获取当前场景对象

    # 获取卫星对象
    satellite = stk_scenario.Children(satellite_num)#获取指定编号的卫星对象
    ground_station = stk_scenario.Children('Beijing')  # 获取地面站对象

    # 获取卫星和地面站之间的访问
    access = satellite.GetAccessToObject(ground_station)

    # 获取卫星仰角数据
    satelevation = access.DataProviders.GetDataPrvTimeVarFromPath(
         'Angles/FromObjectBodyElevation')
    result_ev = satelevation.Exec(
        starttime, stoptime, 50)#获取从starttime到stoptime时间范围内的仰角数据，数据间隔为50秒

    sat_ev = result_ev.DataSets[1].GetValues()#获取仰角数据并返回
    return sat_ev
    #print(satellite_num,':',sat_ev)

