from dataclasses import dataclass
import numpy as np
from builtin_interfaces.msg import Time
import math
from nav_msgs.msg import OccupancyGrid
import sys
import os

"""计算两个时间戳之间的纳秒差（stamp2 - stamp1）"""
def get_time_diff_ns(stamp1: Time, stamp2: Time) -> int:
    # 将两个时间都转换为纳秒总数
    total_ns1 = stamp1.sec * 10**9 + stamp1.nanosec
    total_ns2 = stamp2.sec * 10**9 + stamp2.nanosec
    return total_ns1 - total_ns2

@dataclass
class pos_data:
    name = None
    x = None
    y = None
    angle = None

def check_position(x, y, element_positions, min_dist):
    pos = True
    for element in element_positions:
        distance_vector = [element[0] - x, element[1] - y]
        distance = np.linalg.norm(distance_vector)
        if distance < min_dist:
            pos = False
    return pos

def set_random_position(name, element_positions):
    angle = np.random.uniform(-np.pi, np.pi)
    pos = False
    while not pos:
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-4.0, 4.0)
        pos = check_position(x, y, element_positions, 1.8)
    element_positions.append([x, y])
    eval_element = pos_data()
    eval_element.name = name
    eval_element.x = x
    eval_element.y = y
    eval_element.angle = angle
    return eval_element

def record_eval_positions(n_eval_scenarios=10):
    scenarios = []
    for _ in range(n_eval_scenarios):
        eval_scenario = []
        element_positions = [[-2.93, 3.17], [2.86, -3.0], [-2.77, -0.96], [2.83, 2.93]]
        for i in range(4, 8):
            name = "obstacle" + str(i + 1)
            eval_element = set_random_position(name, element_positions)
            eval_scenario.append(eval_element)

        eval_element = set_random_position("turtlebot3_waffle", element_positions)
        eval_scenario.append(eval_element)

        eval_element = set_random_position("target", element_positions)
        eval_scenario.append(eval_element)

        scenarios.append(eval_scenario)

    return scenarios

__all__ = ['wgs2gcj', 'gcj2wgs', 'gcj2wgs_exact',
           'distance', 'gcj2bd', 'bd2gcj', 'wgs2bd', 'bd2wgs']

earthR = 6378137.0

def outOfChina(lat, lng):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)

def transform(x, y):
    xy = x * y
    absX = math.sqrt(abs(x))
    xPi = x * math.pi
    yPi = y * math.pi
    d = 20.0*math.sin(6.0*xPi) + 20.0*math.sin(2.0*xPi)

    lat = d
    lng = d

    lat += 20.0*math.sin(yPi) + 40.0*math.sin(yPi/3.0)
    lng += 20.0*math.sin(xPi) + 40.0*math.sin(xPi/3.0)

    lat += 160.0*math.sin(yPi/12.0) + 320*math.sin(yPi/30.0)
    lng += 150.0*math.sin(xPi/12.0) + 300.0*math.sin(xPi/30.0)

    lat *= 2.0 / 3.0
    lng *= 2.0 / 3.0

    lat += -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*xy + 0.2*absX
    lng += 300.0 + x + 2.0*y + 0.1*x*x + 0.1*xy + 0.1*absX

    return lat, lng

def delta(lat, lng):
    ee = 0.00669342162296594323
    dLat, dLng = transform(lng-105.0, lat-35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((earthR * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLng = (dLng * 180.0) / (earthR / sqrtMagic * math.cos(radLat) * math.pi)
    return dLat, dLng

def wgs2gcj(wgsLat, wgsLng):
    if outOfChina(wgsLat, wgsLng):
        return wgsLat, wgsLng
    else:
        dlat, dlng = delta(wgsLat, wgsLng)
        return wgsLat + dlat, wgsLng + dlng

def gcj2wgs(gcjLat, gcjLng):
    if outOfChina(gcjLat, gcjLng):
        return gcjLat, gcjLng
    else:
        dlat, dlng = delta(gcjLat, gcjLng)
        return gcjLat - dlat, gcjLng - dlng

def gcj2wgs_exact(gcjLat, gcjLng):
    initDelta = 0.01
    threshold = 0.000001
    dLat = dLng = initDelta
    mLat = gcjLat - dLat
    mLng = gcjLng - dLng
    pLat = gcjLat + dLat
    pLng = gcjLng + dLng
    for i in range(30):
        wgsLat = (mLat + pLat) / 2
        wgsLng = (mLng + pLng) / 2
        tmplat, tmplng = wgs2gcj(wgsLat, wgsLng)
        dLat = tmplat - gcjLat
        dLng = tmplng - gcjLng
        if abs(dLat) < threshold and abs(dLng) < threshold:
            return wgsLat, wgsLng
        if dLat > 0:
            pLat = wgsLat
        else:
            mLat = wgsLat
        if dLng > 0:
            pLng = wgsLng
        else:
            mLng = wgsLng
    return wgsLat, wgsLng

def distance(latA, lngA, latB, lngB):
    pi180 = math.pi / 180
    arcLatA = latA * pi180
    arcLatB = latB * pi180
    x = (math.cos(arcLatA) * math.cos(arcLatB) *
         math.cos((lngA - lngB) * pi180))
    y = math.sin(arcLatA) * math.sin(arcLatB)
    s = x + y
    if s > 1:
        s = 1
    if s < -1:
        s = -1
    alpha = math.acos(s)
    distance = alpha * earthR
    return distance

def gcj2bd(gcjLat, gcjLng):
    if outOfChina(gcjLat, gcjLng):
        return gcjLat, gcjLng

    x = gcjLng
    y = gcjLat
    z = math.hypot(x, y) + 0.00002 * math.sin(y * math.pi)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * math.pi)
    bdLng = z * math.cos(theta) + 0.0065
    bdLat = z * math.sin(theta) + 0.006
    return bdLat, bdLng

def bd2gcj(bdLat, bdLng):
    if outOfChina(bdLat, bdLng):
        return bdLat, bdLng

    x = bdLng - 0.0065
    y = bdLat - 0.006
    z = math.hypot(x, y) - 0.00002 * math.sin(y * math.pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi)
    gcjLng = z * math.cos(theta)
    gcjLat = z * math.sin(theta)
    return gcjLat, gcjLng

def wgs2bd(wgsLat, wgsLng):
    return gcj2bd(*wgs2gcj(wgsLat, wgsLng))

def bd2wgs(bdLat, bdLng):
    return gcj2wgs(*bd2gcj(bdLat, bdLng))

def transfor_action(
    action,
    max_velocity=1.0,
    max_yawrate=45.0,
    prev_linear_velocity=0.0,
    max_acceleration=None,
    max_deceleration=None,
    step_time=0.1,
):
    """
    限制动作的范围
    :param action: 模型输出动作数组 [线速度, 角速度]（范围[-1,1]）
    :return: 限制后的动作数组
    """
    # 模型输出[-1,1] -> 目标线速度[0, max_velocity]
    target_linear = (action[0] + 1) / (2 / max_velocity)

    # 加速度/减速度限制（允许配置为负值，内部取绝对值）
    if max_acceleration is not None:
        max_acc = abs(max_acceleration)
        # 不能超过上一时刻速度 + 加速度步长，也不能超过 max_velocity
        target_linear = min(target_linear, prev_linear_velocity + max_acc * step_time, max_velocity)
    if max_deceleration is not None:
        max_dec = abs(max_deceleration)
        # 不能低于上一时刻速度 - 减速度步长，且不低于 0
        target_linear = max(target_linear, prev_linear_velocity - max_dec * step_time, 0.0)

    # 双保险：保持在线速度允许范围内
    target_linear = np.clip(target_linear, 0.0, max_velocity)

    # 角速度缩放到 [-max_yawrate, max_yawrate]（输入是-1到1）
    target_yawrate = action[1] * (max_yawrate / 180) * math.pi

    return [target_linear, target_yawrate]

def calculate_trajectory(start_pos, action, steps=10, step_size=0.2, resolution=0.1,edge_length=100):
    """
    根据动作计算机器狗预测轨迹 (修正坐标系)
    
    参数:
        start_pos: 起始位置 (row, col) - 图像坐标系
        action: 动作 [线速度, 角速度]
        steps: 预测步数
        step_size: 每步模拟的时间（秒）
    
    返回:
        轨迹点列表 [(row1, col1), (row2, col2), ...] - 图像坐标系
    """
    if not action:
        return []
    
    v, w = action  # 线速度(m/s), 角速度(rad/s)
    
    # 起始位置（图像坐标系）
    x, y = start_pos  # (x,y)
    
    # 轨迹点集合（避免重复）
    trajectory = []
    
    # 初始方向（朝向x轴正方向 - 向右）
    theta = 0.0
    
    # 根据动作模拟轨迹（使用标准笛卡尔坐标系）
    for _ in range(steps):
        # 更新方向：w > 0 时逆时针旋转（向y正方向）
        theta += w * step_size
        
        # 计算位移 (y方向使用减法适配图像坐标系)
        dx = v * step_size * np.cos(theta) / resolution
        dy = v * step_size * np.sin(theta) / resolution

        x += dx
        y += dy
        
        # 转换为整数网格坐标
        index_x = int(math.floor(x))
        index_y = int(math.floor(y))
        #print(f"计算位置: ({index_x}, {index_y})")
        
        # 确保位置在网格范围内
        if 0 <= index_x < edge_length and 0 <= index_y < edge_length:
            # 添加轨迹点（避免重复）
            point = (index_x, index_y)
            if point not in trajectory:
                trajectory.append(point)
    
    return trajectory