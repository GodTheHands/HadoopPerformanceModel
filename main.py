import numpy as np
from timeline import MapTask, ReduceTask, construct_timeline, append_timeline
from precedenceTree import *

def test_construct_timeline():
    # 创建 Map 任务
    M = [MapTask(1, i, np.random.randint(5, 15), np.random.randint(1, 5)) for i in range(numMapTasks)]

    # 创建 Reduce 任务
    R = [ReduceTask(1, i + numMapTasks, np.random.randint(5, 15), np.random.randint(1, 5)) for i in range(numReduceTasks)]

    # 创建 Map 任务
    M1 = [MapTask(2, i, np.random.randint(5, 15), np.random.randint(1, 5)) for i in range(numMapTasks)]

    # 创建 Reduce 任务
    R1 = [ReduceTask(2, i + numMapTasks, np.random.randint(5, 15), np.random.randint(1, 5)) for i in
         range(numReduceTasks)]

    # 设定节点数和慢启动
    K = numNodes
    slow_start = False  # 设置是否启用慢启动

    # 构造时间线
    TL = construct_timeline(M, R, K, slow_start)
    TL = append_timeline(TL, M1, R1, K, slow_start)

    map_TL = {node: [task for task in tasks if isinstance(task, MapTask)] for node, tasks in TL.items()}
    reduce_TL = {node: [task for task in tasks if isinstance(task, ReduceTask)] for node, tasks in TL.items()}

    print("\nMap Timeline (map_TL):")
    for node, tasks in map_TL.items():
        print(f"\nNode {node}:")
        for task in tasks:
            print(
                f" {task.job_id} Task ID: {task.task_id}, Start Time: {task.st}, End Time: {task.et}, Duration: {task.d}, Assigned Node: {task.an}")

    print("\nReduce Timeline (reduce_TL):")
    for node, tasks in reduce_TL.items():
        print(f"\nNode {node}:")
        for task in tasks:
            print(
                f" {task.job_id} Task ID: {task.task_id}, Start Time: {task.st}, End Time: {task.et}, Duration: {task.d}, Assigned Node: {task.an}")


def ratio(i, j, AvgResponseTime):
    st_i, et_i = i.st, i.et
    st_j, et_j = j.st, j.et

    overlap = max(0, min(et_i, et_j) - max(st_i, st_j))
    avg_response_i = AvgResponseTime[i.task_type]
    alpha_value = overlap / avg_response_i if avg_response_i != 0 else 0
    return alpha_value


def calculate_alpha_12(map_TL, reduce_TL, AvgResponseTime):
    ratios = []

    # 遍历 map_TL 和 reduce_TL
    for map_tasks in map_TL.values():
        for reduce_tasks in reduce_TL.values():
            for map_task in map_tasks:
                for reduce_task in reduce_tasks:
                    # 检查 job_id 是否相同
                    if map_task.job_id == reduce_task.job_id:
                        # 计算 ratio
                        r = ratio(map_task, reduce_task, AvgResponseTime)
                        ratios.append(r)

    # 计算所有 ratio 的平均值
    alpha_12 = sum(ratios) / len(ratios) if ratios else 0
    return alpha_12


def calculate_alpha_21(map_TL, reduce_TL, AvgResponseTime):
    ratios = []

    # 遍历 map_TL 和 reduce_TL
    for map_tasks in map_TL.values():
        for reduce_tasks in reduce_TL.values():
            for map_task in map_tasks:
                for reduce_task in reduce_tasks:
                    # 检查 job_id 是否相同
                    if map_task.job_id == reduce_task.job_id:
                        # 计算 ratio
                        r = ratio(reduce_task, map_task, AvgResponseTime)
                        ratios.append(r)

    # 计算所有 ratio 的平均值
    alpha_21 = sum(ratios) / len(ratios) if ratios else 0
    return alpha_21


def calculate_beta_11(map_TL, AvgResponseTime):
    ratios = []
    for tasks in map_TL.values():
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if tasks[i].job_id != tasks[j].job_id:
                    r = ratio(tasks[i], tasks[j], AvgResponseTime)
                    ratios.append(r)
    return sum(ratios) / len(ratios) if ratios else 0


def calculate_beta_12(map_TL, reduce_TL, AvgResponseTime):
    ratios = []
    for map_tasks in map_TL.values():
        for reduce_tasks in reduce_TL.values():
            for map_task in map_tasks:
                for reduce_task in reduce_tasks:
                    if map_task.job_id != reduce_task.job_id:
                        r = ratio(map_task, reduce_task, AvgResponseTime)
                        ratios.append(r)
    return sum(ratios) / len(ratios) if ratios else 0


def calculate_beta_21(map_TL, reduce_TL, AvgResponseTime):
    ratios = []
    for map_tasks in map_TL.values():
        for reduce_tasks in reduce_TL.values():
            for map_task in map_tasks:
                for reduce_task in reduce_tasks:
                    if map_task.job_id != reduce_task.job_id:
                        r = ratio(reduce_task, map_task, AvgResponseTime)
                        ratios.append(r)
    return sum(ratios) / len(ratios) if ratios else 0


def calculate_beta_22(reduce_TL, AvgResponseTime):
    ratios = []
    for tasks in reduce_TL.values():
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if tasks[i].job_id != tasks[j].job_id:
                    r = ratio(tasks[i], tasks[j], AvgResponseTime)
                    ratios.append(r)
    return sum(ratios) / len(ratios) if ratios else 0


# A1:Initialize
numNodes = 3 # 节点数量
cpuPerNode = 8 # 每个节点的 CPU 数量
diskPerNode = 4 # 每个节点的磁盘数量

S = np.random.rand(2, 5) # S[i][k] 表示第 i 类任务在第 k 个任务中心的主流时间
AvgResponseTime = np.random.rand(2) # AvgResponseTime[i] 表示第 i 类任务的平均响应时间

numMapTasks = 10 # Map 任务数
numReduceTasks = 5 # Reduce 任务数
maxMapPerNode = 10 # 每个节点最大 Map 容器数
maxReducePerNode = 5 # 每个节点最大 Reduce 容器数

# A2: Construct Precedence Tree
M = [MapTask(1, i, np.random.randint(5, 15), np.random.randint(1, 5)) for i in range(numMapTasks)]
R = [ReduceTask(1, i + numMapTasks, np.random.randint(5, 15), np.random.randint(1, 5)) for i in range(numReduceTasks)]
slow_start = False

# 构造时间线
TL = construct_timeline(M, R, numNodes, slow_start)

map_TL = {node: [task for task in tasks if isinstance(task, MapTask)] for node, tasks in TL.items()}
reduce_TL = {node: [task for task in tasks if isinstance(task, ReduceTask)] for node, tasks in TL.items()}

# A3:Precedence Tree

priority_tree = build_priority_tree(map_TL, reduce_TL)
priority_tree = convert_to_binary_tree(priority_tree)

