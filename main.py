import numpy as np
from numpy.ma.extras import average

from timeline import MapTask, ReduceTask, construct_timeline, append_timeline
from precedenceTree import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#Modified
def build_global_timeline(map_TL, reduce_TL):
    """
    构建全局时间线，包括所有任务的时间、资源使用情况和依赖关系。

    :param map_TL: Map任务的时间线
    :param reduce_TL: Reduce任务的时间线
    :return: 全局时间线
    """
    global_timeline = []

    # 整合 Map 和 Reduce 任务的时间线
    for node, map_tasks in map_TL.items():
        for task in map_tasks:
            global_timeline.append({
                "Task ID": task.task_id,
                "Job ID": task.job_id,
                "Type": "Map",
                "Assigned Node": task.an,
                "Start Time": task.st,
                "End Time": task.et,
                "Duration": task.et - task.st,
                "Resources": f"CPU: {cpuPerNode}, Disk: {diskPerNode}",
                "Dependencies": task.dependencies if hasattr(task, "dependencies") else None
            })

    for node, reduce_tasks in reduce_TL.items():
        for task in reduce_tasks:
            global_timeline.append({
                "Task ID": task.task_id,
                "Job ID": task.job_id,
                "Type": "Reduce",
                "Assigned Node": task.an,
                "Start Time": task.st,
                "End Time": task.et,
                "Duration": task.et - task.st,
                "Resources": f"CPU: {cpuPerNode}, Disk: {diskPerNode}",
                "Dependencies": task.dependencies if hasattr(task, "dependencies") else None
            })

    # 按任务开始时间排序
    global_timeline.sort(key=lambda x: x["Start Time"])

    return global_timeline

def visualize_timeline(global_timeline):
    """
    使用 Matplotlib 可视化最终的时间线
    :param global_timeline: 全局时间线列表，每个元素是任务的详细信息字典
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 定义颜色映射 (Map 和 Reduce)
    task_colors = {"Map": "skyblue", "Reduce": "orange"}
    y_labels = []
    y_positions = []

    for i, task in enumerate(global_timeline):
        # 解析任务信息
        task_id = task["Task ID"]
        job_id = task["Job ID"]
        start_time = task["Start Time"]
        end_time = task["End Time"]
        duration = end_time - start_time
        task_type = task["Type"]
        assigned_node = task["Assigned Node"]

        # 绘制任务条 (start_time ~ end_time)
        ax.broken_barh([(start_time, duration)], (assigned_node * 2, 1.5),
                       facecolors=task_colors[task_type], edgecolor="black", label=task_type)

        # 添加标签信息
        y_labels.append(f"Node {assigned_node}")
        y_positions.append(assigned_node * 2 + 0.75)
        ax.text(start_time + duration / 2, assigned_node * 2 + 0.75, f"Job-{job_id}\nT-{task_id}",
                ha='center', va='center', fontsize=8, color='black')

    # 配置图表轴
    ax.set_xlabel("Time")
    ax.set_ylabel("Nodes")
    ax.set_yticks(list(set(y_positions)))
    ax.set_yticklabels(list(set(y_labels)))
    ax.set_title("Task Execution Timeline (Gantt Chart)")

    # 添加图例
    map_patch = mpatches.Patch(color="skyblue", label="Map Tasks")
    reduce_patch = mpatches.Patch(color="orange", label="Reduce Tasks")
    ax.legend(handles=[map_patch, reduce_patch])

    plt.tight_layout()
    plt.show()


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

    overlap = max(0, min(et_i, et_j) - max(st_i, st_j))#计算重叠时间
    avg_response_i = AvgResponseTime[i.task_type]#标准化重叠系数
    alpha_value = overlap / avg_response_i if avg_response_i != 0 else 0
    return alpha_value


def calculate_alpha_12(map_TL, reduce_TL, AvgResponseTime):
    ratios = []

    # 遍历 map_TL 和 reduce_TL
    for map_tasks in map_TL.values():
        for reduce_tasks in reduce_TL.values():
            for map_task in map_tasks:
                for reduce_task in reduce_tasks:
                    # 检查 job_id 是否相同,计算工作内重叠
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
A = np.random.rand(2, 5)#队列长度，也设定为随机
m=cpuPerNode
AvgResponseTime = np.random.rand(2) # AvgResponseTime[i] 表示第 i 类任务的平均响应时间

numMapTasks = 10  # Map 任务数
numReduceTasks = 5  # Reduce 任务数
maxMapPerNode = 10  # 每个节点最大 Map 容器数
maxReducePerNode = 5  # 每个节点最大 Reduce 容器数

# A2: 构建任务并初始化时间线
# 创建 Map 任务
M = [MapTask(1, i, np.random.randint(5, 15), np.random.randint(2, 5)) for i in range(numMapTasks)]

# 创建 Reduce 任务
R = [ReduceTask(1, i + numMapTasks, np.random.randint(5, 15), np.random.randint(3, 6)) for i in range(numReduceTasks)]

# 创建 Map 任务
M1 = [MapTask(2, i, np.random.randint(5, 15), np.random.randint(2, 5)) for i in range(numMapTasks)]

# 创建 Reduce 任务
R1 = [ReduceTask(2, i + numMapTasks, np.random.randint(5, 15), np.random.randint(4, 7)) for i in
      range(numReduceTasks)]

# 创建 Map 任务
M2 = [MapTask(3, i, np.random.randint(5, 15), np.random.randint(2, 5)) for i in range(numMapTasks)]

# 创建 Reduce 任务
R2 = [ReduceTask(3, i + numMapTasks, np.random.randint(5, 15), np.random.randint(1, 5)) for i in
      range(numReduceTasks)]
slow_start = False

# 构造时间线
TL = construct_timeline(M, R, numNodes, slow_start)
# 构造时间线
TL = append_timeline(TL, M1, R1,numNodes , slow_start)
TL = append_timeline(TL, M2, R2,numNodes , slow_start)
# 提取 Map 和 Reduce 的时间线
map_TL = {node: [task for task in tasks if isinstance(task, MapTask)] for node, tasks in TL.items()}
reduce_TL = {node: [task for task in tasks if isinstance(task, ReduceTask)] for node, tasks in TL.items()}

# # 打印 Map 任务详情
# print("\nMap Task Details:")
# for map_tasks in map_TL.values():
#     for map_task in map_tasks:
#         print(
#             f"Map Task ID: {map_task.task_id}, Job ID: {map_task.job_id}, Assigned Node: {map_task.an}, Start Time: {map_task.st}, End Time: {map_task.et}"
#         )
#
# # 打印 Reduce 任务详情
# print("\nReduce Task Details:")
# for reduce_tasks in reduce_TL.values():
#     for reduce_task in reduce_tasks:
#         print(
#             f"Reduce Task ID: {reduce_task.task_id}, Job ID: {reduce_task.job_id}, Assigned Node: {reduce_task.an}, Start Time: {reduce_task.st}, End Time: {reduce_task.et}"
#         )

# 构建优先级树
priority_tree = build_priority_tree(map_TL, reduce_TL)
priority_tree = convert_to_binary_tree(priority_tree)


#Modified
 # A3: Estimate intra and inter job's overlap factors
def estimate_overlap_factors(map_TL, reduce_TL, AvgResponseTime):
    alpha_12 = calculate_alpha_12(map_TL, reduce_TL, AvgResponseTime)
    alpha_21 = calculate_alpha_21(map_TL, reduce_TL, AvgResponseTime)
    beta_11 = calculate_beta_11(map_TL, AvgResponseTime)
    beta_12 = calculate_beta_12(map_TL, reduce_TL, AvgResponseTime)
    beta_21 = calculate_beta_21(map_TL, reduce_TL, AvgResponseTime)
    beta_22 = calculate_beta_22(reduce_TL, AvgResponseTime)

    print(f"Alpha_12: {alpha_12}, Alpha_21: {alpha_21}")
    print(f"Beta_11: {beta_11}, Beta_12: {beta_12}, Beta_21: {beta_21}, Beta_22: {beta_22}")

    return {
        "alpha_12": alpha_12,
        "alpha_21": alpha_21,
        "beta_11": beta_11,
        "beta_12": beta_12,
        "beta_21": beta_21,
        "beta_22": beta_22,
    }
# A4: Estimate task response time
    #Modified
def calculate_task_response_time(tasks, S, m, A):
    for task in tasks:
            # 获取任务信息
        task_class = task.task_type
        node = task.an  # Assigned Node
        A_ik = A[task_class][node]
        S_ik = S[task_class][node]

            # 使用 MVA 公式计算响应时间
        R_ik = S_ik * (1 + A_ik) / min(1 + A_ik, m)
        task.response_time = R_ik  # 将结果赋值给任务

        print(f"Task ID: {task.task_id}, Assigned Node: {node}, Response Time: {R_ik:.2f}")
#Modifed 迭代测试
previous_avg_response_time = None#保存前一次的平均响应时间
converged = False
tolerance = 1e-15
max_iterations = 100  # 最大迭代次数

for iteration in range(max_iterations):


    overlap_factors = estimate_overlap_factors(map_TL, reduce_TL, AvgResponseTime)

    # 动态更新队列长度 A
    for task in M + R:
        task_class = task.task_type
        node = task.an
        A[task_class][node] = (
                                      overlap_factors["alpha_12"] + overlap_factors["alpha_21"] +
                                      overlap_factors["beta_11"] + overlap_factors["beta_12"] +
                                      overlap_factors["beta_21"] + overlap_factors["beta_22"]
                              ) / 6.0  # 平均重叠因子


    calculate_task_response_time(M, S, m, A)
    calculate_task_response_time(R, S, m, A)

    #A5 计算作业平均响应时间
    total_response_time = calculate_time_top_down(priority_tree)
    average_response_time = total_response_time / (numMapTasks + numReduceTasks)
    print(average_response_time)
    # Convergence check
    if previous_avg_response_time is not None:
        if abs(average_response_time - previous_avg_response_time) < tolerance:
            print("Converged!")
            converged = True
            # A7: 构建时间线
            global_timeline = build_global_timeline(map_TL, reduce_TL)
            visualize_timeline(global_timeline)
            break

    previous_avg_response_time = average_response_time


# 循环结束后的处理
if not converged:
    print("收敛失败。")
else:
    print(f"在第 {iteration + 1} 次收敛")
print(f"平均作业响应时间: {previous_avg_response_time:.4f}")