import numpy as np

class Task:
    def __init__(self, task_id, duration, shuffle_duration=0):
        self.task_id = task_id
        self.d = duration  # 任务的计算持续时间
        self.sd = shuffle_duration  # shuffle 阶段的持续时间，默认为 0
        self.an = None  # 分配的节点
        self.st = None  # 任务开始时间
        self.et = None  # 任务结束时间

    def __repr__(self):
        return f"Task {self.task_id}: Duration={self.d}, ShuffleDuration={self.sd}"

class MapTask(Task):
    def __init__(self, job_id, task_id, duration, shuffle_duration=0):
        super().__init__(task_id, duration, shuffle_duration)
        self.task_type = 'Map'
        self.job_id = job_id

class ReduceTask(Task):
    def __init__(self, job_id, task_id, duration, shuffle_duration=0):
        super().__init__(task_id, duration, shuffle_duration)
        self.task_type = 'Reduce'
        self.job_id = job_id

def initialize_TL(K):
    TL = {i: [] for i in range(K)}  # 初始化 K 条时间线
    return TL

def construct_timeline(M, R, K, slow_start=False):
    TL = initialize_TL(K)

    # 处理 Map 任务
    for m in M:
        # 找到最少已用时间线
        i = min(TL, key=lambda x: sum(task.d for task in TL[x]), default=0)
        m.an = i

        m.st = TL[i][-1].et if TL[i] else 0
        m.et = m.st + m.d
        TL[i].append(m)

    # 选择边界任务，根据是否慢启动 (slow_start)
    if slow_start:
        border = TL[min(TL, key=lambda x: sum(task.d for task in TL[x]))][-1].et
    else:
        border = TL[max(TL, key=lambda x: sum(task.d for task in TL[x]))][-1].et

    # 处理 Reduce 任务
    for r in R:
        # 找到最少已用时间线
        i = min(TL, key=lambda x: sum(task.d for task in TL[x]))
        r.an = i

        # 计算 Reduce 任务的开始时间，保证不早于边界任务
        r.st = max(TL[i][-1].et if TL[i] else 0, border)

        # 计算 Shuffle 时间，Map 任务的 Shuffle 时间会对 Reduce 任务产生影响
        for m in M:
            if m.an != i:  # 如果 Map 任务和 Reduce 任务不在同一节点上
                r.d += m.sd / len(R)  # 加上 shuffle 延迟

        r.et = r.st + r.d
        TL[i].append(r)

    return TL

def append_timeline(TL, M, R, K, slow_start=False):
    # 处理 Map 任务
    for m in M:
        # 找到最少已用时间线
        i = min(TL, key=lambda x: sum(task.d for task in TL[x]), default=0)
        m.an = i

        m.st = TL[i][-1].et if TL[i] else 0
        m.et = m.st + m.d
        TL[i].append(m)

    # 选择边界任务，根据是否慢启动 (slow_start)
    if slow_start:
        border = TL[min(TL, key=lambda x: sum(task.d for task in TL[x]))][-1].et
    else:
        border = TL[max(TL, key=lambda x: sum(task.d for task in TL[x]))][-1].et

    # 处理 Reduce 任务
    for r in R:
        # 找到最少已用时间线
        i = min(TL, key=lambda x: sum(task.d for task in TL[x]))
        r.an = i

        # 计算 Reduce 任务的开始时间，保证不早于边界任务
        r.st = max(TL[i][-1].et if TL[i] else 0, border)

        # 计算 Shuffle 时间，Map 任务的 Shuffle 时间会对 Reduce 任务产生影响
        for m in M:
            if m.an != i:  # 如果 Map 任务和 Reduce 任务不在同一节点上
                r.d += m.sd / len(R)  # 加上 shuffle 延迟

        r.et = r.st + r.d
        TL[i].append(r)

    return TL