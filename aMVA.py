import numpy as np

class aMVA:
    def __init__(self, num_task_classes, num_nodes, max_tasks_per_node):
        self.num_task_classes = num_task_classes
        self.num_nodes = num_nodes
        self.max_tasks_per_node = max_tasks_per_node

        # 初始化模型参数
        self.response_time = np.zeros(num_task_classes)  # 每个任务类别的响应时间
        self.visit_ratios = np.ones((num_task_classes, num_nodes))  # 访问率，假设均匀分布
        self.service_time = np.random.rand(num_task_classes, num_nodes)  # 服务时间
        self.queue_length = np.zeros((num_task_classes, num_nodes))  # 队列长度

    def set_service_time(self, task_class, node, time):
        self.service_time[task_class, node] = time

    def set_visit_ratios(self, task_class, ratios):
        if len(ratios) != self.num_nodes:
            raise ValueError("Visit ratios must mathc the number of nodes.")
        self.visit_ratios[task_class] = ratios

    def estimate_response_time(self, iteration_limit=100, tolerance=1e-5):
        num_classes, num_nodes = self.num_task_classes, self.num_nodes

        for _ in range(iteration_limit):
            prev_response_time = self.response_time.copy()

            for c in range(num_classes):
                for k in range(num_nodes):
                    self.queue_length[c, k] = self.response_time[c] * self.visit_ratios[c, k] / self.service_time[c, k]

                total_tasks = np.sum(self.queue_length[c])

                if total_tasks > self.max_tasks_per_node:
                    scale_factor = self.max_tasks_per_node / total_tasks
                    self.queue_length[c] *= scale_factor

                self.response_time[c] = np.sum(
                    self.visit_ratios[c] * (self.service_time[c] * (1 + self.queue_length[c]))
                )

                if np.max(np.abs(self.response_time - prev_response_time)) < tolerance:
                    break;

                return self.response_time