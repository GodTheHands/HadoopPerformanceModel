class TreeNode:
    def __init__(self, node_type, children=None, task=None):
        self.node_type = node_type
        self.children = children if children else []
        self.task = task
        self.left = None
        self.right = None

    def __repr__(self):
        if self.node_type == 'Task':
            return f"TaskNode(Task ID: {self.task.task_id}, Start Time: {self.task.st}, End Time: {self.task.et})"
        else:
            return f"{self.node_type}Node(Children: {len(self.children)})"

    def to_binary_tree(self):
        if not self.children:
            return self
        elif len(self.children) == 2:
            self.left = self.children[0]
            self.right = self.children[1]
            return self

        self.right = self.children[-1].to_binary_tree()

        if len(self.children) > 1:
            left_subtree_root = TreeNode(self.node_type, self.children[:-1])
            self.left = left_subtree_root.to_binary_tree()

        return self


def build_priority_tree(map_TL, reduce_TL):
    LChild = build_subtree(map_TL)
    RChild = build_subtree(reduce_TL)
    # 根节点声明
    root = TreeNode('S', children=[
        LChild, RChild
    ])

    root.left = LChild
    root.right = RChild

    return root


def build_subtree(TL):
    if not TL:
        return None

    # 并行执行不同 Node 的任务
    root = TreeNode('P')

    for node, tasks in TL.items():
        if tasks:
            s_node = TreeNode('S')

            for task in tasks:
                task_node = TreeNode('Task', task=task)
                s_node.children.append(task_node)

            root.children.append(s_node)

    return root


def print_tree(node, level=0):
    indent = "  " * level
    if node.node_type == 'Task':
        print(f"{indent}{node}")
    else:
        print(f"{indent}{node.node_type}Node:")
        for child in node.children:
            print_tree(child, level + 1)


def print_binary_tree(node, level=0):
    if node is None:
        return
    indent = "  " * level
    print(f"{indent}{node}")
    if node.left:
        print(f"{indent}Left:")
        print_binary_tree(node.left, level + 1)
    if node.right:
        print(f"{indent}Right:")
        print_binary_tree(node.right, level + 1)


def convert_to_binary_tree(root):
    childL = root.children[0]

    for idx, s_node in enumerate(childL.children):
        childL.children[idx] = s_node.to_binary_tree()

    childR = root.children[1]

    for idx, s_node in enumerate(childR.children):
        childR.children[idx] = s_node.to_binary_tree()

    root.left = convert_to_balanced_binary_tree(childL)
    root.right = convert_to_balanced_binary_tree(childR)

    return root

def convert_to_balanced_binary_tree(p_node):
    s_nodes = p_node.children

    if not s_nodes:
        return None

    if len(s_nodes) == 1:
        return s_nodes[0]

    mid = len(s_nodes) // 2
    left_s_nodes = s_nodes[:mid]
    right_s_nodes = s_nodes[mid:]

    left_root = build_binary_subtree(left_s_nodes)
    right_root = build_binary_subtree(right_s_nodes)

    root = TreeNode('P', children=[left_root, right_root])
    root.left = left_root
    root.right = right_root

    return root


def build_binary_subtree(s_nodes):
    if not s_nodes:
        return None

    # 如果只有一个节点，直接返回这个节点
    if len(s_nodes) == 1:
        return s_nodes[0]

    mid = len(s_nodes) // 2
    left_s_nodes = s_nodes[:mid]  # 左半部分
    right_s_nodes = s_nodes[mid:]  # 右半部分

    # 创建左子树
    left_root = build_binary_subtree(left_s_nodes)
    # 创建右子树
    right_root = build_binary_subtree(right_s_nodes)

    # 创建当前节点并连接左右子树
    root = TreeNode('P', children=[left_root, right_root])
    root.left = left_root
    root.right = right_root

    return root