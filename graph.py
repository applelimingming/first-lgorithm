import networkx as nx
import time
from argsss import parameter_parser
import numpy as np
import pandas as pd
import math
import csv
from tqdm import tqdm

class NEWGraph:

    def __init__(self, args):

        self.g = nx.read_edgelist('data/'+args.old_file)
        self.g_new = self.g
        self.node_number = self.g.number_of_nodes()
        self.edge_number = self.g.number_of_edges()
        self.nodes = self.g.nodes()
        self.edges = self.g.edges()
        self.count = 0  # 计算加边的次数
        self.max_degree = 0   # 节点的最大度
        print('number of nodes',self.node_number)
        self.A = np.array(nx.adjacency_matrix(self.g).todense())
        print(self.A)

        # 找出节点的最大度
        for node in self.nodes:
            if self.g.neighbors(node) != '':
                node_degree = len(self.g.neighbors(node))
                if node_degree > self.max_degree:
                    self.max_degree = node_degree

        self.nodes_degree = [[] for i in np.arange(0,self.max_degree+3)]

        # 把相同度的点加入相同的下标
        for u in self.nodes:
            u_degree = len(self.g.neighbors(u))
            self.nodes_degree[u_degree].append(u)
        print('最大度',self.max_degree)
        print(self.nodes_degree)

        # 开始加边的操作，如果两点之间的二阶相似性大于阈值就在两点之间加边
        for l_node in tqdm(self.nodes):
            # print(l_node)
            if args.rho != 1 and len(self.g.neighbors(l_node)) != 0:
                self.r_min = args.rho*len(self.g.neighbors(l_node))# 右边点的最小度
                self.r_man = len(self.g.neighbors(l_node))/args.rho# 有节点的最大度
                if self.r_min <= 0: self.r_min = 0
                if self.r_man >= self.max_degree: self.r_man = self.max_degree
                # print('rho range',(self.r_min,self.r_man))
                for i in np.arange(int(math.ceil(self.r_min)), int(np.floor(self.r_man))):
                    # print('i',i)
                    if len(self.nodes_degree[i]) != 0:
                        # print('度为：',i,'的点都有:\033[0m',self.nodes_degree[i])
                        for r_node in self.nodes_degree[i]:
                            # print('度为',i,'的点v是：',r_node)
                            if int(r_node) != int(l_node) and int(self.A[int(l_node)][int(r_node)])==0 and int(self.A[int(r_node)][int(l_node)])==0:
                                # print('r_node:',r_node)
                                # 把节点的邻居节点并转成列表
                                self.l_node_neighbors = set(list(self.g.neighbors(l_node)))
                                self.r_node_neighbors = set(list(self.g.neighbors(r_node)))

                                # 返回图中两个节点的公共邻居。
                                self.common_neighbors = self.l_node_neighbors.intersection(self.r_node_neighbors)
                                self.all_neighbors = self.l_node_neighbors.union(self.r_node_neighbors)

                                self.pro = len(self.common_neighbors)/len(self.all_neighbors)
                                # print('pro',self.pro)
                                if self.pro > args.rho:
                                    # print('\033[1;45m  恭喜，找到一个加边的条件 \033[0m ')
                                    with open('dataafter/new_'+str(args.old_file)[:-4]+str(args.rho)+'.txt', 'a', newline='') as f:
                                        #加边
                                        # self.g_new.add_edge(l_node,r_node)
                                        # 将新加的边写入txt文件中
                                        f.writelines([str(l_node)+' ',str(r_node)])
                                        f.writelines('\n')
                                        self.A[int(l_node)][int(r_node)] == 1
                                        self.A[int(r_node)][int(l_node)] == 1
                                        self.count = self.count+1
                                        print('\033[1;45m 第 \033[0m',self.count,'次加边')

            if args.rho == 1 and len(self.nodes_degree[len(self.g.neighbors(l_node))]) != 0:
                for r_node in self.nodes_degree[len(self.g.neighbors(l_node))]:
                        if r_node != l_node and nx.has_path(self.g, l_node, r_node) == False:
                            self.l_node_neighbors = set(list(self.g.neighbors(l_node)))
                            self.r_node_neighbors = set(list(self.g.neighbors(r_node)))

                            # 返回图中两个节点的公共邻居。
                            self.common_neighbors = self.l_node_neighbors.intersection(self.r_node_neighbors)
                            self.all_neighbors = self.l_node_neighbors.union(self.r_node_neighbors)

                            self.pro = len(self.common_neighbors) / len(self.all_neighbors)
                            if self.pro > args.rho:
                                with open('dataafter/'+'new_'+args.graph_file, 'w', newline=' ') as f:
                                    # 加边
                                    print('开始加边操作，请稍后。。。。')
                                    # self.g_new.add_edge(l_node, r_node)
                                    f.writelines([str(l_node) + ' ', str(r_node)])
                                    f.writelines('\n')
                                    self.count = self.count + 1
                                    print('第', self.count, '次加边')


        # return self.g_new


if __name__ == '__main__':

    args = parameter_parser()
    time1 = time.time()
    NEWGraph(args)
    time2 = time.time()
    print('加边的时间为：',time2-time1)