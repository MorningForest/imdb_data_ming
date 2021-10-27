# coding=utf-8
from .utils import load_logger
import itertools

class Apriori:
    def __init__(self, data_set, log_path:str):
        self.ds = data_set
        self.logger = load_logger(log_path)

    def generate_freq_supports(self, item_set, min_support):
        """从候选项集中选出频繁项集并计算支持度"""
        self.logger.info('筛选频繁项集并计算支持度')
        freq_set = set()  # 保存频繁项集元素
        item_count = {}  # 保存元素频次，用于计算支持度
        supports = {}  # 保存支持度

        ## 计数
        for rc in self.ds:
            for item in item_set:
                if item.issubset(rc):
                    if item in item_count:
                        item_count[item] += 1
                    else: item_count[item] = 1
        data_len = float(len(self.ds))
        #计算支持度
        for item in item_count:
            if item_count[item] >= min_support:
                freq_set.add(item)
                supports[item] = item_count[item]
        # for item in item_count:
        #     if (item_count[item] / data_len) >= min_support:
        #         freq_set.add(item)
        #         supports[item] = item_count[item] / data_len

        return freq_set, supports

    def generate_new_combinations(self, freq_set, k):
        """
        根据频繁项集，生成新的候选项1项集
        参数：频繁项集列表 freq_set 与项集元素个数 k
        """
        self.logger.info('生成新组合')
        new_combinations = set()  # 保存新组合
        sets_len = len(freq_set)  # 集合含有元素个数，用于遍历求得组合
        freq_set_list = list(freq_set)  # 集合转为列表用于索引

        for i in range(sets_len):
            for j in range(i + 1, sets_len):
                l1 = list(freq_set_list[i])
                l2 = list(freq_set_list[j])
                # print(l1, l2)
                l1.sort()
                l2.sort()

                # 若两个集合的前k-2个项相同时,则将两个集合合并
                if l1[0:k - 2] == l2[0:k - 2]:
                    freq_item = freq_set_list[i] | freq_set_list[j]
                    new_combinations.add(freq_item)

        return new_combinations

    def apriori(self, min_support, max_len=None):
        """循环生成候选集并计算其支持度"""

        max_items = 2  # 初始项集元素个数
        freq_sets = []  # 保存所有频繁项集
        supports = {}  # 保存所有支持度

        # 候选项1项集
        c1 = set()
        for items in self.ds:
            for item in items:
                item_set = frozenset([item])
                c1.add(item_set)

        # 频繁项1项集及其支持度
        l1, support1 = self.generate_freq_supports(c1, min_support)

        freq_sets.append(l1)
        supports.update(support1)

        if max_len is None:
            max_len = float('inf')

        while max_items and max_items <= max_len:
            # 生成候选集
            ci = self.generate_new_combinations(freq_sets[-1], max_items)
            # 生成频繁项集和支持度
            li, support = self.generate_freq_supports(ci, min_support)

            # 如果有频繁项集则进入下个循环
            if li:
                freq_sets.append(li)
                supports.update(support)
                max_items += 1
            else:
                max_items = 0

        return freq_sets, supports

    def association_rules(self, freq_sets, supports, min_conf):
        """生成关联规则"""
        self.logger.info('生成关联规则')
        rules = []
        max_len = len(freq_sets)
        fp = open('association_rules.txt', 'a+', encoding="utf8")
        # 筛选符合规则的频繁集计算置信度，满足最小置信度的关联规则添加到列表
        for k in range(max_len - 1):
            for freq_set in freq_sets[k]:
                for sub_set in freq_sets[k + 1]:
                    if freq_set.issubset(sub_set):
                        frq = supports[sub_set]
                        conf = supports[sub_set] / supports[freq_set]
                        rule = (freq_set, sub_set - freq_set, frq, conf)
                        if conf >= min_conf:
                            print(freq_set, "-->", sub_set - freq_set, 'frq:', frq, 'conf:', conf)
                            fp.write(str(freq_set) + "-->" + str(sub_set - freq_set) + 'frq:' + str(frq) + 'conf:' +str(conf)+'\n')
                            rules.append(rule)
        fp.close()
        return rules
