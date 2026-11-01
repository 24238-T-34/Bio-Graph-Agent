"""
hypothesis_engine.py - 智能假说推演与科研探勘引擎 (Hypothesis & Discovery Engine)
===================================================================================
功能定位：图谱分析工作台功能三核心算法与调度中枢
核心逻辑：
1. 拓扑-内生语义坐标建模：基于图论连通分量、中心度与阶段一本体分类自动定位图谱结构洞；
2. 四大探勘模式：自动感知模式 (Auto Mode)、跨簇孤岛桥接、周界边缘薄层生长、意向定向探究；
3. 两阶段解耦流程：一次性推演输出 ~5 项高质量学术猜想卡片，学者按需挑选并一键触发并发裁判；
4. 多线程并发文献裁判：基于 WebSearcher 并发检索 PubMed 真实摘要，由 BioBrainAgent 终审裁决；
5. 结果精准分流：文献确证者提供一键回填主图谱并返回主界面；科研空白者留作专享学术灵感（支持子工程导出），绝不污染主图谱。
"""

import os
import re
import uuid
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np

from workbench_engine import RulePreFilter, OntologyManager
from WebSearcher import get_searcher


class HypothesisEngine:
    """智能假说推演与科研探勘引擎"""

    def __init__(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]],
                 workbench_state: Optional[Dict[str, Any]] = None,
                 agent: Optional[Any] = None,
                 searcher: Optional[Any] = None,
                 lang: str = "zh"):
        self.entities = entities or []
        self.relations = relations or []
        self.workbench_state = workbench_state or {}
        self.agent = agent
        self.searcher = searcher or get_searcher("PubMed")
        self.lang = lang

        # 构建图论模型与拓扑属性 (同时维护无向图与有向图)
        self.G = nx.Graph()
        self.DiG = nx.DiGraph()
        self._build_graph()

        # 确保阶段一本体数据可用（无缝 0-Token 保底）
        self.node_ontology = self._ensure_stage1_ontology()

        # 初始化图拓扑语义特征与消息传递 (0-Token, 毫秒级)
        self.base_vectors: Dict[str, Any] = {}
        self.fused_vectors: Dict[str, Any] = {}
        self._init_semantic_features()
        self._run_message_passing(iterations=2)

    def _build_graph(self):
        """构建底层无向图与有向图以进行连通分量与拓扑流向计算"""
        alias_map = {}
        for ent in self.entities:
            name = ent.get("standard_name", "").strip()
            if name:
                self.G.add_node(name, **ent)
                self.DiG.add_node(name, **ent)
                alias_map[name.lower()] = name
                for alias in ent.get("aliases", []):
                    if alias and isinstance(alias, str):
                        alias_map[alias.strip().lower()] = name

        for rel in self.relations:
            src_raw = rel.get("source", "").strip()
            tgt_raw = rel.get("target", "").strip()
            if not src_raw or not tgt_raw:
                continue

            # 别名与大小写归一化对齐
            src = src_raw if src_raw in self.G else alias_map.get(src_raw.lower(), src_raw)
            tgt = tgt_raw if tgt_raw in self.G else alias_map.get(tgt_raw.lower(), tgt_raw)

            if src in self.G and tgt in self.G:
                w = rel.get("weight", 1)
                r_type = rel.get("relation") or rel.get("type", "相关")
                edge_attrs = dict(rel)
                edge_attrs["weight"] = w
                edge_attrs["relation"] = r_type

                if self.G.has_edge(src, tgt):
                    self.G[src][tgt]["weight"] = self.G[src][tgt].get("weight", 1) + w
                else:
                    self.G.add_edge(src, tgt, **edge_attrs)

                if self.DiG.has_edge(src, tgt):
                    self.DiG[src][tgt]["weight"] = self.DiG[src][tgt].get("weight", 1) + w
                else:
                    self.DiG.add_edge(src, tgt, **edge_attrs)

    def _init_semantic_features(self):
        """0-Token 本地高维稠密特征编码"""
        cat_list = ["Gene_Protein", "Chemical_Drug", "Disease", "Phenotype", "Biological_Process", "Organism_Taxon", "Other"]
        cat_to_idx = {c: i for i, c in enumerate(cat_list)}

        for node in self.G.nodes():
            cat = self._get_node_category(node)
            c_idx = cat_to_idx.get(cat, 6)
            vec = np.zeros(24, dtype=float)
            vec[c_idx] = 2.0

            # 融入词元哈希（保证同名/相似词具有微弱局部亲和力）
            for tok in node.lower().split():
                h = abs(hash(tok)) % 8
                vec[16 + h] += 0.5

            norm = np.linalg.norm(vec)
            self.base_vectors[node] = vec / norm if norm > 0 else vec

    def _run_message_passing(self, iterations: int = 2):
        """
        图拓扑语义消息传递 (Message Passing)：
        语义向量沿关系边流动聚合，使每个节点吸收局部 2-Hop 调控微环境。
        """
        pol_map = {
            "正作用": 1.0, "Positive_Regulation": 1.0, "Activation": 1.0, "Promotion": 1.0,
            "负作用": -1.0, "Negative_Regulation": -1.0, "Inhibition": -1.0, "Suppression": -1.0,
            "相关": 0.5, "Correlation": 0.5, "Association": 0.5,
            "包含": 0.3, "Part_Of": 0.3
        }

        h = {k: v.copy() for k, v in self.base_vectors.items()}
        all_nodes = list(self.G.nodes())
        if not all_nodes:
            self.fused_vectors = h
            return

        adj = {k: [] for k in all_nodes}
        for u, v, data in self.G.edges(data=True):
            r_type = data.get("relation", "相关")
            pol = pol_map.get(r_type, 0.5)
            adj[u].append((v, pol, 1.0))
            adj[v].append((u, pol, 1.0))

        for it in range(iterations):
            new_h = {}
            for node in all_nodes:
                neighbors = adj.get(node, [])
                if not neighbors:
                    new_h[node] = h[node]
                    continue
                msg = np.zeros(24, dtype=float)
                for nbr, pol, direction in neighbors:
                    msg += pol * direction * h.get(nbr, np.zeros(24))
                msg = msg / max(1, len(neighbors))
                comb = 0.5 * h[node] + 0.5 * msg
                norm = np.linalg.norm(comb)
                new_h[node] = comb / norm if norm > 0 else comb
            h = new_h

        self.fused_vectors = h

    def _calc_semantic_similarity(self, u: str, v: str) -> float:
        """计算两个节点在融合微环境语义后的余弦相似度"""
        vu = self.fused_vectors.get(u)
        vv = self.fused_vectors.get(v)
        if vu is None or vv is None:
            return 0.0
        return float(np.dot(vu, vv))

    def _ensure_stage1_ontology(self) -> Dict[str, Dict[str, Any]]:
        """
        保证所有节点具备阶段一本体分类数据。
        若工作台已运行阶段一，直接提取；若未运行，以 0-Token、0 毫秒规则初筛快速保底。
        """
        ontology_cache = {}
        existing_ea = self.workbench_state.get("entity_annotations", {}) or self.workbench_state.get("entity_analysis", {})

        om = OntologyManager()
        biored = om.get_preset("universal_biored") or {}

        for ent in self.entities:
            name = ent.get("standard_name", "").strip()
            if not name:
                continue

            if name in existing_ea:
                ea_record = existing_ea[name]
                cat = ea_record.get("primary_type") or ea_record.get("category", "Gene_Protein")
                sub = ea_record.get("sub_type") or ea_record.get("subcategory", "")
                lbl = ea_record.get("tags", [cat])[0] if ea_record.get("tags") else ea_record.get("label", cat)
                ontology_cache[name] = {
                    "category": cat,
                    "primary_type": cat,
                    "subcategory": sub,
                    "label": lbl
                }
            else:
                # 0-Token 快速规则初筛保底
                aliases = ent.get("aliases", [])
                rule_match = RulePreFilter.classify(name, aliases=aliases, ontology=biored, language=self.lang)
                if rule_match:
                    cat = rule_match.get("primary_type", "Gene_Protein")
                    ontology_cache[name] = {
                        "category": cat,
                        "primary_type": cat,
                        "subcategory": rule_match.get("sub_type", ""),
                        "label": rule_match.get("tags", [cat])[0] if rule_match.get("tags") else cat
                    }
                else:
                    ontology_cache[name] = {
                        "category": "Gene_Protein",
                        "primary_type": "Gene_Protein",
                        "subcategory": "Unspecified",
                        "label": "未知/待定"
                    }

        return ontology_cache

    def detect_auto_mode(self) -> Dict[str, str]:
        """
        拓扑-语义流场自感知：
        自动评估网络连通性、社区结构与调控极性平衡，推荐混合探勘策略。
        """
        is_en = (self.lang == "en")
        if len(self.G) < 2:
            return {
                "mode": "auto",
                "label": "全局自感知多元探勘 (Auto Ensemble)" if not is_en else "Autonomous Multi-Motif Ensemble",
                "reason": "当前网络节点较少，推荐向外进行周界边缘薄层延展。" if not is_en else "Graph has few nodes; peripheral expansion recommended."
            }

        components = sorted(list(nx.connected_components(self.G)), key=len, reverse=True)
        total_nodes = len(self.G)
        giant = components[0]
        giant_size = len(giant)
        giant_ratio = giant_size / max(1, total_nodes)
        sub_clusters = [c for c in components[1:] if len(c) >= 2]
        singletons = [c for c in components if len(c) == 1]

        # 调控极性统计
        pos_cnt = sum(1 for _, _, d in self.G.edges(data=True) if "正" in str(d.get("relation", "")) or "Positive" in str(d.get("relation", "")))
        neg_cnt = sum(1 for _, _, d in self.G.edges(data=True) if "负" in str(d.get("relation", "")) or "Negative" in str(d.get("relation", "")))

        balance_hint = ""
        if pos_cnt > 0 and neg_cnt == 0:
            balance_hint = "（检测到图谱正向驱动极度显著，多元组合将强化逆向负反馈稳态寻孔）" if not is_en else " (Strong activation detected; homeostatic negative feedback prioritized)"

        if len(sub_clusters) >= 1:
            reason_str = (
                f"图谱呈现分块聚集态（覆盖 {giant_size} 主干节点与 {len(sub_clusters)} 个独立子簇）。算法采用多元投资组合：重点加固跨模块脆弱瓶颈，并同步探索拓扑绕远捷径与逆向稳态闭环{balance_hint}。"
                if not is_en else
                f"Multi-cluster topology detected ({giant_size} nodes in main component, {len(sub_clusters)} sub-clusters). Multi-motif ensemble exploring bottleneck reinforcement, detour shortcuts, and homeostatic loops{balance_hint}."
            )
        else:
            single_info = f"，另有 {len(singletons)} 个游离单点" if singletons else ""
            reason_str = (
                f"图谱呈现核心主体高度连通态（核心网络覆盖 {giant_size}/{total_nodes} 节点，占比 {giant_ratio*100:.1f}%{single_info}）。算法采用多元投资组合：深挖拓扑绕远隐性捷径、探索跨功能社区结构洞与周界薄层级联{balance_hint}。"
                if not is_en else
                f"Predominantly connected network ({giant_size}/{total_nodes} nodes, {giant_ratio*100:.1f}%{single_info}). Multi-motif ensemble exploring detour shortcuts, community structural holes, and frontier cascades{balance_hint}."
            )

        return {
            "mode": "auto",
            "label": "全局自感知多元探勘 (Auto Ensemble)" if not is_en else "Autonomous Multi-Motif Ensemble",
            "reason": reason_str
        }

    def _get_node_category(self, node: str) -> str:
        return self.node_ontology.get(node, {}).get("category", "Gene_Protein")

    def _calc_compatibility_score(self, cat_a: str, cat_b: str) -> float:
        """
        本体语义相容性打分：
        根据生物学生化常识，为不同类型节点之间的互作潜力赋权。
        """
        ca, cb = cat_a.lower(), cat_b.lower()
        if ("drug" in ca or "chemical" in ca) and ("gene" in cb or "protein" in cb):
            return 3.5
        if ("drug" in cb or "chemical" in cb) and ("gene" in ca or "protein" in ca):
            return 3.5
        if ("gene" in ca or "protein" in ca) and ("process" in cb or "phenotype" in cb or "disease" in cb):
            return 3.0
        if ("gene" in cb or "protein" in cb) and ("process" in ca or "phenotype" in ca or "disease" in ca):
            return 3.0
        if ("gene" in ca or "protein" in ca) and ("gene" in cb or "protein" in cb):
            return 2.5
        if ("drug" in ca or "chemical" in ca) and ("disease" in cb or "phenotype" in cb):
            return 2.8
        if ("drug" in cb or "chemical" in cb) and ("disease" in ca or "phenotype" in ca):
            return 2.8
        return 1.5

    def _mine_detour_shortcuts(self, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        模体一：拓扑绕远捷径挖掘 (Detour Shortcut)
        寻找在消息传递后局部语义微环境极度吻合 (高余弦相似度)，但图谱中必须绕行 2~3 个中介的节点对，推演直接旁路短路。
        """
        if len(self.G) < 3:
            return []

        all_nodes = list(self.G.nodes())
        candidates = []

        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                u, v = all_nodes[i], all_nodes[j]
                if self.G.has_edge(u, v):
                    continue
                if not nx.has_path(self.G, u, v):
                    continue
                path = nx.shortest_path(self.G, u, v)
                if len(path) < 3 or len(path) > 4:
                    continue

                sim = self._calc_semantic_similarity(u, v)
                cat_u = self._get_node_category(u)
                cat_v = self._get_node_category(v)
                compat = self._calc_compatibility_score(cat_u, cat_v)
                score = sim * 3.0 + compat * 1.5

                c_nodes = []
                for n in path:
                    role = "source" if n == u else ("target" if n == v else "mediator")
                    c_nodes.append({"name": n, "category": self._get_node_category(n), "role": role})

                c_edges = []
                for idx in range(len(path) - 1):
                    p_s, p_t = path[idx], path[idx + 1]
                    r_type = self.G[p_s][p_t].get("relation", "相关")
                    c_edges.append({"source": p_s, "target": p_t, "relation": r_type, "status": "known"})
                c_edges.append({"source": u, "target": v, "relation": "直接短路串扰", "status": "hypothesized"})

                inter_names = ", ".join(path[1:-1])
                hint = (
                    f"环境语义高度重合 (相似度: {sim:.2f})，现有图谱绕经中介 [{inter_names}]，推测存在未被证实的直接旁路短路"
                    if self.lang != "en" else
                    f"High semantic overlap ({sim:.2f}) with detour via [{inter_names}]; direct shortcut hypothesized."
                )

                candidates.append({
                    "mode": "detour_shortcut",
                    "motif_type": "detour_shortcut",
                    "motif_label": "⚡ 捷径短路模体" if self.lang != "en" else "⚡ Detour Shortcut",
                    "node_a": u,
                    "category_a": cat_u,
                    "node_b": v,
                    "category_b": cat_v,
                    "circuit_nodes": c_nodes,
                    "circuit_edges": c_edges,
                    "score": score,
                    "rationale_hint": hint
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def _mine_cluster_bottlenecks(self, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        模体二：跨模块脆性瓶颈加固 (Bottleneck Reinforce)
        识别宏观功能子簇间脆弱或割裂的连接，推演建立深层跨簇冗余调控通道。
        """
        if len(self.G) < 4:
            return []

        components = sorted(list(nx.connected_components(self.G)), key=len, reverse=True)
        giant = self.G.subgraph(components[0])

        communities = []
        if len(components) >= 2 and len(components[1]) >= 2:
            communities = [components[0], components[1]]
            if len(components) >= 3 and len(components[2]) >= 2:
                communities.append(components[2])
        else:
            try:
                communities = sorted(list(nx_comm.greedy_modularity_communities(giant)), key=len, reverse=True)
            except Exception:
                communities = []

        if len(communities) < 2:
            return self._mine_detour_shortcuts(top_k=top_k)

        candidates = []
        for i in range(min(3, len(communities))):
            for j in range(i + 1, min(3, len(communities))):
                c_i, c_j = communities[i], communities[j]
                sub_i = self.G.subgraph(c_i)
                sub_j = self.G.subgraph(c_j)
                deg_i = dict(sub_i.degree())
                deg_j = dict(sub_j.degree())
                if not deg_i or not deg_j:
                    continue
                hub_i = sorted(deg_i.keys(), key=lambda n: deg_i[n], reverse=True)[0]
                hub_j = sorted(deg_j.keys(), key=lambda n: deg_j[n], reverse=True)[0]

                if self.G.has_edge(hub_i, hub_j):
                    continue

                cat_i = self._get_node_category(hub_i)
                cat_j = self._get_node_category(hub_j)
                compat = self._calc_compatibility_score(cat_i, cat_j)
                sim = self._calc_semantic_similarity(hub_i, hub_j)
                score = compat * 2.2 + (deg_i[hub_i] + deg_j[hub_j]) * 0.3 + sim * 1.5

                c_nodes = [{"name": hub_i, "category": cat_i, "role": "hub_source"}]
                c_edges = []
                if nx.has_path(self.G, hub_i, hub_j):
                    path = nx.shortest_path(self.G, hub_i, hub_j)
                    if len(path) <= 4:
                        for m_n in path[1:-1]:
                            c_nodes.append({"name": m_n, "category": self._get_node_category(m_n), "role": "bottleneck_gateway"})
                        for p_idx in range(len(path) - 1):
                            p_s, p_t = path[p_idx], path[p_idx + 1]
                            r_type = self.G[p_s][p_t].get("relation", "相关")
                            c_edges.append({"source": p_s, "target": p_t, "relation": r_type, "status": "known"})
                c_nodes.append({"name": hub_j, "category": cat_j, "role": "hub_target"})
                c_edges.append({"source": hub_i, "target": hub_j, "relation": "跨模块核心桥接", "status": "hypothesized"})

                hint = (
                    f"宏观功能模块 社区{i+1}（{hub_i}，度数{deg_i[hub_i]}）与 社区{j+1}（{hub_j}，度数{deg_j[hub_j]}）存在深层调控结构洞，推演加固跨界冗余桥梁"
                    if self.lang != "en" else
                    f"Structural hole between Community {i+1} ({hub_i}) and Community {j+1} ({hub_j}); cross-module bridge hypothesized."
                )

                candidates.append({
                    "mode": "cluster_bridge",
                    "motif_type": "bottleneck_reinforce",
                    "motif_label": "🛡️ 瓶颈加固模体" if self.lang != "en" else "🛡️ Bottleneck Reinforce",
                    "node_a": hub_i,
                    "category_a": cat_i,
                    "node_b": hub_j,
                    "category_b": cat_j,
                    "circuit_nodes": c_nodes,
                    "circuit_edges": c_edges,
                    "score": score,
                    "rationale_hint": hint
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def _mine_homeostasis_loops(self, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        模体三：逆向稳态闭环探索 (Homeostatic Feedback Loop)
        针对单向过度激活/正向传导链 (A -> B -> C)，计算逆向平衡力，推演缺失的负反馈自抑回路以维持稳态。
        """
        if len(self.DiG) < 3:
            return []

        candidates = []
        for u in self.DiG.nodes():
            for v in self.DiG.successors(u):
                rel_uv = self.DiG[u][v].get("relation", "")
                for w in self.DiG.successors(v):
                    if w == u:
                        continue
                    if self.G.has_edge(w, u):
                        continue

                    rel_vw = self.DiG[v][w].get("relation", "")
                    cat_w = self._get_node_category(w)
                    cat_u = self._get_node_category(u)
                    compat = self._calc_compatibility_score(cat_w, cat_u)
                    sim = self._calc_semantic_similarity(w, u)
                    score = compat * 2.0 + sim * 1.5 + (self.G.degree(u) + self.G.degree(w)) * 0.2

                    c_nodes = [
                        {"name": u, "category": cat_u, "role": "cascade_start"},
                        {"name": v, "category": self._get_node_category(v), "role": "cascade_effector"},
                        {"name": w, "category": cat_w, "role": "cascade_terminal"}
                    ]
                    c_edges = [
                        {"source": u, "target": v, "relation": rel_uv or "正作用", "status": "known"},
                        {"source": v, "target": w, "relation": rel_vw or "正作用", "status": "known"},
                        {"source": w, "target": u, "relation": "负反馈自抑", "status": "hypothesized"}
                    ]

                    hint = (
                        f"检测到顺向强激活传导级联 [{u} -> {v} -> {w}]，推演潜在的反向负反馈回路 [{w} 抑制 {u}] 以闭合维持系统稳态"
                        if self.lang != "en" else
                        f"Forward activation cascade [{u} -> {v} -> {w}] detected; homeostatic negative feedback [{w} -| {u}] hypothesized."
                    )

                    candidates.append({
                        "mode": "feedback_homeostasis",
                        "motif_type": "feedback_homeostasis",
                        "motif_label": "⚖️ 稳态闭环模体" if self.lang != "en" else "⚖️ Homeostatic Loop",
                        "node_a": w,
                        "category_a": cat_w,
                        "node_b": u,
                        "category_b": cat_u,
                        "circuit_nodes": c_nodes,
                        "circuit_edges": c_edges,
                        "score": score,
                        "rationale_hint": hint
                    })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def _mine_frontier_cascades(self, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        模体四：周界前沿薄层延展 (Frontier Cascade)
        寻找网络周界（度数 1~2）但本体重要的外围叶子节点，结合其关键中介向核心网络纵深延展。
        """
        if len(self.G) < 3:
            return []

        degrees = dict(self.G.degree())
        all_nodes = list(self.G.nodes())
        leafs = [n for n in all_nodes if degrees[n] <= 2 and degrees[n] > 0]
        hubs = [n for n in all_nodes if degrees[n] >= 3]
        if not hubs:
            hubs = sorted(all_nodes, key=lambda n: degrees[n], reverse=True)[:3]

        candidates = []
        for leaf in leafs:
            cat_l = self._get_node_category(leaf)
            nbrs = list(self.G.neighbors(leaf))
            if not nbrs:
                continue
            mediator = sorted(nbrs, key=lambda n: degrees[n], reverse=True)[0]
            cat_m = self._get_node_category(mediator)

            for hub in hubs:
                if hub == leaf or hub == mediator or self.G.has_edge(leaf, hub):
                    continue

                cat_h = self._get_node_category(hub)
                compat = self._calc_compatibility_score(cat_l, cat_h)
                sim = self._calc_semantic_similarity(leaf, hub)

                try:
                    path_len = nx.shortest_path_length(self.G, source=leaf, target=hub)
                except Exception:
                    path_len = 4

                score = compat * 2.2 + sim * 1.8 + path_len * 0.8

                c_nodes = [
                    {"name": leaf, "category": cat_l, "role": "frontier_leaf"},
                    {"name": mediator, "category": cat_m, "role": "cascade_mediator"},
                    {"name": hub, "category": cat_h, "role": "core_hub"}
                ]
                rel_lm = self.G[leaf][mediator].get("relation", "相关")
                c_edges = [
                    {"source": leaf, "target": mediator, "relation": rel_lm, "status": "known"},
                    {"source": mediator, "target": hub, "relation": "纵深级联调控", "status": "hypothesized"}
                ]

                hint = (
                    f"周界边缘实体 [{leaf}] 通过已知中介 [{mediator}] 深入偶联核心网络中枢 [{hub}]，推动认知向外薄层延展"
                    if self.lang != "en" else
                    f"Peripheral leaf [{leaf}] coupled with core hub [{hub}] via mediator [{mediator}]; frontier cascade hypothesized."
                )

                candidates.append({
                    "mode": "peripheral_growth",
                    "motif_type": "frontier_cascade",
                    "motif_label": "🌿 周界前沿级联" if self.lang != "en" else "🌿 Frontier Cascade",
                    "node_a": leaf,
                    "category_a": cat_l,
                    "node_b": hub,
                    "category_b": cat_h,
                    "circuit_nodes": c_nodes,
                    "circuit_edges": c_edges,
                    "score": score,
                    "rationale_hint": hint
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def _mine_intent_exploration(self, intent: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        模式五：学者科研意向定向探究 (Intent Exploration)
        基于学者设定的研究焦点词，在多模体候选库中精准锁定契合该意图的微型回路。
        """
        if not intent:
            return self._mine_detour_shortcuts(top_k=top_k)

        intent_lower = intent.lower()
        pool = []
        pool.extend(self._mine_detour_shortcuts(top_k=8))
        pool.extend(self._mine_cluster_bottlenecks(top_k=6))
        pool.extend(self._mine_homeostasis_loops(top_k=6))
        pool.extend(self._mine_frontier_cascades(top_k=6))

        for cand in pool:
            match_score = 0.0
            nodes_text = " ".join([cn["name"].lower() for cn in cand.get("circuit_nodes", [])])
            if any(w in intent_lower for w in ["药", "drug", "inhibitor", "抑制", "化合物"]):
                if any("drug" in cn.get("category", "").lower() or "chemical" in cn.get("category", "").lower() for cn in cand.get("circuit_nodes", [])):
                    match_score += 3.0
            if any(w in intent_lower for w in ["病", "cancer", "瘤", "症", "disease", "uti", "感染"]):
                if any("disease" in cn.get("category", "").lower() for cn in cand.get("circuit_nodes", [])):
                    match_score += 3.0
            for tok in intent_lower.split():
                if len(tok) >= 2 and tok in nodes_text:
                    match_score += 4.0
            cand["score"] += match_score
            cand["mode"] = "intent_exploration"

        pool.sort(key=lambda x: x["score"], reverse=True)
        return pool[:top_k]

    def mine_and_formulate(self, mode: str = "auto", intent: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        阶段一：图拓扑寻孔 + 多节点微型机制模体生成
        一次性推演输出 ~5 项高价值学术假说卡片（支持 Auto 多元投资组合）。
        """
        if len(self.G) < 2:
            return []

        resolved_mode = mode
        if mode == "auto":
            candidates = []
            candidates.extend(self._mine_detour_shortcuts(top_k=2))
            candidates.extend(self._mine_cluster_bottlenecks(top_k=1))
            candidates.extend(self._mine_homeostasis_loops(top_k=1))
            candidates.extend(self._mine_frontier_cascades(top_k=1))

            if len(candidates) < top_k:
                fillers = self._mine_detour_shortcuts(top_k=top_k) + self._mine_cluster_bottlenecks(top_k=top_k)
                for f in fillers:
                    if not any((c["node_a"] == f["node_a"] and c["node_b"] == f["node_b"]) or (c["node_a"] == f["node_b"] and c["node_b"] == f["node_a"]) for c in candidates):
                        candidates.append(f)
                    if len(candidates) >= top_k:
                        break
        elif mode == "detour_shortcut":
            candidates = self._mine_detour_shortcuts(top_k=top_k)
        elif mode in ["cluster_bridge", "bottleneck_reinforce"]:
            candidates = self._mine_cluster_bottlenecks(top_k=top_k)
        elif mode in ["feedback_homeostasis", "homeostasis"]:
            candidates = self._mine_homeostasis_loops(top_k=top_k)
        elif mode in ["peripheral_growth", "frontier_cascade"]:
            candidates = self._mine_frontier_cascades(top_k=top_k)
        elif mode == "intent_exploration":
            candidates = self._mine_intent_exploration(intent=intent, top_k=top_k)
        else:
            candidates = self._mine_detour_shortcuts(top_k=top_k)

        if not candidates:
            return []

        # 调用大模型中枢为每对模体生成专业叙事与明确科研方向
        if self.agent and hasattr(self.agent, "formulate_hypothesis_candidates"):
            llm_formulated = self.agent.formulate_hypothesis_candidates(
                candidate_pairs=candidates,
                intent=intent,
                lang=self.lang
            )
        else:
            llm_formulated = []

        cards = []
        for i, pair in enumerate(candidates):
            formulation = llm_formulated[i] if i < len(llm_formulated) else {}
            stmt = formulation.get("hypothesis_statement") or f"{pair['node_a']} 与 {pair['node_b']} 在局部机制回路中存在直接或级联调控互作"
            rel = formulation.get("proposed_relation") or "Positive_Regulation"
            rationale = formulation.get("biological_rationale") or pair.get("rationale_hint", "")
            direction = formulation.get("research_direction") or f"深入探究 {pair['node_a']} 对 {pair['node_b']} 的直接分子作用机制及表型响应"

            cards.append({
                "id": f"hypo_{uuid.uuid4().hex[:8]}",
                "mode": pair.get("mode", resolved_mode),
                "motif_type": pair.get("motif_type", "detour_shortcut"),
                "motif_label": pair.get("motif_label", "⚡ 捷径短路模体"),
                "node_a": pair["node_a"],
                "category_a": pair["category_a"],
                "node_b": pair["node_b"],
                "category_b": pair["category_b"],
                "circuit_nodes": pair.get("circuit_nodes", []),
                "circuit_edges": pair.get("circuit_edges", []),
                "hypothesis_statement": stmt,
                "proposed_relation": rel,
                "biological_rationale": rationale,
                "research_direction": direction,
                "verification_status": "unverified",
                "query": "",
                "retrieved_articles": [],
                "adjudication": None
            })

        return cards

    def verify_single_hypothesis(self, hypothesis: Dict[str, Any], intent: str = "") -> Dict[str, Any]:
        """
        单卡片文献检索与裁判大模型审读流程
        """
        node_a = hypothesis.get("node_a", "")
        node_b = hypothesis.get("node_b", "")
        rel = hypothesis.get("proposed_relation", "")

        # 1. 生成高召回布尔检索式
        if self.agent and hasattr(self.agent, "generate_hypothesis_pubmed_query"):
            query = self.agent.generate_hypothesis_pubmed_query(node_a, node_b, proposed_relation=rel, intent=intent)
        else:
            clean_a = re.sub(r'[\(\)\[\]]', '', node_a).strip()
            clean_b = re.sub(r'[\(\)\[\]]', '', node_b).strip()
            query = f'("{clean_a}"[Title/Abstract] AND "{clean_b}"[Title/Abstract])'

        hypothesis["query"] = query

        # 2. 检索前沿文献摘要 (多级容灾降级检索 + 完整抓取真实摘要文本)
        clean_a = re.sub(r'[\(\)\[\]]', '', node_a).strip()
        clean_b = re.sub(r'[\(\)\[\]]', '', node_b).strip()
        articles = []
        try:
            # 策略一：尝试模型生成的精细检索式
            articles = self.searcher.search_articles(query, max_results=3)

            # 策略二降级：若精细检索式未召回文献，降级为核心实体 Title/Abstract 双词布尔约束
            if not articles:
                fallback_query_1 = f'("{clean_a}"[Title/Abstract] AND "{clean_b}"[Title/Abstract])'
                if fallback_query_1.lower() != query.lower():
                    articles = self.searcher.search_articles(fallback_query_1, max_results=3)
                    if articles:
                        hypothesis["query"] = fallback_query_1

            # 策略三降级：若仍未召回，降级为核心实体全局自由布尔检索
            if not articles:
                fallback_query_2 = f'("{clean_a}" AND "{clean_b}")'
                if fallback_query_2.lower() != query.lower() and fallback_query_2.lower() != fallback_query_1.lower():
                    articles = self.searcher.search_articles(fallback_query_2, max_results=3)
                    if articles:
                        hypothesis["query"] = fallback_query_2

            # 核心强化：确保每篇检索到的文献都具备真实摘要文本，供裁判大模型细读
            for art in articles:
                if not art.get("abstract"):
                    pmid = art.get("pmid", "")
                    if pmid and hasattr(self.searcher, "fetch_abstract"):
                        try:
                            fetched_ab = self.searcher.fetch_abstract(pmid)
                            if fetched_ab:
                                art["abstract"] = fetched_ab
                        except Exception as fe:
                            print(f"⚠️ [HypothesisEngine] 摘要获取失败 (PMID: {pmid}): {fe}")
        except Exception as e:
            print(f"⚠️ [HypothesisEngine] 文献检索调用异常: {e}")
            articles = []

        hypothesis["retrieved_articles"] = articles

        # 3. 呼叫文献终审专家大模型进行深度判决
        if self.agent and hasattr(self.agent, "adjudicate_literature_novelty"):
            adjudication = self.agent.adjudicate_literature_novelty(
                hypothesis_info=hypothesis,
                articles=articles,
                lang=self.lang
            )
        else:
            adjudication = {
                "verdict": "novel_gap" if not articles else "known",
                "confidence": 0.8,
                "direct_evidence_pmid": articles[0].get("pmid", "") if articles else "",
                "evidence_quote": "",
                "research_direction": hypothesis.get("research_direction", ""),
                "adjudication_rationale": "基于规则与检索文献匹配完成初步判定。"
            }

        hypothesis["adjudication"] = adjudication
        hypothesis["verification_status"] = "verified"
        return hypothesis

    def verify_hypotheses_concurrent(self, hypotheses: List[Dict[str, Any]],
                                      intent: str = "",
                                      max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        阶段二：多线程高并发文献裁判
        利用线程池对学者勾选的一个或多个假说并发执行检索与裁决，显著降低等待时长。
        """
        if not hypotheses:
            return []

        actual_workers = min(max_workers, len(hypotheses))
        results_map = {}

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_hypo = {
                executor.submit(self.verify_single_hypothesis, hypo, intent): hypo["id"]
                for hypo in hypotheses
            }

            for future in as_completed(future_to_hypo):
                hypo_id = future_to_hypo[future]
                try:
                    updated_hypo = future.result()
                    results_map[hypo_id] = updated_hypo
                except Exception as e:
                    print(f"⚠️ [HypothesisEngine] 并发裁判任务异常 ({hypo_id}): {e}")

        # 按原有顺序组装返回
        final_list = []
        for hypo in hypotheses:
            if hypo["id"] in results_map:
                final_list.append(results_map[hypo["id"]])
            else:
                hypo["verification_status"] = "error"
                final_list.append(hypo)

        return final_list

    @staticmethod
    def export_hypothesis_as_biokg(hypothesis: Dict[str, Any],
                                   master_entities: List[Dict[str, Any]],
                                   master_relations: List[Dict[str, Any]],
                                   project_id: str = "",
                                   project_name: str = "",
                                   workbench_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        将单项假说及其局部微环境打包为合法的 .biokg 独立工程字典：
        - 包含探勘假说两端核心实体（node_a, node_b）及其在原图中的 1 阶邻居实体；
        - 包含该局部微环境内部已有的全部关系；
        - 包含所推演出的假说关联（标记为科研探勘假说边）；
        - 附带文献裁判检索回的 PMID 列表作为初始文献依据；
        - 继承工作台已有本体分类数据，供学者导入主系统作为专题工程进行二次深度文献拓展。
        """
        import time

        node_a = hypothesis.get("node_a", "")
        node_b = hypothesis.get("node_b", "")
        clean_title = f"{node_a}_{node_b}"
        clean_title = re.sub(r'[\$\\/:\*\?"<>\|]', '_', clean_title)[:30]

        sub_pid = f"proj_hypo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        sub_pname = f"{project_name} - 专题探勘: {clean_title}" if project_name else f"专题探勘: {clean_title}"

        core_nodes = {node_a, node_b}
        for cn in hypothesis.get("circuit_nodes", []):
            cn_name = cn.get("name", "").strip() if isinstance(cn, dict) else str(cn).strip()
            if cn_name:
                core_nodes.add(cn_name)
        local_nodes = set(core_nodes)

        local_relations = []
        for rel in master_relations:
            s = rel.get("source", "").strip()
            t = rel.get("target", "").strip()
            if s in core_nodes or t in core_nodes:
                local_nodes.add(s)
                local_nodes.add(t)
                local_relations.append(rel)

        ent_lookup = {e.get("standard_name", "").strip(): e for e in master_entities if isinstance(e, dict)}
        sub_entities = []
        for n in sorted(list(local_nodes)):
            if not n:
                continue
            if n in ent_lookup:
                sub_entities.append(ent_lookup[n])
            else:
                sub_entities.append({
                    "standard_name": n,
                    "aliases": [],
                    "doc_source": "Hypothesis Subproject"
                })

        hypo_edge = {
            "source": node_a,
            "target": node_b,
            "relation": hypothesis.get("proposed_relation", "Hypothesized_Interaction"),
            "doc_source": "AI-Hypothesis (科研探勘推演)",
            "evidence": hypothesis.get("hypothesis_statement", "") + " | 建议方向: " + hypothesis.get("research_direction", ""),
            "weight": 1,
            "is_hypothesis": True
        }
        sub_relations = list(local_relations) + [hypo_edge]

        analyzed_files = []
        for art in hypothesis.get("retrieved_articles", []):
            pmid = art.get("pmid", "")
            if pmid:
                tag = f"PMID: {pmid}"
                if tag not in analyzed_files:
                    analyzed_files.append(tag)

        sub_wb = {}
        if workbench_state:
            old_ea = workbench_state.get("entity_annotations", {})
            sub_wb["entity_annotations"] = {n: old_ea[n] for n in local_nodes if n in old_ea}
            old_ra = workbench_state.get("relation_annotations", {})
            sub_wb["relation_annotations"] = old_ra
            sub_wb["active_preset"] = workbench_state.get("active_preset", "universal_biored")

        return {
            "project_id": sub_pid,
            "project_name": sub_pname,
            "entities": sub_entities,
            "relations": sub_relations,
            "analyzed_files": analyzed_files,
            "workbench_state": sub_wb,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "source_type": "hypothesis_subproject"
        }
