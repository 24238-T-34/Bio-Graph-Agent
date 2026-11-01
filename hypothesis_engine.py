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

        # 构建图论模型与拓扑属性
        self.G = nx.Graph()
        self._build_graph()

        # 确保阶段一本体数据可用（无缝 0-Token 保底）
        self.node_ontology = self._ensure_stage1_ontology()

    def _build_graph(self):
        """构建底层无向图以进行连通分量与拓扑中心度计算"""
        for ent in self.entities:
            name = ent.get("standard_name", "").strip()
            if name:
                self.G.add_node(name, **ent)

        for rel in self.relations:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            if src and tgt and src in self.G and tgt in self.G:
                w = rel.get("weight", 1)
                if self.G.has_edge(src, tgt):
                    self.G[src][tgt]["weight"] = self.G[src][tgt].get("weight", 1) + w
                else:
                    edge_attrs = dict(rel)
                    edge_attrs["weight"] = w
                    self.G.add_edge(src, tgt, **edge_attrs)

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
        拓扑自感知算法：
        自动评估网络连通性，智能推荐最契合的探勘策略。
        """
        if len(self.G) < 2:
            return {
                "mode": "peripheral_growth",
                "label": "周界边缘生长",
                "reason": "当前网络节点较少，推荐向外进行周界边缘薄层延展。"
            }

        n_comp = nx.number_connected_components(self.G)
        if n_comp >= 2:
            return {
                "mode": "cluster_bridge",
                "label": "跨簇孤岛桥接",
                "reason": f"检测到图谱存在 {n_comp} 个互不连通的知识孤岛，算法推荐优先寻找跨孤岛深层调控桥梁。"
            }
        else:
            return {
                "mode": "peripheral_growth",
                "label": "周界边缘生长",
                "reason": "图谱呈现全局强连通态，算法推荐围绕周界高价值节点探索合理的上下游薄层延展。"
            }

    def _get_node_category(self, node: str) -> str:
        return self.node_ontology.get(node, {}).get("category", "Gene_Protein")

    def _calc_compatibility_score(self, cat_a: str, cat_b: str) -> float:
        """
        本体语义相容性打分：
        根据生物学生化常识，为不同类型节点之间的互作潜力赋权。
        """
        ca, cb = cat_a.lower(), cat_b.lower()
        # 药物/化合物 与 靶点/受体/基因
        if ("drug" in ca or "chemical" in ca) and ("gene" in cb or "protein" in cb):
            return 3.5
        if ("drug" in cb or "chemical" in cb) and ("gene" in ca or "protein" in ca):
            return 3.5
        # 基因/蛋白 与 生物过程/通路/表型
        if ("gene" in ca or "protein" in ca) and ("process" in cb or "phenotype" in cb or "disease" in cb):
            return 3.0
        if ("gene" in cb or "protein" in cb) and ("process" in ca or "phenotype" in ca or "disease" in ca):
            return 3.0
        # 基因/蛋白 与 基因/蛋白 (同源/复合体/级联)
        if ("gene" in ca or "protein" in ca) and ("gene" in cb or "protein" in cb):
            return 2.5
        # 药物 与 疾病/表型 (适应症关联)
        if ("drug" in ca or "chemical" in ca) and ("disease" in cb or "phenotype" in cb):
            return 2.8
        if ("drug" in cb or "chemical" in cb) and ("disease" in ca or "phenotype" in ca):
            return 2.8
        return 1.5

    def _mine_cluster_bridges(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        模式一：跨簇孤岛桥接
        识别不同连通子图中的核心 Hub 与 Gateway 节点，构建高相容性跨岛桥梁候选对。
        """
        if len(self.G) < 2:
            return []

        components = sorted(list(nx.connected_components(self.G)), key=len, reverse=True)
        if len(components) < 2:
            # 若本身连通，平滑回落为寻找具有高介数中心度的分割瓶颈点对
            return self._mine_peripheral_growth(top_k=top_k)

        # 提取各主要簇的 Hub 候选
        cluster_hubs = []
        for comp in components[:4]:
            subgraph = self.G.subgraph(comp)
            degrees = dict(subgraph.degree())
            sorted_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)
            # 取各簇度数最高的前 2 名作为 Hub
            top_nodes = sorted_nodes[:2]
            cluster_hubs.append(top_nodes)

        candidate_pairs = []
        # 跨簇两两配对打分
        for i in range(len(cluster_hubs)):
            for j in range(i + 1, len(cluster_hubs)):
                for u in cluster_hubs[i]:
                    for v in cluster_hubs[j]:
                        if self.G.has_edge(u, v):
                            continue
                        cat_u = self._get_node_category(u)
                        cat_v = self._get_node_category(v)
                        compat = self._calc_compatibility_score(cat_u, cat_v)
                        deg_score = (self.G.degree(u) + self.G.degree(v)) * 0.5

                        total_score = compat * 2.0 + deg_score
                        candidate_pairs.append({
                            "node_a": u,
                            "category_a": cat_u,
                            "node_b": v,
                            "category_b": cat_v,
                            "score": total_score,
                            "rationale_hint": f"跨孤岛核心 Hub 对撞：{u}(度数 {self.G.degree(u)}) 与 {v}(度数 {self.G.degree(v)})"
                        })

        candidate_pairs.sort(key=lambda x: x["score"], reverse=True)
        return candidate_pairs[:top_k]

    def _mine_peripheral_growth(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        模式二：周界边缘薄层生长
        寻找网络周界（度数 1~2）但本体重要性高的功能实体，与其临近或核心中枢建立深层前沿偶联。
        """
        if len(self.G) < 2:
            return []

        degrees = dict(self.G.degree())
        all_nodes = list(self.G.nodes())

        # 区分核心中枢 Hub 与 周界叶子节点 Leaf
        hubs = sorted([n for n in all_nodes if degrees[n] >= 3], key=lambda n: degrees[n], reverse=True)
        if not hubs:
            hubs = sorted(all_nodes, key=lambda n: degrees[n], reverse=True)[:3]

        leaf_candidates = [n for n in all_nodes if degrees[n] <= 2]
        if not leaf_candidates:
            leaf_candidates = sorted(all_nodes, key=lambda n: degrees[n])[:5]

        candidate_pairs = []
        for leaf in leaf_candidates:
            cat_leaf = self._get_node_category(leaf)
            for hub in hubs:
                if leaf == hub or self.G.has_edge(leaf, hub):
                    continue

                cat_hub = self._get_node_category(hub)
                compat = self._calc_compatibility_score(cat_leaf, cat_hub)

                # 计算拓扑最短路径距离 (若已连通)
                try:
                    path_len = nx.shortest_path_length(self.G, source=leaf, target=hub)
                except Exception:
                    path_len = 5

                # 距离大于 2 步且本体高度相容者，具有最高探勘价值
                score = compat * 2.5 + path_len * 1.2
                candidate_pairs.append({
                    "node_a": leaf,
                    "category_a": cat_leaf,
                    "node_b": hub,
                    "category_b": cat_hub,
                    "score": score,
                    "rationale_hint": f"周界边缘实体 {leaf} 与主干中枢 {hub} (当前拓扑距离: {path_len} 步)"
                })

        candidate_pairs.sort(key=lambda x: x["score"], reverse=True)
        return candidate_pairs[:top_k]

    def _mine_intent_exploration(self, intent: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        模式三：意向定向探究
        基于学者设定的研究焦点词，在图中锁定与该意图高度契合但拓扑尚未直连的节点对。
        """
        if len(self.G) < 2 or not intent:
            return self._mine_peripheral_growth(top_k=top_k)

        intent_lower = intent.lower()
        all_nodes = list(self.G.nodes())

        # 节点与意图匹配度评分
        node_intent_scores = {}
        for node in all_nodes:
            score = 0.0
            n_low = node.lower()
            if n_low in intent_lower or intent_lower in n_low:
                score += 5.0

            aliases = self.G.nodes[node].get("aliases", [])
            for al in aliases:
                if al.lower() in intent_lower or intent_lower in al.lower():
                    score += 4.0

            cat = self._get_node_category(node)
            if "drug" in cat.lower() and any(w in intent_lower for w in ["药", "drug", "inhibitor", "抑制", "化合物"]):
                score += 2.0
            if "disease" in cat.lower() and any(w in intent_lower for w in ["病", "cancer", "瘤", "症", "disease"]):
                score += 2.0
            if "process" in cat.lower() and any(w in intent_lower for w in ["通路", "pathway", "凋亡", "自噬", "apoptosis"]):
                score += 2.0

            node_intent_scores[node] = score

        # 选出与意图最相关的 Top 节点池
        relevant_pool = sorted(all_nodes, key=lambda n: node_intent_scores[n], reverse=True)[:8]
        if len(relevant_pool) < 2:
            relevant_pool = sorted(all_nodes, key=lambda n: self.G.degree(n), reverse=True)[:6]

        candidate_pairs = []
        for i in range(len(relevant_pool)):
            for j in range(i + 1, len(relevant_pool)):
                u, v = relevant_pool[i], relevant_pool[j]
                if self.G.has_edge(u, v):
                    continue

                cat_u = self._get_node_category(u)
                cat_v = self._get_node_category(v)
                compat = self._calc_compatibility_score(cat_u, cat_v)
                i_score = node_intent_scores.get(u, 0) + node_intent_scores.get(v, 0)

                total_score = i_score * 2.0 + compat * 1.5
                candidate_pairs.append({
                    "node_a": u,
                    "category_a": cat_u,
                    "node_b": v,
                    "category_b": cat_v,
                    "score": total_score,
                    "rationale_hint": f"围绕科研意向 '{intent[:15]}' 聚焦的潜在断裂带"
                })

        candidate_pairs.sort(key=lambda x: x["score"], reverse=True)
        return candidate_pairs[:top_k]

    def mine_and_formulate(self, mode: str = "auto", intent: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        阶段一：图论寻孔 + 假说生成
        一次性推演出 ~5 项具有高学术价值的候选假说卡片。
        """
        if len(self.G) < 2:
            return []

        resolved_mode = mode
        if mode == "auto":
            auto_info = self.detect_auto_mode()
            resolved_mode = auto_info["mode"]

        # 1. 结构洞拓扑初筛
        if resolved_mode == "cluster_bridge":
            candidate_pairs = self._mine_cluster_bridges(top_k=top_k)
        elif resolved_mode == "peripheral_growth":
            candidate_pairs = self._mine_peripheral_growth(top_k=top_k)
        elif resolved_mode == "intent_exploration":
            candidate_pairs = self._mine_intent_exploration(intent=intent, top_k=top_k)
        else:
            candidate_pairs = self._mine_cluster_bridges(top_k=top_k)

        if not candidate_pairs:
            return []

        # 2. 调用大模型中枢为每对生成专业叙事与明确科研方向
        if self.agent and hasattr(self.agent, "formulate_hypothesis_candidates"):
            llm_formulated = self.agent.formulate_hypothesis_candidates(
                candidate_pairs=candidate_pairs,
                intent=intent,
                lang=self.lang
            )
        else:
            llm_formulated = []

        # 3. 封装为完整的假说卡片结构
        cards = []
        for i, pair in enumerate(candidate_pairs):
            formulation = llm_formulated[i] if i < len(llm_formulated) else {}
            stmt = formulation.get("hypothesis_statement") or f"{pair['node_a']} 与 {pair['node_b']} 存在直接或级联调控互作"
            rel = formulation.get("proposed_relation") or "Positive_Regulation"
            rationale = formulation.get("biological_rationale") or pair.get("rationale_hint", "")
            direction = formulation.get("research_direction") or f"深入探究 {pair['node_a']} 对 {pair['node_b']} 的直接分子作用机制"

            cards.append({
                "id": f"hypo_{uuid.uuid4().hex[:8]}",
                "mode": resolved_mode,
                "node_a": pair["node_a"],
                "category_a": pair["category_a"],
                "node_b": pair["node_b"],
                "category_b": pair["category_b"],
                "hypothesis_statement": stmt,
                "proposed_relation": rel,
                "biological_rationale": rationale,
                "research_direction": direction,
                "verification_status": "unverified",  # unverified | verified | error
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

        # 2. 检索前沿文献摘要 (Top 3 篇即可支撑严谨裁判)
        try:
            articles = self.searcher.search_articles(query, max_results=3)
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
