"""
workbench_engine.py - 图谱分析工作台核心引擎
包含：
1. RulePreFilter: 基于生物医学命名法与权威词缀的高速 0-Token 正则预过滤分流器（涵盖分子、细胞、微生物、器官、物种多尺度）
2. OntologyManager: 本体分类预设配置加载、管理与保存系统
3. WorkbenchAnalysisEngine: 分级升级状态机（Round 0-3）、多线程并发推理、热度优先级调度、微上下文组装、精准文献回溯、通用生物知识兜底及人工专家覆写
"""

import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Callable
import networkx as nx


class RulePreFilter:
    """
    基于本体标准配置文件 (ontology_presets/*.json) 中实体分类内定义的三要素（名称、描述、正则表达式）
    进行数据驱动的高速确定性分类，无任何写死在代码中的规则。
    """

    _compiled_pattern_cache: Dict[str, Optional[re.Pattern]] = {}

    @classmethod
    def _get_compiled_regex(cls, pattern_str: str) -> Optional[re.Pattern]:
        if pattern_str not in cls._compiled_pattern_cache:
            try:
                cls._compiled_pattern_cache[pattern_str] = re.compile(pattern_str, re.IGNORECASE)
            except Exception as e:
                print(f"⚠️ [RulePreFilter] 正则编译失败: {pattern_str} ({e})")
                cls._compiled_pattern_cache[pattern_str] = None
        return cls._compiled_pattern_cache[pattern_str]

    @classmethod
    def _compile_regex(cls, pattern_str: str) -> Optional[re.Pattern]:
        return cls._get_compiled_regex(pattern_str)

    @classmethod
    def classify(cls, node_name: str, aliases: Optional[List[str]] = None,
                 ontology: Optional[Dict[str, Any]] = None, language: str = "zh") -> Optional[Dict[str, Any]]:
        """
        根据当前本体预设中各实体分类显式声明的 patterns 正则表达式列表进行匹配。
        - 仅对 entity_categories 中包含 patterns 且非 Uncertain 的类别进行匹配。
        - 遇到命中时，统一标注理由为极简字符串（"规则正则匹配" / "Rule-based regex match"）。
        - 若未命中，安全返回 None，平滑交由后续大模型阶段处理。
        """
        if not ontology or not isinstance(ontology, dict):
            return None

        ent_cats = ontology.get("entity_categories", {})
        if not ent_cats or not isinstance(ent_cats, dict):
            return None

        raw_name = node_name.strip()
        lower_name = raw_name.lower()
        all_candidates = [lower_name]
        if aliases:
            all_candidates.extend([a.strip().lower() for a in aliases if a.strip()])

        is_en = "en" in str(language).lower()
        unified_reason = "Rule-based regex match" if is_en else "规则正则匹配"

        for cat_id, cat_info in ent_cats.items():
            if cat_id == "Uncertain" or not isinstance(cat_info, dict):
                continue

            patterns = cat_info.get("patterns", [])
            if not patterns or not isinstance(patterns, list):
                continue

            cat_name = (cat_info.get("en_name") or cat_info.get("en_label", cat_id)) if is_en else (cat_info.get("name") or cat_info.get("label", cat_id))

            for pat_str in patterns:
                compiled = cls._get_compiled_regex(pat_str)
                if not compiled:
                    continue
                for cand in all_candidates:
                    if compiled.search(cand):
                        return {
                            "primary_type": cat_id,
                            "sub_type": "",
                            "tags": [cat_name],
                            "confidence": 0.95,
                            "reasoning": unified_reason,
                            "round": 0,
                            "status": "resolved"
                        }

        return None


class OntologyManager:
    """本体预设与分类方案管理系统"""

    def __init__(self, presets_dir: Optional[str] = None):
        if presets_dir is None:
            self.presets_dir = os.path.join(os.path.dirname(__file__), "ontology_presets")
        else:
            self.presets_dir = presets_dir

        if not os.path.exists(self.presets_dir):
            os.makedirs(self.presets_dir, exist_ok=True)

    def list_presets(self) -> List[Dict[str, Any]]:
        """扫描本地 ontology_presets 目录并返回所有合法预设"""
        presets = []
        if not os.path.exists(self.presets_dir):
            return presets

        for fname in sorted(os.listdir(self.presets_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(self.presets_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "entity_categories" in data:
                        if "id" not in data:
                            data["id"] = fname.replace(".json", "")
                        presets.append(data)
                except Exception as e:
                    print(f"⚠️ [OntologyManager] 读取预设文件 {fname} 失败: {e}")
        return presets

    def get_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        """根据 preset_id 获取预设配置"""
        for p in self.list_presets():
            if p.get("id") == preset_id:
                return p
        fpath = os.path.join(self.presets_dir, f"{preset_id}.json")
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def save_preset(self, preset_dict: Dict[str, Any]) -> str:
        """保存自定义本体预设到 JSON 文件"""
        preset_id = preset_dict.get("id", f"custom_{int(time.time())}")
        fpath = os.path.join(self.presets_dir, f"{preset_id}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(preset_dict, f, ensure_ascii=False, indent=2)
        return fpath


class WorkbenchAnalysisEngine:
    """图谱分析工作台全流程分析与并发执行引擎"""

    def __init__(self,
                 llm_agent=None,
                 ontology: Optional[Dict[str, Any]] = None,
                 paper_full_texts: Optional[Dict[str, str]] = None,
                 cached_entities: Optional[Dict[str, Any]] = None,
                 cached_relations: Optional[Dict[str, Any]] = None):
        self.llm_agent = llm_agent
        self.ontology = ontology or {}
        self.paper_full_texts = paper_full_texts or {}
        self.entity_annotations: Dict[str, Dict[str, Any]] = cached_entities or {}
        self.relation_annotations: Dict[str, Dict[str, Any]] = cached_relations or {}
        self.stats = {
            "total_nodes": 0,
            "resolved_nodes": 0,
            "uncertain_nodes": 0,
            "round0_hits": 0,
            "round1_hits": 0,
            "round2_hits": 0,
            "round3_hits": 0,
            "total_relations": 0,
            "analyzed_relations": 0
        }

    @staticmethod
    def compute_node_degrees(relations: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算节点中心度 (热度权重与连接数)，用于降序排定分析优先级"""
        degrees = Counter()
        for r in relations:
            src = r.get("source")
            tgt = r.get("target")
            w = r.get("weight", 1)
            if src:
                degrees[src] += w
            if tgt:
                degrees[tgt] += w
        return dict(degrees)

    def _extract_micro_context(self, node_name: str, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """为单节点提取紧凑局部的拓扑微上下文与文献证据"""
        aliases = []
        doc_source = "未知"
        for ent in entities:
            if ent.get("standard_name") == node_name:
                aliases = ent.get("aliases", [])
                doc_source = ent.get("doc_source", "未知")
                break

        connected_triplets = []
        evidence_snippets = []
        for r in relations:
            src = r.get("source")
            tgt = r.get("target")
            rel = r.get("relation", "相关")
            evi = r.get("evidence", "")
            if src == node_name:
                connected_triplets.append(f"{node_name} --[{rel}]--> {tgt}")
                if evi and evi not in evidence_snippets:
                    evidence_snippets.append(evi)
            elif tgt == node_name:
                connected_triplets.append(f"{src} --[{rel}]--> {node_name}")
                if evi and evi not in evidence_snippets:
                    evidence_snippets.append(evi)

        return {
            "node_name": node_name,
            "aliases": aliases,
            "doc_source": doc_source,
            "neighbors": connected_triplets[:8],
            "evidence": evidence_snippets[:3]
        }

    def _fetch_paper_sentences(self, node_name: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Round 2: 根据正则精准在原文献全文中截取命中关键句 (避免通读全文)"""
        matched_sentences = []
        if not self.paper_full_texts:
            return matched_sentences

        target_doc = None
        aliases = []
        for ent in entities:
            if ent.get("standard_name") == node_name:
                target_doc = ent.get("doc_source", "")
                aliases = ent.get("aliases", [])
                break

        search_texts = []
        if target_doc:
            clean_doc = re.sub(r'\[.*?\]', '', target_doc).strip()
            for k, text in self.paper_full_texts.items():
                if clean_doc and (clean_doc in k or k in clean_doc):
                    search_texts.append(text)
        if not search_texts:
            search_texts = list(self.paper_full_texts.values())

        search_terms = [node_name] + [a for a in aliases if len(a) >= 3][:3]

        for text in search_texts:
            for term in search_terms:
                try:
                    pat = re.compile(rf'([^.\n]{{10,250}}?\b{re.escape(term)}\b[^.\n]{{10,250}}?\.)', re.IGNORECASE)
                    for match in pat.finditer(text):
                        sentence = match.group(1).strip().replace("\n", " ")
                        if sentence not in matched_sentences:
                            matched_sentences.append(sentence)
                        if len(matched_sentences) >= 3:
                            return matched_sentences
                except Exception:
                    continue

        return matched_sentences

    def _get_allowed_entity_ids(self) -> List[str]:
        """获取当前生效本体预设的合法实体分类 ID 列表（包含 Uncertain）"""
        cats = list(self.ontology.get("entity_categories", {}).keys())
        if "Uncertain" not in cats:
            cats.append("Uncertain")
        return cats

    def _get_allowed_relation_ids(self) -> List[str]:
        """获取当前生效本体预设的合法关系机制分类 ID 列表"""
        return list(self.ontology.get("relation_categories", {}).keys())

    def canonicalize_entity_category(self, raw_cat: str) -> Tuple[str, bool]:
        """
        智能规范化实体类别：
        1. 清洗反引号、引号、换行符、首尾空白与括号备注
        2. 精确匹配 valid_cats 的 key
        3. 忽略大小写比对 key
        4. 容错比对各分类声明的中英文 label / name / en_label / en_name 全称
        5. 分词语义重合比对（例如将 '基因/蛋白/RNA'、'细菌'、'Bacteria' 智能对齐至标准分类）
        6. 若匹配到合法且非 Uncertain 的类别，返回 (valid_key, True)
        7. 若无法对齐或为 Uncertain，统一返回 ('Uncertain', False)
        """
        if not raw_cat:
            return "Uncertain", False
        clean = str(raw_cat).replace("`", "").replace("'", "").replace('"', '').strip()
        # 消除类似 "Disease_Phenotype (疾病/病理表型)" 的附加备注
        clean = re.sub(r'[\(（].*?[\)）]', '', clean).strip()

        valid_cats = self.ontology.get("entity_categories", {})
        if not valid_cats:
            return "Uncertain", False

        # 1. 精确匹配
        if clean in valid_cats:
            if clean == "Uncertain":
                return "Uncertain", False
            return clean, True

        def _norm(s: str) -> str:
            return re.sub(r'[\s/／\-_]+', '', str(s).lower())

        clean_norm = _norm(clean)
        if not clean_norm:
            return "Uncertain", False

        # 2. 大小写与连接符无关匹配
        for k in valid_cats.keys():
            if _norm(k) == clean_norm:
                if k == "Uncertain":
                    return "Uncertain", False
                return k, True

        # 3. 标签与多语言全称完整匹配
        for k, v in valid_cats.items():
            if not isinstance(v, dict):
                continue
            candidates = [
                v.get("label", ""),
                v.get("name", ""),
                v.get("en_label", ""),
                v.get("en_name", "")
            ]
            for cand in candidates:
                if cand and _norm(cand) == clean_norm:
                    if k == "Uncertain":
                        return "Uncertain", False
                    return k, True

        # 4. 关键分词重合度匹配 (容错复合标签与子词)
        in_tokens = set([p.strip().lower() for p in re.split(r'[\s/／,，、;；|_\-]+', clean) if len(p.strip()) >= 2])
        if in_tokens:
            for k, v in valid_cats.items():
                if not isinstance(v, dict):
                    continue
                combined_desc = " ".join([k, v.get("label", ""), v.get("en_label", ""), v.get("name", ""), v.get("en_name", "")])
                cat_tokens = set([p.strip().lower() for p in re.split(r'[\s/／,，、;；|_\-]+', combined_desc) if len(p.strip()) >= 2])
                overlap = in_tokens.intersection(cat_tokens)
                if len(overlap) >= 2 or (len(overlap) == 1 and len(in_tokens) == 1 and list(overlap)[0] not in ["cell", "gene", "protein", "disease", "factor", "pathway"]):
                    if k == "Uncertain":
                        return "Uncertain", False
                    return k, True

        # 5. 框架外自创新词（如不在当前预设中的词汇）一律拦截为未分类
        return "Uncertain", False

    def canonicalize_relation_category(self, raw_cat: str) -> str:
        """
        智能规范化关系机制类别：
        1. 清洗反引号、引号、换行与空白
        2. 精确与大小写匹配当前生效机制类别
        3. 标签容错匹配
        4. 若不在框架内，安全降级为当前预设的首选默认机制（如 Association_Correlation）
        """
        valid_rels = self.ontology.get("relation_categories", {})
        if not valid_rels:
            return "Association_Correlation"

        default_rel = "Association_Correlation" if "Association_Correlation" in valid_rels else list(valid_rels.keys())[0]

        if not raw_cat:
            return default_rel

        clean = str(raw_cat).replace("`", "").replace("'", "").replace('"', '').strip()
        clean = re.sub(r'[\(（].*?[\)）]', '', clean).strip()

        if clean in valid_rels:
            return clean

        clean_lower = clean.lower()
        for k in valid_rels.keys():
            if k.lower() == clean_lower:
                return k

        for k, v in valid_rels.items():
            if not isinstance(v, dict):
                continue
            candidates = [
                v.get("label", ""),
                v.get("name", ""),
                v.get("en_label", ""),
                v.get("en_name", "")
            ]
            for cand in candidates:
                if cand and (cand.strip().lower() == clean_lower or cand.strip() == clean):
                    return k

        return default_rel

    def _format_ontology_categories(self, language: str = "zh") -> str:
        """格式化实体候选类别清单供 Prompt 注入 (支持中英双语候选描述适配)"""
        cats = self.ontology.get("entity_categories", {})
        lines = []
        is_en = "en" in str(language).lower()
        for k, v in cats.items():
            lbl = v.get("en_label", v.get("label", k)) if is_en else v.get("label", k)
            desc = v.get("en_description", v.get("description", "")) if is_en else v.get("description", "")
            lines.append(f"- `{k}` ({lbl}): {desc}")
        return "\n".join(lines)

    def _format_relation_categories(self, language: str = "zh") -> str:
        """格式化关系机制候选类别清单供 Prompt 注入 (支持中英双语候选描述适配)"""
        cats = self.ontology.get("relation_categories", {})
        lines = []
        is_en = "en" in str(language).lower()
        for k, v in cats.items():
            lbl = v.get("en_label", v.get("label", k)) if is_en else v.get("label", k)
            desc = v.get("en_description", v.get("description", "")) if is_en else v.get("description", "")
            if desc:
                lines.append(f"- `{k}` ({lbl}): {desc}")
            else:
                lines.append(f"- `{k}` ({lbl})")
        return "\n".join(lines)

    def _parse_llm_json(self, raw_str: str) -> Optional[Dict[str, Any]]:
        """清洗并严格反序列化大模型返回的 JSON"""
        if not raw_str:
            return None
        cleaned = raw_str.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def _llm_classify_entity_round1(self, node_name: str, micro_ctx: Dict[str, Any], language: str = "zh") -> Dict[str, Any]:
        """Round 1: 依据局部网络微上下文分类单节点 (委托 BioBrainAgent 统一处理，保留安全降级)"""
        is_en = "en" in str(language).lower()
        cat_str = self._format_ontology_categories(language=language)
        allowed_ids = self._get_allowed_entity_ids()

        if self.llm_agent:
            try:
                if hasattr(self.llm_agent, "classify_workbench_entity_round1"):
                    res = self.llm_agent.classify_workbench_entity_round1(
                        node_name=node_name,
                        micro_ctx=micro_ctx,
                        cat_str=cat_str,
                        allowed_ids=allowed_ids,
                        language=language
                    )
                    if isinstance(res, dict) and "primary_type" in res:
                        return res
            except Exception as e:
                print(f"⚠️ [Round 1 Entity Delegation Error] {node_name}: {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "LLM call error or invalid response" if is_en else "大模型调用异常或未能返回合法结构"
        }

    def _llm_classify_entity_round2(self, node_name: str, matched_sentences: List[str], language: str = "zh") -> Dict[str, Any]:
        """Round 2: 根据正则精准回溯原文语句进行判定 (委托 BioBrainAgent 统一处理，保留安全降级)"""
        is_en = "en" in str(language).lower()
        cat_str = self._format_ontology_categories(language=language)
        allowed_ids = self._get_allowed_entity_ids()

        if self.llm_agent:
            try:
                if hasattr(self.llm_agent, "classify_workbench_entity_round2"):
                    res = self.llm_agent.classify_workbench_entity_round2(
                        node_name=node_name,
                        matched_sentences=matched_sentences,
                        cat_str=cat_str,
                        allowed_ids=allowed_ids,
                        language=language
                    )
                    if isinstance(res, dict) and "primary_type" in res:
                        return res
            except Exception as e:
                print(f"⚠️ [Round 2 Entity Delegation Error] {node_name}: {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "Context in paper is insufficient to confirm category" if is_en else "文献上下文证据不足以明确分类"
        }

    def _llm_classify_entity_round3(self, node_name: str, language: str = "zh") -> Dict[str, Any]:
        """Round 3: 兜底通用生物知识判定 (委托 BioBrainAgent 统一处理，保留安全降级)"""
        is_en = "en" in str(language).lower()
        cat_str = self._format_ontology_categories(language=language)
        allowed_ids = self._get_allowed_entity_ids()

        if self.llm_agent:
            try:
                if hasattr(self.llm_agent, "classify_workbench_entity_round3"):
                    res = self.llm_agent.classify_workbench_entity_round3(
                        node_name=node_name,
                        cat_str=cat_str,
                        allowed_ids=allowed_ids,
                        language=language
                    )
                    if isinstance(res, dict) and "primary_type" in res:
                        return res
            except Exception as e:
                print(f"⚠️ [Round 3 Entity Delegation Error] {node_name}: {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "Features remain insufficient to classify after multi-stage analysis" if is_en else "经多级综合研判，实体特征仍不足以明确分类"
        }

    def _llm_classify_relation(self, source: str, target: str, orig_rel: str, evidence: str, language: str = "zh") -> Dict[str, Any]:
        """对单条关系进行深层机制分类与机制动词归纳 (委托 BioBrainAgent 统一处理，保留安全降级)"""
        is_en = "en" in str(language).lower()
        rel_cat_str = self._format_relation_categories(language=language)
        allowed_rel_ids = self._get_allowed_relation_ids()

        src_type = self.entity_annotations.get(source, {}).get("primary_type", "Unknown" if is_en else "未知")
        tgt_type = self.entity_annotations.get(target, {}).get("primary_type", "Unknown" if is_en else "未知")

        if self.llm_agent:
            try:
                if hasattr(self.llm_agent, "classify_workbench_relation"):
                    res = self.llm_agent.classify_workbench_relation(
                        source=source,
                        target=target,
                        orig_rel=orig_rel,
                        evidence=evidence,
                        rel_cat_str=rel_cat_str,
                        allowed_rel_ids=allowed_rel_ids,
                        src_type=src_type,
                        tgt_type=tgt_type,
                        language=language
                    )
                    if isinstance(res, dict) and "mechanism_category" in res:
                        return res
            except Exception as e:
                print(f"⚠️ [Relation Delegation Error] {source} -> {target}: {e}")

        default_cat = "Association_Correlation" if "Association_Correlation" in allowed_rel_ids else (allowed_rel_ids[0] if allowed_rel_ids else "Association_Correlation")
        return {
            "mechanism_category": default_cat,
            "mechanism_verb": orig_rel,
            "confidence": 0.5,
            "reasoning": "Analysis defaulted to general correlation" if is_en else "分析降级至默认关联"
        }

    def _heal_inconsistent_cache(self) -> Tuple[int, int]:
        """
        自动体检并愈合缓存中与当前生效本体预设不一致的脏数据：
        1. 扫描 self.entity_annotations：若其 primary_type 无法在当前预设中识别，
           强制重置为 primary_type: 'Uncertain', status: 'uncertain'，解除 resolved 锁死并排入分析队列；
        2. 扫描 self.relation_annotations：若其 mechanism_category 不属于当前预设，
           将其移出已解析缓存，强制重新分析；
        返回: (healed_entities_count, healed_relations_count)
        """
        healed_ents = 0
        for n, ann in list(self.entity_annotations.items()):
            cur_type = ann.get("primary_type", "")
            canon_type, is_valid = self.canonicalize_entity_category(cur_type)
            if not is_valid or canon_type == "Uncertain":
                if ann.get("status") == "resolved" or (cur_type and cur_type != "Uncertain"):
                    self.entity_annotations[n]["primary_type"] = "Uncertain"
                    self.entity_annotations[n]["status"] = "uncertain"
                    self.entity_annotations[n]["round"] = 0
                    healed_ents += 1

        healed_rels = 0
        valid_rels = set(self.ontology.get("relation_categories", {}).keys())
        for ek, r_ann in list(self.relation_annotations.items()):
            cur_mech = r_ann.get("mechanism_category", "")
            if valid_rels and cur_mech not in valid_rels:
                del self.relation_annotations[ek]
                healed_rels += 1

        return healed_ents, healed_rels

    def run_analysis(self,
                     entities: List[Dict[str, Any]],
                     relations: List[Dict[str, Any]],
                     enable_entities: bool = True,
                     enable_relations: bool = True,
                     enable_round0_rules: bool = True,
                     enable_round1_context: bool = True,
                     enable_round2_regex: bool = True,
                     enable_round3_world: bool = True,
                     force_reanalyze: bool = False,
                     max_workers: int = 4,
                     language: str = "zh",
                     progress_callback: Optional[Callable[[float, str], None]] = None,
                     log_callback: Optional[Callable[[str], None]] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        执行完整的工作台分级分析工作流（支持 ThreadPoolExecutor 多线程高并发加速与中英双语运行回显）
        """
        is_en = "en" in str(language).lower()

        def log(msg: str):
            if log_callback:
                log_callback(msg)
            print(f"[WorkbenchEngine] {msg}")

        workers_cnt = max(1, min(max_workers, 16))
        log(f"🚀 Initializing Graph Analysis Workbench Pipeline ({workers_cnt} workers)..." if is_en else f"🚀 启动图谱分析工作台流水线 (并发线程数: {workers_cnt})...")

        # 1. 计算热度与优先级
        degrees = self.compute_node_degrees(relations)
        all_node_names = [e.get("standard_name") for e in entities if e.get("standard_name")]
        sorted_nodes = sorted(all_node_names, key=lambda n: degrees.get(n, 0), reverse=True)

        total_nodes = len(sorted_nodes)
        total_rels = len(relations)
        self.stats["total_nodes"] = total_nodes
        self.stats["total_relations"] = total_rels

        # ----------------------------------------------------
        # 阶段 A: 实体分级升级状态机 (Round 0 -> 1 -> 2 -> 3)
        # ----------------------------------------------------
        if enable_entities and total_nodes > 0:
            log(f"📊 Entity queue ready ({total_nodes} entities), sorted by degree centrality." if is_en else f"📊 实体分析队列已就绪，共计 {total_nodes} 个实体，已按拓扑中心度降序排列。")

            # === 缓存一致性自动体检与愈合 (Auto-Healing Cache Inconsistencies) ===
            healed_ents, healed_rels = self._heal_inconsistent_cache()
            if healed_ents > 0 or healed_rels > 0:
                log(f"🧹 [Cache Healing] Reset {healed_ents} out-of-ontology entities and {healed_rels} relations for active preset." if is_en else f"🧹 [缓存一致性自动体检] 发现并重置了 {healed_ents} 个非当前预设实体与 {healed_rels} 条非规范机制连线，已将其纳入重审队列。")

            # 确保全图所有候选实体在分析启动前均具备有效状态（未解析者赋予 uncertain）
            for n in sorted_nodes:
                if force_reanalyze or n not in self.entity_annotations:
                    self.entity_annotations[n] = {
                        "primary_type": "Uncertain",
                        "sub_type": "",
                        "tags": [],
                        "confidence": 0.0,
                        "reasoning": "待进一步分级判断" if not is_en else "Pending analysis",
                        "round": 0,
                        "status": "uncertain"
                    }

            # === Round 0: 快速规则匹配 (0 Token, 极速预过滤) ===
            r0_hits = 0
            if enable_round0_rules:
                log("⚡ [Stage 1: Nomenclature Recognition] Screening biomedical nomenclature & affixes..." if is_en else "⚡ [阶段一: 命名规范识别] 正在依据生物医学命名规范与权威词缀高速解析...")
                for idx, node in enumerate(sorted_nodes):
                    if not force_reanalyze and node in self.entity_annotations:
                        cur = self.entity_annotations[node]
                        c_type, is_v = self.canonicalize_entity_category(cur.get("primary_type", ""))
                        if cur.get("status") == "resolved" and is_v and c_type != "Uncertain":
                            continue

                    aliases = []
                    for ent in entities:
                        if ent.get("standard_name") == node:
                            aliases = ent.get("aliases", [])
                            break

                    rule_res = RulePreFilter.classify(node, aliases, self.ontology, language=language)
                    if rule_res:
                        self.entity_annotations[node] = rule_res
                        r0_hits += 1
                    else:
                        if node not in self.entity_annotations or force_reanalyze:
                            self.entity_annotations[node] = {
                                "primary_type": "Uncertain",
                                "sub_type": "",
                                "tags": [],
                                "confidence": 0.0,
                                "reasoning": "Pending further escalation analysis" if is_en else "待进一步分级判断",
                                "round": 0,
                                "status": "uncertain"
                            }
                self.stats["round0_hits"] = r0_hits
                log(f"✅ [Stage 1 Complete] Successfully identified {r0_hits} entities via nomenclature rules." if is_en else f"✅ [阶段一完成] 命名规范与词缀规则成功解析 {r0_hits} 个实体。")

            # === Round 1: 微上下文分析 (ThreadPoolExecutor 并发) ===
            if enable_round1_context and self.llm_agent:
                r1_candidates = [n for n in sorted_nodes if self.entity_annotations.get(n, {}).get("status") == "uncertain"]
                total_r1 = len(r1_candidates)
                log(f"🧠 [Stage 2: Topology Reasoning] {total_r1} candidates queued for network topological analysis ({workers_cnt} workers)..." if is_en else f"🧠 [阶段二: 局部拓扑推理] 共有 {total_r1} 个待决实体进入网络拓扑并发分析 (并发数: {workers_cnt})...")

                r1_lock = threading.Lock()
                r1_done = 0

                def task_r1(n_name: str):
                    ctx = self._extract_micro_context(n_name, entities, relations)
                    res = self._llm_classify_entity_round1(n_name, ctx, language=language)
                    return n_name, res

                with ThreadPoolExecutor(max_workers=workers_cnt) as executor:
                    futures = {executor.submit(task_r1, n): n for n in r1_candidates}
                    for fut in as_completed(futures):
                        node, res = fut.result()
                        with r1_lock:
                            r1_done += 1
                            if progress_callback:
                                progress_callback(
                                    0.1 + 0.3 * (r1_done / max(1, total_r1)),
                                    f"Topology Reasoning ({r1_done}/{total_r1}): {node}" if is_en else f"局部拓扑推理 ({r1_done}/{total_r1}): {node}"
                                )
                            cat = res.get("primary_type")
                            canon_cat, is_valid = self.canonicalize_entity_category(cat)
                            if is_valid and canon_cat != "Uncertain":
                                self.entity_annotations[node] = {
                                    **res,
                                    "primary_type": canon_cat,
                                    "round": 1,
                                    "status": "resolved"
                                }
                                self.stats["round1_hits"] += 1
                            else:
                                # 不在框架内的类一律识别为未分类，赋予 status: "uncertain" 以流入 Round 2 / Round 3 兜底
                                self.entity_annotations[node] = {
                                    **res,
                                    "primary_type": "Uncertain",
                                    "round": 1,
                                    "status": "uncertain"
                                }

                log(f"✅ [Stage 2 Complete] Resolved {self.stats['round1_hits']} entities." if is_en else f"✅ [阶段二完成] 局部拓扑推理成功解析 {self.stats['round1_hits']} 个实体。")

            # === Round 2: 文献正则精准回溯 (ThreadPoolExecutor 并发) ===
            if enable_round2_regex and self.llm_agent and self.paper_full_texts:
                r2_candidates = [n for n in sorted_nodes if self.entity_annotations.get(n, {}).get("status") == "uncertain"]
                total_r2 = len(r2_candidates)
                log(f"📖 [Stage 3: Literature Grounding] {total_r2} unresolved candidates queued for full-text grounding..." if is_en else f"📖 [阶段三: 文献全文溯源] 共有 {total_r2} 个待决实体进入文献原文深度检索...")

                r2_lock = threading.Lock()
                r2_done = 0

                def task_r2(n_name: str):
                    matched = self._fetch_paper_sentences(n_name, entities)
                    if matched:
                        res = self._llm_classify_entity_round2(n_name, matched, language=language)
                        return n_name, res, True
                    return n_name, None, False

                with ThreadPoolExecutor(max_workers=workers_cnt) as executor:
                    futures = {executor.submit(task_r2, n): n for n in r2_candidates}
                    for fut in as_completed(futures):
                        node, res, had_sents = fut.result()
                        with r2_lock:
                            r2_done += 1
                            if progress_callback:
                                progress_callback(
                                    0.4 + 0.2 * (r2_done / max(1, total_r2)),
                                    f"Literature Grounding ({r2_done}/{total_r2}): {node}" if is_en else f"文献全文溯源 ({r2_done}/{total_r2}): {node}"
                                )
                            if had_sents and res:
                                cat = res.get("primary_type")
                                canon_cat, is_valid = self.canonicalize_entity_category(cat)
                                if is_valid and canon_cat != "Uncertain":
                                    self.entity_annotations[node] = {
                                        **res,
                                        "primary_type": canon_cat,
                                        "round": 2,
                                        "status": "resolved"
                                    }
                                    self.stats["round2_hits"] += 1
                                else:
                                    self.entity_annotations[node] = {
                                        **res,
                                        "primary_type": "Uncertain",
                                        "round": 2,
                                        "status": "uncertain"
                                    }

                log(f"✅ [Stage 3 Complete] Resolved {self.stats['round2_hits']} entities from paper text." if is_en else f"✅ [阶段三完成] 文献全文溯源成功解析 {self.stats['round2_hits']} 个实体。")

            # === Round 3: 通用常识兜底 (ThreadPoolExecutor 并发) ===
            if enable_round3_world and self.llm_agent:
                r3_candidates = [n for n in sorted_nodes if self.entity_annotations.get(n, {}).get("status") == "uncertain"]
                total_r3 = len(r3_candidates)
                log(f"🌐 [Stage 4: Domain Prior Knowledge] {total_r3} candidates queued for biomedical knowledge reasoning..." if is_en else f"🌐 [阶段四: 全局先验研判] 共有 {total_r3} 个待决实体进入领域先验知识综合研判...")

                r3_lock = threading.Lock()
                r3_done = 0

                def task_r3(n_name: str):
                    res = self._llm_classify_entity_round3(n_name, language=language)
                    return n_name, res

                with ThreadPoolExecutor(max_workers=workers_cnt) as executor:
                    futures = {executor.submit(task_r3, n): n for n in r3_candidates}
                    for fut in as_completed(futures):
                        node, res = fut.result()
                        with r3_lock:
                            r3_done += 1
                            if progress_callback:
                                progress_callback(
                                    0.6 + 0.15 * (r3_done / max(1, total_r3)),
                                    f"Domain Knowledge ({r3_done}/{total_r3}): {node}" if is_en else f"领域先验研判 ({r3_done}/{total_r3}): {node}"
                                )
                            cat = res.get("primary_type")
                            canon_cat, is_valid = self.canonicalize_entity_category(cat)
                            if is_valid and canon_cat != "Uncertain":
                                self.entity_annotations[node] = {
                                    **res,
                                    "primary_type": canon_cat,
                                    "round": 3,
                                    "status": "resolved"
                                }
                                self.stats["round3_hits"] += 1
                            else:
                                self.entity_annotations[node] = {
                                    **res,
                                    "primary_type": "Uncertain",
                                    "round": 3,
                                    "status": "uncertain"
                                }

                log(f"✅ [Stage 4 Complete] Resolved {self.stats['round3_hits']} entities via domain prior knowledge." if is_en else f"✅ [阶段四完成] 领域先验知识研判成功解析 {self.stats['round3_hits']} 个实体。")

        # ----------------------------------------------------
        # 阶段 B: 关系机制深层分析与机制动词归纳 (ThreadPoolExecutor 并发)
        # ----------------------------------------------------
        if enable_relations and total_rels > 0 and self.llm_agent:
            target_relations = []
            valid_rels = set(self.ontology.get("relation_categories", {}).keys())
            for rel in relations:
                src = rel.get("source")
                tgt = rel.get("target")
                orig_rel = rel.get("relation", "相关")
                edge_key = f"{src}|{tgt}|{orig_rel}"
                if not force_reanalyze and edge_key in self.relation_annotations:
                    cur = self.relation_annotations[edge_key]
                    if cur.get("status") == "resolved" and cur.get("mechanism_category") in valid_rels:
                        continue
                target_relations.append(rel)

            total_rc = len(target_relations)
            log(f"🔗 Relation queue ready ({total_rc} relations, {workers_cnt} workers)..." if is_en else f"🔗 关系分析队列已就绪，共计 {total_rc} 条待分析关系 (并发线程数: {workers_cnt})...")

            rel_lock = threading.Lock()
            rel_done = 0

            def task_rel(r_item: Dict[str, Any]):
                src = r_item.get("source")
                tgt = r_item.get("target")
                orig_rel = r_item.get("relation", "相关")
                evidence = r_item.get("evidence", "")
                edge_key = f"{src}|{tgt}|{orig_rel}"
                res = self._llm_classify_relation(src, tgt, orig_rel, evidence, language=language)
                return edge_key, res

            with ThreadPoolExecutor(max_workers=workers_cnt) as executor:
                futures = {executor.submit(task_rel, r): r for r in target_relations}
                for fut in as_completed(futures):
                    edge_key, res = fut.result()
                    with rel_lock:
                        rel_done += 1
                        if progress_callback:
                            progress_callback(
                                0.75 + 0.25 * (rel_done / max(1, total_rc)),
                                f"Analyzing Relation Mechanism ({rel_done}/{total_rc})" if is_en else f"关系机制分析 ({rel_done}/{total_rc})"
                            )
                        mech_cat = res.get("mechanism_category", "")
                        canon_mech = self.canonicalize_relation_category(mech_cat)
                        self.relation_annotations[edge_key] = {
                            **res,
                            "mechanism_category": canon_mech,
                            "round": 1,
                            "status": "resolved"
                        }
                        self.stats["analyzed_relations"] += 1

            log(f"✅ [Relation Distillation Complete] Successfully distilled {self.stats['analyzed_relations']} mechanism verbs and categories." if is_en else f"✅ [关系机制分析完成] 成功提炼 {self.stats['analyzed_relations']} 条深层机制动词与类别。")

        # ----------------------------------------------------
        # 阶段 C: 统计与指标计算
        # ----------------------------------------------------
        valid_cats = set(self.ontology.get("entity_categories", {}).keys())
        resolved_cnt = sum(1 for a in self.entity_annotations.values() if a.get("status") == "resolved" and a.get("primary_type") in valid_cats and a.get("primary_type") != "Uncertain")
        uncertain_cnt = total_nodes - resolved_cnt
        self.stats["resolved_nodes"] = resolved_cnt
        self.stats["uncertain_nodes"] = uncertain_cnt
        self.stats["resolved_nodes"] = resolved_cnt
        self.stats["uncertain_nodes"] = uncertain_cnt

        if progress_callback:
            progress_callback(1.0, "分析全部完成！正在渲染工作台图谱...")

        log("🎉 [Workbench Pipeline Analysis Completed!]" if is_en else "🎉 [图谱分析工作台流水线全部执行完毕！]")
        return self.get_state()

    def override_entity(self, node_name: str, primary_type: str, sub_type: str = "", tags: Optional[List[str]] = None, reasoning: str = "专家手动调换修正") -> Dict[str, Any]:
        """学者人工手动调换/修正单实体分类"""
        record = {
            "primary_type": primary_type,
            "sub_type": sub_type,
            "tags": tags or [],
            "confidence": 1.0,
            "reasoning": reasoning,
            "round": "manual",
            "status": "resolved" if primary_type != "Uncertain" else "uncertain"
        }
        self.entity_annotations[node_name] = record
        return record

    def override_relation(self, source: str, target: str, orig_rel: str, mechanism_category: str, mechanism_verb: str, reasoning: str = "专家手动修正机制") -> Dict[str, Any]:
        """学者人工手动调换/修正单关系机制与动词"""
        edge_key = f"{source}|{target}|{orig_rel}"
        record = {
            "mechanism_category": mechanism_category,
            "mechanism_verb": mechanism_verb,
            "confidence": 1.0,
            "reasoning": reasoning,
            "round": "manual",
            "status": "resolved"
        }
        self.relation_annotations[edge_key] = record
        return record

    def get_state(self) -> Dict[str, Any]:
        """打包导出当前工作台全量状态"""
        return {
            "ontology_id": self.ontology.get("id", "universal_biored"),
            "ontology": self.ontology,
            "entity_annotations": self.entity_annotations,
            "relation_annotations": self.relation_annotations,
            "stats": self.stats
        }

    def load_state(self, state_dict: Dict[str, Any]):
        """从持久化字典加载恢复工作台全量状态"""
        if not isinstance(state_dict, dict):
            return
        if "entity_annotations" in state_dict:
            self.entity_annotations = state_dict.get("entity_annotations", {})
        if "relation_annotations" in state_dict:
            self.relation_annotations = state_dict.get("relation_annotations", {})
        if "ontology" in state_dict:
            self.ontology = state_dict.get("ontology", {})
        if "stats" in state_dict:
            self.stats = state_dict.get("stats", self.stats)


# =====================================================================
# 4. PathwayDecompositionEngine: 信号/关系通路拆解与建模核心引擎
# =====================================================================
class PathwayDecompositionEngine:
    """
    工作台第二阶段：信号/关系通路拆解与建模核心引擎 (Pathway Decomposition Engine)
    重构准则：
    1. 图谱不变性 (Graph Invariance)：严格从大图谱已知真实实体与关系中诱导有向子网络，绝不改写或虚构关系
    2. 严格有向性与动力学反馈循环 (Strict Directed Feedback Loops)：
       - 环路上每一条边必须是具备明确调控方向的有向功能边 (正作用/负作用/机制动词)
       - 坚决剔除包含“包含”、“语义相关”、“同义分类”的伪环，杜绝将单向“促进”或对称“相关”误判为“自促进”
    3. 文献来源清洗与工程权威对齐 (Canonical Literature Alignment)：
       - 拆分 | 等拼接符，过滤 AI分析/规则推导等噪音，严格映射至工程真实导入的 analyzed_files
    4. 极简学术规范大模型总结 (Academic Naming & Concise Mechanistic Summary)：
       - 输出专业学术命名 (LaTeX 形式，如 $\\text{CRL5}^{\\text{SOCS4}}$ 介导的泛素化降解途径与神经退行抗性)
       - 2~3 句精炼机制阐释 (若涉及真实闭环，适当融入动力学分析)
    5. 独立交互子图落盘与持久化存储 (projects/<pid>/pathways/)
    """

    def __init__(self,
                 llm_agent=None,
                 ontology: Optional[Dict[str, Any]] = None,
                 entity_annotations: Optional[Dict[str, Any]] = None,
                 relation_annotations: Optional[Dict[str, Any]] = None,
                 project_dir: Optional[str] = None,
                 canonical_papers: Optional[List[str]] = None):
        self.llm_agent = llm_agent
        self.ontology = ontology or {}
        self.entity_annotations = entity_annotations or {}
        self.relation_annotations = relation_annotations or {}
        self.project_dir = project_dir
        self.pathways_dir = os.path.join(project_dir, "pathways") if project_dir else None

        # 优先读取工程权威真实文献列表，用于文献对齐与硬上限约束
        self.canonical_papers = list(canonical_papers) if canonical_papers else []
        if not self.canonical_papers and project_dir:
            pkg_file = os.path.join(project_dir, "project.biokg")
            if os.path.exists(pkg_file):
                try:
                    with open(pkg_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.canonical_papers = data.get("analyzed_files", [])
                except Exception:
                    pass

    @staticmethod
    def parse_and_canonicalize_doc_sources(raw_doc: str, canonical_papers: Optional[List[str]] = None) -> List[str]:
        """
        清洗并规范化文献来源：
        1. 按 | ; \\n 拆分多文献合并串
        2. 彻底过滤 AI分析、逻辑推导、未知 等非真实文献标记
        3. 修复递归嵌套 PMID 字符串
        4. 严格对齐至当前工程真实导入的 canonical_papers 集合
        """
        if not raw_doc:
            return []
        parts = re.split(r'[\r\n|;；]+', str(raw_doc))
        clean_docs = set()
        canonical_list = canonical_papers or []

        for p in parts:
            p_str = p.strip()
            if not p_str:
                continue
            p_lower = p_str.lower()

            # 过滤非真实文献标识
            if any(ign in p_lower for ign in [
                "ai 分析", "ai分析", "ai 推导", "ai推导", "ai 智能逻辑推导", "ai analysis",
                "系统规则", "逻辑推导", "未知文献", "未知", "unknown", "none"
            ]):
                continue

            # 消除递归嵌套的 (PMID: ... (PMID: 42417502))))
            if "pmid:" in p_lower:
                m = re.findall(r'(\d{6,9})', p_str)
                if m:
                    target_pmid = m[-1]
                    matched_canonical = None
                    for c_doc in canonical_list:
                        if target_pmid in c_doc:
                            matched_canonical = c_doc
                            break
                    if matched_canonical:
                        clean_docs.add(matched_canonical)
                        continue
                    else:
                        clean_docs.add(f"PMID: {target_pmid}")
                        continue

            # 尝试与当前工程权威文献列表对齐
            matched = False
            for c_doc in canonical_list:
                c_clean = os.path.splitext(os.path.basename(c_doc))[0].strip().lower()
                p_clean = os.path.splitext(os.path.basename(p_str))[0].strip().lower()
                if c_clean in p_clean or p_clean in c_clean or c_clean[:35] in p_clean or p_clean[:35] in c_clean:
                    clean_docs.add(c_doc)
                    matched = True
                    break

            if not matched:
                clean_name = os.path.basename(p_str).strip()
                if clean_name and len(clean_name) >= 3:
                    clean_docs.add(clean_name)

        return sorted(list(clean_docs))

    def _fallback_text_match_score(self, node_name: str, user_query: str) -> float:
        """纯文本匹配保底评分 (范围 0.0 ~ 5.0)"""
        if not user_query or not node_name:
            return 0.0
        q_clean = user_query.strip().lower()
        n_clean = node_name.strip().lower()
        if n_clean == q_clean:
            return 5.0
        if n_clean in q_clean or q_clean in n_clean:
            return 4.0
        q_tokens = [w for w in re.split(r'[\s,，、;；。.!！?？]+', q_clean) if len(w) >= 2]
        if any(tok in n_clean for tok in q_tokens):
            return 2.5
        return 0.0

    def _llm_batch_score_nodes(self,
                               candidate_nodes: List[Dict[str, Any]],
                               user_query: str,
                               language: str = "zh") -> Dict[str, Dict[str, Any]]:
        """
        利用大模型对候选节点进行自然语言假说意图相关性与生物学角色偏好综合评分 (范围 0.0 ~ 5.0)
        - 委托 BioBrainAgent 统一处理，保持无侵入高内聚
        """
        if not self.llm_agent or not user_query or not candidate_nodes:
            return {}

        if hasattr(self.llm_agent, "score_workbench_nodes_intent"):
            try:
                return self.llm_agent.score_workbench_nodes_intent(
                    candidate_nodes=candidate_nodes,
                    user_query=user_query,
                    language=language
                )
            except Exception as e:
                print(f"⚠️ [PathwayDecompositionEngine] 节点意图评分委托调用异常: {e}")
                return {}
        return {}

    def score_candidate_seeds(self,
                              entities: List[Dict[str, Any]],
                              relations: List[Dict[str, Any]],
                              user_query: str = "",
                              max_seeds: int = 15,
                              language: str = "zh") -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        大模型与图论混合双轨评分候选节点发现 (LLM & Graph-Theory Hybrid Node Scoring)
        - 图论拓扑基础分 (0.0 ~ 5.0): deg_norm (0~2.5) + bet_norm (0~2.5)
        - 若有 user_query:
          - LLM 语义与情境角色打分 (0.0 ~ 5.0)
          - 总分 = graph_score + llm_score (理论满分 10.0)
        - 若 user_query 为空:
          - 全局自主模式，使用纯图论拓扑分 + 基础本体先验权重
        """
        if not relations:
            return []

        G = nx.DiGraph()
        for r in relations:
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "")).strip()
            if s and t and s != t:
                w = float(r.get("weight", 1))
                G.add_edge(s, t, weight=w)

        if len(G) == 0:
            return []

        deg_dict = dict(G.degree())
        max_deg = max(deg_dict.values()) if deg_dict else 1

        try:
            betweenness_dict = nx.betweenness_centrality(G) if len(G) > 2 else {}
        except Exception:
            betweenness_dict = {}

        candidate_nodes_meta = []
        for n in G.nodes():
            cat_annot = self.entity_annotations.get(n, {})
            primary_type = str(cat_annot.get("primary_type", "Uncertain"))
            sub_type = str(cat_annot.get("sub_type", ""))
            candidate_nodes_meta.append({
                "name": n,
                "primary_type": primary_type,
                "sub_type": sub_type,
                "degree": deg_dict.get(n, 0)
            })

        candidate_nodes_meta.sort(key=lambda x: x["degree"], reverse=True)

        user_query_clean = user_query.strip()
        llm_scores = {}
        if user_query_clean and self.llm_agent:
            llm_scores = self._llm_batch_score_nodes(candidate_nodes_meta, user_query_clean, language=language)

        scores = []
        for n in G.nodes():
            deg_val = deg_dict.get(n, 0)
            deg_norm = (deg_val / max_deg) * 2.5
            bet_norm = betweenness_dict.get(n, 0.0) * 2.5
            graph_score = deg_norm + bet_norm  # [0.0, 5.0]

            cat_annot = self.entity_annotations.get(n, {})
            primary_type = str(cat_annot.get("primary_type", "")).lower()
            sub_type = str(cat_annot.get("sub_type", "")).lower()
            combined_type = f"{primary_type} {sub_type}"

            if user_query_clean:
                # 模式 B: 焦点导向定向模式 (Targeted Mode)
                llm_item = llm_scores.get(n, {})
                llm_val = float(llm_item.get("score", 0.0))
                llm_role = str(llm_item.get("role", ""))

                # 如果 LLM 未返回任何结果（如断网或模型故障），执行保底文本匹配
                if not llm_scores:
                    llm_val = self._fallback_text_match_score(n, user_query_clean)
                    if llm_val > 0:
                        llm_role = "文本模糊匹配保底"

                total_score = graph_score + llm_val  # [0.0, 10.0]
                meta = {
                    "degree": deg_val,
                    "betweenness": betweenness_dict.get(n, 0.0),
                    "primary_type": cat_annot.get("primary_type", "Uncertain"),
                    "graph_score": round(graph_score, 3),
                    "llm_score": round(llm_val, 3),
                    "llm_role": llm_role,
                    "mode": "targeted"
                }
            else:
                # 模式 A: 全局自主涌现模式 (Global Emergent Mode)
                cat_weight = 0.0
                if any(k in combined_type for k in ["疾病", "disease", "表型", "phenotype", "pathology", "cancer", "tumor", "syndrome", "死", "凋亡", "自噬"]):
                    cat_weight += 2.0
                elif any(k in combined_type for k in ["药物", "drug", "化学", "chemical", "small molecule", "compound", "inhibitor", "agonist"]):
                    cat_weight += 1.8
                elif any(k in combined_type for k in ["激酶", "kinase", "受体", "receptor", "转录因子", "transcription factor"]):
                    cat_weight += 1.6
                elif any(k in combined_type for k in ["基因", "gene", "蛋白", "protein"]):
                    cat_weight += 1.2
                elif any(k in combined_type for k in ["过程", "process", "通路", "pathway"]):
                    cat_weight += 1.4

                total_score = graph_score + cat_weight
                meta = {
                    "degree": deg_val,
                    "betweenness": betweenness_dict.get(n, 0.0),
                    "primary_type": cat_annot.get("primary_type", "Uncertain"),
                    "graph_score": round(graph_score, 3),
                    "cat_weight": cat_weight,
                    "mode": "global"
                }

            scores.append((n, total_score, meta))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:max_seeds]

    def cluster_and_induce_subgraphs(self,
                                      relations: List[Dict[str, Any]],
                                      seeds: List[Any],
                                      max_pathways: int = 3,
                                      user_query: str = "",
                                      enable_diversity_penalty: bool = True,
                                      diversity_penalty_rate: float = 0.5) -> List[Dict[str, Any]]:
        """
        基于动力学语义阻抗与 MMR 累积惩罚的严格有向级联诱导 (Strict Directed Cascade Induction with Impedance & MMR Diversity)
        - 节点 100% 取自已知图谱，关系 100% 直取自 master_relations
        - 生化动力学语义阻抗建模：因果调控 R=1.0 (超导体流速快)；弱关联 R=2.5；包含从属 R=10.0 (高阻抗，保底连通不夺舍)
        - 修复 downstream[:5] 字典序机械截断 Bug：按边导纳 1/R 与节点度数排序探索
        - MMR 累积路径惩罚调度：动态衰减已走过节点对后续通路的吸引力，自然逼出第二、第三条平行黄金主轴
        - 父子节点层级容器化：包含从属不作伪因果箭头，识别为空间场域/复合物容器
        """
        if not relations or not seeds:
            return []

        from collections import Counter

        # 1. 规范化种子列表与其初始基准得分
        candidate_seeds = []
        base_scores = {}
        for idx, item in enumerate(seeds):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                s_name = str(item[0]).strip()
                s_score = float(item[1])
            else:
                s_name = str(item).strip()
                s_score = float(len(seeds) - idx)
            if s_name and s_name not in base_scores:
                candidate_seeds.append(s_name)
                base_scores[s_name] = s_score

        # 2. 构建有向网络，计算生化动力学语义阻抗与导纳，并记录从属包含层级结构
        G_dir = nx.DiGraph()
        parent_map = {}  # child -> set of parents
        containment_edges = set()

        for r in relations:
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "")).strip()
            if s and t and s != t:
                orig_rel = str(r.get("relation", r.get("relation_type", ""))).strip()
                edge_k = f"{s}|{t}|{orig_rel}"
                ann = self.relation_annotations.get(edge_k, {})
                verb = str(ann.get("mechanism_verb", orig_rel)).lower()
                cat = str(ann.get("mechanism_category", "")).lower()

                # 从属/包含/分类层级边 (高阻抗 R=10.0，导纳 0.05，仅作场域容器，不作为传导级联步)
                is_containment = (orig_rel in ["包含", "属于", "从属", "part_of", "subclass_of", "member_of", "hierarchy", "component"]) or any(
                    kw in (cat + " " + verb) for kw in ["part_of", "subclass", "hierarchy", "component", "complex", "包含", "属于", "从属", "复合体", "亚型", "分类"]
                )

                # 生化功能因果调控边 (超导体 R=1.0，导纳 1.0)
                is_causal = (orig_rel in ["正作用", "负作用"]) or any(kw in (cat + " " + verb) for kw in [
                    "inhibit", "activat", "promot", "suppress", "degrad", "phospho", "cleav", "hydroly",
                    "bind", "cataly", "induc", "target", "upregulat", "downregulat", "modulat",
                    "抑制", "激活", "促进", "阻断", "降解", "诱导", "磷酸化", "催化", "水解", "裂解", "结合", "调控", "上调", "下调"
                ])

                if is_containment:
                    r_imp = 10.0
                    conductance = 0.05
                    parent_map.setdefault(t, set()).add(s)
                    containment_edges.add((s, t))
                elif is_causal:
                    r_imp = 1.0
                    conductance = 1.0
                else:
                    r_imp = 2.5
                    conductance = 0.4

                G_dir.add_edge(s, t, rel=r, is_regulatory=is_causal, is_containment=is_containment,
                               impedance=r_imp, conductance=conductance, weight=r_imp)

        if len(G_dir) == 0:
            return []

        deg_dict = dict(G_dir.degree())
        max_deg = max(deg_dict.values()) if deg_dict else 1

        # 统计节点因果出度，上游驱动因子/中继酶更优先作为级联源头，纯终点表型后置
        out_causal_dict = {n: sum(1 for succ in G_dir.successors(n) if G_dir[n][succ].get("is_regulatory")) for n in G_dir}
        for s in candidate_seeds:
            out_c = out_causal_dict.get(s, 0)
            driver_factor = 1.3 if out_c > 0 else 0.75
            base_scores[s] = base_scores.get(s, 1.0) * driver_factor

        candidate_seeds.sort(key=lambda x: base_scores.get(x, 1.0), reverse=True)

        visited_node_counts = Counter()
        selected_subsets = []
        seen_node_sets = []

        # 预计算全图中所有长度为 2~3 的精炼生物功能因果闭环 (Feedback Cycles)，用于后续豁免保护
        G_causal = nx.DiGraph()
        for u, v, d in G_dir.edges(data=True):
            if d.get("is_regulatory"):
                G_causal.add_edge(u, v, **d)

        all_cycles = []
        try:
            raw_cycles = list(nx.simple_cycles(G_causal))
            for c in raw_cycles:
                if 2 <= len(c) <= 3:
                    all_cycles.append(c)
        except Exception:
            pass

        # 3. 循环多轮诱导不同方向的高清因果通路 (级联主干 + 反馈闭环优先编织)
        while len(selected_subsets) < max_pathways:
            # MMR 动态种子重排：已曝光节点得分受衰减
            if enable_diversity_penalty and visited_node_counts:
                active_seeds_order = sorted(
                    candidate_seeds,
                    key=lambda s: base_scores.get(s, 1.0) * ((1.0 - diversity_penalty_rate) ** visited_node_counts[s]),
                    reverse=True
                )
            else:
                active_seeds_order = candidate_seeds

            found_new = False
            for seed in active_seeds_order:
                if seed not in G_dir:
                    continue

                # 动态阻抗图：已走过节点间的边阻抗按访问频次累进增加
                G_curr = nx.DiGraph()
                for u, v, d in G_dir.edges(data=True):
                    base_w = d["impedance"]
                    if enable_diversity_penalty:
                        w = base_w * (1.6 ** (visited_node_counts[u] + visited_node_counts[v]))
                    else:
                        w = base_w
                    d_copy = dict(d)
                    d_copy["weight"] = w
                    G_curr.add_edge(u, v, **d_copy)

                cascade_nodes = {seed}
                immune_nodes = set()
                immune_edges = set()

                # A. 确定前向因果传导主干长链 (Downstream Spine Tracing，深入 3~4 步)
                curr = seed
                for _ in range(4):
                    causal_succs = []
                    for succ in G_curr.successors(curr):
                        if G_curr[curr][succ].get("is_regulatory") and succ not in cascade_nodes:
                            # 亚型折叠：若父复合体已在主干中，不拉入下属组分以消除局部密集团簇
                            if any(p in cascade_nodes for p in parent_map.get(succ, set())):
                                continue
                            vis_factor = ((1.0 - diversity_penalty_rate) ** visited_node_counts[succ]) if enable_diversity_penalty else 1.0
                            score = (1.0 + (deg_dict.get(succ, 1) / max_deg)) * vis_factor
                            causal_succs.append((succ, score))
                    if causal_succs:
                        causal_succs.sort(key=lambda x: x[1], reverse=True)
                        best_next = causal_succs[0][0]
                        cascade_nodes.add(best_next)
                        curr = best_next
                    else:
                        break

                # B. 向上追溯上游因果触发输入 (Upstream Spine Tracing，向上 2~3 步)
                curr = seed
                for _ in range(3):
                    causal_preds = []
                    for pred in G_curr.predecessors(curr):
                        if G_curr[pred][curr].get("is_regulatory") and pred not in cascade_nodes:
                            if any(p in cascade_nodes for p in parent_map.get(pred, set())):
                                continue
                            vis_factor = ((1.0 - diversity_penalty_rate) ** visited_node_counts[pred]) if enable_diversity_penalty else 1.0
                            score = (1.0 + (deg_dict.get(pred, 1) / max_deg)) * vis_factor
                            causal_preds.append((pred, score))
                    if causal_preds:
                        causal_preds.sort(key=lambda x: x[1], reverse=True)
                        best_prev = causal_preds[0][0]
                        cascade_nodes.add(best_prev)
                        curr = best_prev
                    else:
                        break

                # C. 打通到未被充分探索的高优先级代表元的最短路径
                target_peers = [o for o in active_seeds_order if o != seed and o in G_curr]
                for other in target_peers[:6]:
                    if len(cascade_nodes) >= 8:
                        break
                    if any(p in cascade_nodes for p in parent_map.get(other, set())):
                        continue
                    try:
                        if nx.has_path(G_curr, seed, other):
                            p = nx.shortest_path(G_curr, seed, other, weight="weight")
                            if len(p) <= 4:
                                cascade_nodes.update(p)
                        elif nx.has_path(G_curr, other, seed):
                            p = nx.shortest_path(G_curr, other, seed, weight="weight")
                            if len(p) <= 4:
                                cascade_nodes.update(p)
                    except Exception:
                        pass

                # D. 反馈闭环优先检测与绝对豁免保护 (Feedback Loop Immunity，保留与主干耦合的闭环)
                matched_cycles = []
                for c in all_cycles:
                    c_set = set(c)
                    overlap = len(c_set.intersection(cascade_nodes))
                    if overlap >= 1:
                        matched_cycles.append((c, overlap))
                matched_cycles.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)

                for c, _ in matched_cycles[:2]:
                    c_set = set(c)
                    if len(cascade_nodes.union(c_set)) <= 11:
                        cascade_nodes.update(c_set)
                        immune_nodes.update(c_set)
                        for i in range(len(c)):
                            u = c[i]
                            v = c[(i + 1) % len(c)]
                            immune_edges.add((u, v))

                # E. 连通性保全修剪 (确保 100% 连通图，保护割点桥梁分子，杜绝游离孤岛)
                G_sub_undir = nx.Graph()
                for n in cascade_nodes:
                    G_sub_undir.add_node(n)
                for u, v in G_dir.edges():
                    if u in cascade_nodes and v in cascade_nodes:
                        G_sub_undir.add_edge(u, v)

                # 严格保留与主种子 seed 连通的主分量 (彻底清除任何游离孤岛)
                if seed in G_sub_undir:
                    seed_comp = nx.node_connected_component(G_sub_undir, seed)
                    cascade_nodes = set(seed_comp)
                    G_sub_undir = G_sub_undir.subgraph(cascade_nodes).copy()

                # 超限节点精炼修剪：绝不可裁切割点桥梁与豁免闭环节点！
                max_limit = 10
                while len(cascade_nodes) > max_limit:
                    articulation_points = set(nx.articulation_points(G_sub_undir))
                    candidates_to_prune = [
                        n for n in cascade_nodes
                        if n != seed and n not in immune_nodes and n not in articulation_points
                    ]
                    if not candidates_to_prune:
                        break
                    candidates_to_prune.sort(
                        key=lambda x: (visited_node_counts[x], -deg_dict.get(x, 0)),
                        reverse=True
                    )
                    node_to_remove = candidates_to_prune[0]
                    cascade_nodes.remove(node_to_remove)
                    G_sub_undir.remove_node(node_to_remove)

                # F. 边的高质量筛选 (因果骨干 + 闭环回路 + 结构包含，稀疏化旁路冗余弱相关)
                sub_rels = []
                for r in relations:
                    src = str(r.get("source", "")).strip()
                    tgt = str(r.get("target", "")).strip()
                    if src in cascade_nodes and tgt in cascade_nodes and src != tgt:
                        rel_type = str(r.get("relation", r.get("relation_type", ""))).strip()
                        is_immune = (src, tgt) in immune_edges
                        is_reg = (rel_type in ["正作用", "负作用"])
                        is_comp = (src, tgt) in containment_edges
                        if is_immune or is_reg or is_comp:
                            sub_rels.append(r)
                        elif rel_type in ["相关", "相互作用"]:
                            # 仅当两点间无直接因果或闭环连接时补充保底
                            sub_rels.append(r)

                if len(cascade_nodes) >= 3 and len(sub_rels) >= 2:
                    # 检查 Jaccard 相似度，多样性模式下阈值收紧至 0.45
                    jaccard_max = 0.0
                    for existing in seen_node_sets:
                        jacc = len(cascade_nodes.intersection(existing)) / max(1, len(cascade_nodes.union(existing)))
                        if jacc > jaccard_max:
                            jaccard_max = jacc

                    sim_threshold = 0.45 if enable_diversity_penalty else 0.65
                    if jaccard_max < sim_threshold:
                        seen_node_sets.append(cascade_nodes)
                        visited_node_counts.update(cascade_nodes)
                        selected_subsets.append({
                            "cluster_id": len(selected_subsets) + 1,
                            "seed_nodes": [seed],
                            "nodes": sorted(list(cascade_nodes)),
                            "relations": sub_rels,
                            "immune_nodes": list(immune_nodes)
                        })
                        found_new = True
                        break

            if not found_new:
                break

        return selected_subsets[:max_pathways]

    def detect_feedback_loops(self,
                              subgraph_nodes: List[str],
                              subgraph_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        严谨生物学动力学自调节反馈回路检测 (Strict Biological Feedback Loop Detection)
        1. 严格遵循有向性：必须是顺次指向闭合的有向简单环 (Directed Simple Cycles)
        2. 严格限定调控语义：环路上每一条边必须是明确的功能性因果调控边 (正作用/负作用/机制动词)，
           坚决过滤包含“包含”、“语义相关”、“分类层级”的伪环；
        3. 二元环反伪互促过滤：对于 2 元环 A <-> B，必须双方均为明确反向调控机制，杜绝无向关系对称化假象；
        4. 极性代数判定自抑制 (负反馈) 与自促进 (正反馈)。
        """
        G_reg = nx.DiGraph()
        rel_lookup = {}

        for r in subgraph_relations:
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "")).strip()
            if not s or not t or s == t:
                continue

            orig_rel = str(r.get("relation", r.get("relation_type", ""))).strip()
            edge_k = f"{s}|{t}|{orig_rel}"
            ann = self.relation_annotations.get(edge_k, {})
            mech_cat = str(ann.get("mechanism_category", "")).lower()
            mech_verb = str(ann.get("mechanism_verb", orig_rel)).lower()
            evi_text = str(r.get("evidence", r.get("reason", ""))).strip()

            # 严格过滤非因果调控关系（包含、相关、属于等）
            if orig_rel in ["包含", "相关", "属于", "部分", "等价", "hierarchy", "association", "correlation"]:
                if not any(kw in (mech_cat + " " + mech_verb) for kw in [
                    "inhibit", "activat", "promot", "suppress", "degrad", "phospho",
                    "抑制", "激活", "促进", "阻断", "降解", "磷酸化"
                ]):
                    continue

            is_neg = (orig_rel == "负作用") or any(kw in (mech_cat + " " + mech_verb) for kw in [
                "negative", "inhibit", "degrad", "suppress", "block", "repress",
                "downregulat", "负", "抑制", "降解", "阻断", "下调", "去磷酸化", "衰减"
            ])
            is_pos = (orig_rel == "正作用") or any(kw in (mech_cat + " " + mech_verb) for kw in [
                "positive", "activat", "promot", "induc", "upregulat", "phospho",
                "正", "激活", "促进", "诱导", "上调", "磷酸化"
            ])

            # 只有明确属于功能调控的边才进入调控图
            if is_neg or is_pos:
                G_reg.add_edge(s, t)
                rel_lookup[(s, t)] = {
                    "relation": r,
                    "is_neg": is_neg,
                    "verb": ann.get("mechanism_verb", orig_rel),
                    "evidence": evi_text,
                    "doc_source": str(r.get("doc_source", ""))
                }

        try:
            raw_cycles = list(nx.simple_cycles(G_reg))
        except Exception:
            raw_cycles = []

        valid_cycles = [c for c in raw_cycles if 2 <= len(c) <= 5]

        unique_cycles = []
        seen_cycle_sets = set()
        for c in valid_cycles:
            frozen = frozenset(c)
            if frozen not in seen_cycle_sets:
                seen_cycle_sets.add(frozen)
                unique_cycles.append(c)

        detected_loops = []
        for cycle in unique_cycles:
            # 2元环严格鉴伪 (Anti-Symmetric False Positive Check)
            if len(cycle) == 2:
                u, v = cycle[0], cycle[1]
                edge_uv = rel_lookup.get((u, v), {})
                edge_vu = rel_lookup.get((v, u), {})
                # 若双向边依据完全相同且关系动词一致，说明是提取时无向关系对称化产生的伪环，坚决丢弃
                if edge_uv.get("evidence") == edge_vu.get("evidence") and edge_uv.get("verb") == edge_vu.get("verb"):
                    continue

            neg_count = 0
            cycle_steps = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge_info = rel_lookup.get((u, v), {})
                is_neg = edge_info.get("is_neg", False)
                if is_neg:
                    neg_count += 1
                cycle_steps.append((u, v, edge_info.get("verb", "调控"), is_neg))

            is_negative_loop = (neg_count % 2 == 1)
            loop_type = "negative" if is_negative_loop else "positive"
            cycle_str = " ➔ ".join(cycle) + " ➔ " + cycle[0]

            detected_loops.append({
                "loop_type": loop_type,
                "nodes": cycle,
                "cycle_str": cycle_str,
                "neg_count": neg_count,
                "is_self_inhibition": is_negative_loop,
                "is_self_promotion": not is_negative_loop,
                "steps": cycle_steps
            })

        return detected_loops

    def detect_cross_paper_bridges(self,
                                   subgraph_nodes: List[str],
                                   subgraph_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        跨文献合成通路识别 (Cross-Paper Bridge Tagging)
        严格清洗 doc_source 并对齐当前工程权威文献，严谨判定是否跨越 >=2 篇真实文献
        """
        paper_sources = set()
        for r in subgraph_relations:
            raw_doc = str(r.get("doc_source", "")).strip()
            parsed = self.parse_and_canonicalize_doc_sources(raw_doc, self.canonical_papers)
            paper_sources.update(parsed)

        clean_papers = sorted(list(paper_sources))
        return {
            "is_cross_paper": len(clean_papers) >= 2,
            "paper_sources": clean_papers,
            "paper_count": len(clean_papers)
        }

    def _parse_llm_json(self, raw_str: str) -> Optional[Dict[str, Any]]:
        """清洗并严格反序列化大模型返回的 JSON"""
        if not raw_str:
            return None
        cleaned = raw_str.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def synthesize_pathway_narrative(self,
                                     pathway_idx: int,
                                     subgraph_nodes: List[str],
                                     subgraph_relations: List[Dict[str, Any]],
                                     feedback_loops: List[Dict[str, Any]],
                                     cross_paper_info: Dict[str, Any],
                                     user_query: str = "",
                                     language: str = "zh") -> Dict[str, Any]:
        """
        大模型驱动的极简学术命名与机制解读 (Scholarly Naming & Concise Mechanistic Narrative)
        - title: 规范化学术命名 (支持 LaTeX，如 $\\text{CRL5}^{\\text{SOCS4}}$ 介导的泛素化降解途径与神经退行抗性)
        - summary: 2~3 句精炼机制阐释 (若涉及真实闭环回路，适当分析其自反馈动力学效应)
        - 坚决杜绝 AI 凭空捏造不存在的因果关系
        """
        is_en = "en" in str(language).lower()

        # 整理真实存在的已知实体与机制连线
        nodes_desc = []
        for n in subgraph_nodes:
            ann = self.entity_annotations.get(n, {})
            ptype = ann.get("primary_type", "")
            type_str = f"[{ptype}]" if ptype and ptype != "Uncertain" else ""
            nodes_desc.append(f"- {n} {type_str}".strip())

        edges_desc = []
        for r in subgraph_relations:
            s = r.get("source")
            t = r.get("target")
            rel = r.get("relation", r.get("relation_type", "相关"))
            edge_k = f"{s}|{t}|{rel}"
            ann = self.relation_annotations.get(edge_k, {})
            verb = ann.get("mechanism_verb", rel)
            evi = str(r.get("evidence", r.get("reason", "无")))[:80]
            is_comp = (rel in ["包含", "属于", "component", "hierarchy"]) or any(k in str(verb).lower() for k in ["包含", "属于", "从属", "component", "complex"])
            tag = " [空间场域/组成结构]" if is_comp else ""
            edges_desc.append(f"- {s} ➔ {t} : `{verb}`{tag} (原文依据: {evi})")

        loops_desc = []
        for lp in feedback_loops:
            ltype = "自抑制负反馈回路 (Negative Feedback Loop)" if lp.get("is_self_inhibition") else "自促进正反馈回路 (Positive Feedback Loop)"
            loops_desc.append(f"- 有向闭环: {lp.get('cycle_str')} [{ltype}]")

        # 委托 BioBrainAgent 结合拓扑与实体信息生成学术通路命名与机制总结叙事
        if self.llm_agent and hasattr(self.llm_agent, "synthesize_pathway_narrative"):
            try:
                parsed = self.llm_agent.synthesize_pathway_narrative(
                    pathway_idx=pathway_idx,
                    nodes_desc=nodes_desc,
                    edges_desc=edges_desc,
                    loops_desc=loops_desc,
                    user_query=user_query,
                    language=language
                )
                if parsed and isinstance(parsed, dict) and "title" in parsed and "summary" in parsed:
                    parsed["is_fallback"] = False
                    return parsed
            except Exception as e:
                print(f"⚠️ [PathwayDecompositionEngine] LLM 机制叙事委托调用异常: {e}")

        # 确定性拓扑降级保底 (无 LLM 或 LLM 故障时的稳定兜底)
        first_node = subgraph_nodes[0] if subgraph_nodes else "NodeA"
        last_node = subgraph_nodes[-1] if len(subgraph_nodes) > 1 else first_node
        default_title = f"{first_node} ➔ {last_node} 信号调控通路" if not is_en else f"{first_node} to {last_node} Signaling Pathway"

        loop_text = ""
        if feedback_loops:
            l_types = "自抑制负反馈" if feedback_loops[0].get("is_self_inhibition") else "自促进正反馈"
            loop_text = f" 经算法检测，该通路中包含由 {feedback_loops[0].get('cycle_str')} 构成的{l_types}回路，参与维持系统自稳态调节。" if not is_en else f" Features a verified feedback loop ({feedback_loops[0].get('cycle_str')}) for autoregulation."

        default_summary = (
            f"该通路由大图谱中已知实体与真实调控关系诱导构成，展现了以 {first_node} 为起点的分子信号传导级联。{loop_text}"
            if not is_en else
            f"Induced from verified interactions, delineating a molecular cascade initiating from {first_node}.{loop_text}"
        )

        return {
            "title": default_title,
            "summary": default_summary,
            "is_fallback": True
        }

    def run_decomposition(self,
                          entities: List[Dict[str, Any]],
                          relations: List[Dict[str, Any]],
                          user_query: str = "",
                          enable_loops: bool = True,
                          enable_bridges: bool = True,
                          enable_subgraphs: bool = True,
                          max_pathways: int = 3,
                          enable_diversity_penalty: bool = True,
                          diversity_penalty_rate: float = 0.5,
                          layout_mode: str = "force",
                          language: str = "zh",
                          progress_callback: Optional[Callable] = None,
                          log_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        全流程一键执行信号/关系通路拆解与建模
        """
        def log(msg: str):
            if log_callback:
                log_callback(msg)
            print(f"[PathwayDecomposition] {msg}")

        def prog(frac: float, msg: str):
            if progress_callback:
                progress_callback(frac, msg)

        is_en = "en" in str(language).lower()

        log("🚀 开始执行第二阶段：信号/关系通路拆解与建模流水线..." if not is_en else "🚀 Starting Stage 2: Pathway Decomposition & Modeling Pipeline...")
        prog(0.05, "正在初始化图拓扑与多维评分引擎..." if not is_en else "Initializing graph topology and multi-criteria scoring engine...")

        if not relations or len(relations) == 0:
            log("⚠️ 当前工程缺乏关系连线数据，无法拆解通路。" if not is_en else "⚠️ Insufficient relations to decompose pathways.")
            prog(1.0, "分析完成 (无有效通路)" if not is_en else "Completed (No relations)")
            return []

        # 1. 种子节点多维打分与挑选 (大模型语义与图论混合双轨)
        if user_query and self.llm_agent:
            prog(0.12, "正在调用大模型进行假说意图相关性与生物角色智能打分..." if not is_en else "Invoking LLM to score node relevance and role preferences...")
        else:
            prog(0.12, "正在计算多维节点图论重要性得分..." if not is_en else "Computing multi-criteria graph importance scores...")

        seed_scores = self.score_candidate_seeds(entities, relations, user_query=user_query, max_seeds=20, language=language)
        seed_names = [s[0] for s in seed_scores]
        if user_query:
            top_scored_str = ", ".join([f"{s[0]}({s[1]:.1f}分)" for s in seed_scores[:6]])
            log(f"🧠 [混合打分] 成功识别假说相关实体并排定种子优先级: {top_scored_str}" if not is_en else f"🧠 [Hybrid Scoring] Identified focus seeds: {top_scored_str}")
        else:
            log(f"🎯 成功挑选出 {len(seed_names)} 个核心候选种子实体: {', '.join(seed_names[:8])}..." if not is_en else f"🎯 Identified {len(seed_names)} candidate seeds: {', '.join(seed_names[:8])}...")

        # 2. 严格基于已知大图谱有向调控边与动力学语义阻抗诱导子图
        prog(0.30, "正在沿有向调控级联诱导真实子图 (MMR累积多样性探索)..." if not is_en else "Inducing directed subgraphs along verified cascades (MMR diversity)...")
        subgraphs = self.cluster_and_induce_subgraphs(
            relations=relations,
            seeds=seed_scores,
            max_pathways=max_pathways,
            user_query=user_query,
            enable_diversity_penalty=enable_diversity_penalty,
            diversity_penalty_rate=diversity_penalty_rate
        )
        log(f"📦 成功诱导出 {len(subgraphs)} 个高价值通路子网络模块" if not is_en else f"📦 Successfully induced {len(subgraphs)} pathway modules")

        if not subgraphs:
            log("⚠️ 未能提取出足够规模的连通子网络。" if not is_en else "⚠️ Unable to induce connected subgraphs.")
            prog(1.0, "分析完成" if not is_en else "Completed")
            return []

        # 3. 逐一深入建模各通路
        results = []
        total_sub = len(subgraphs)

        for idx, sub in enumerate(subgraphs):
            cid = sub["cluster_id"]
            nodes = sub["nodes"]
            rels = sub["relations"]

            step_base = 0.35 + (0.55 * (idx / total_sub))
            prog(step_base, f"正在深度建模第 {idx+1}/{total_sub} 条通路: 包含 {len(nodes)} 节点, {len(rels)} 连线..." if not is_en else f"Modeling pathway {idx+1}/{total_sub} ({len(nodes)} nodes, {len(rels)} edges)...")

            # A. 严谨动力学自反馈检测
            loops = []
            if enable_loops:
                loops = self.detect_feedback_loops(nodes, rels)
                if loops:
                    neg_l = [l for l in loops if l["is_self_inhibition"]]
                    pos_l = [l for l in loops if l["is_self_promotion"]]
                    log(f"  通路 {cid}: 捕获 {len(loops)} 个反馈闭环 (自抑制: {len(neg_l)}, 自促进: {len(pos_l)})" if not is_en else f"  Pathway {cid}: Captured {len(loops)} loops (Inhibition: {len(neg_l)}, Promotion: {len(pos_l)})")

            # B. 跨文献合成通路识别与文献清洗
            bridges = {"is_cross_paper": False, "paper_sources": [], "paper_count": 1}
            if enable_bridges:
                bridges = self.detect_cross_paper_bridges(nodes, rels)
                if bridges.get("is_cross_paper"):
                    log(f"  通路 {cid}: 标记为跨文献通路 (涉及 {bridges.get('paper_count')} 篇真实文献)" if not is_en else f"  Pathway {cid}: Tagged cross-paper pathway ({bridges.get('paper_count')} papers)")

            # C. 大模型极简学术命名与机制解读
            prog(step_base + 0.1, f"正在调用大模型生成第 {idx+1} 条通路的学术命名与机制阐述..." if not is_en else f"Invoking LLM narrative for pathway {idx+1}...")
            narrative = self.synthesize_pathway_narrative(
                pathway_idx=cid,
                subgraph_nodes=nodes,
                subgraph_relations=rels,
                feedback_loops=loops,
                cross_paper_info=bridges,
                user_query=user_query,
                language=language
            )

            # D. 生成独立 PyVis 精简交互子图
            html_rel_path = ""
            html_abs_path = ""
            if enable_subgraphs and self.pathways_dir:
                try:
                    from IO_SYS import GraphVisualizer
                    os.makedirs(self.pathways_dir, exist_ok=True)
                    fname = f"pathway_{cid}_subgraph.html"
                    html_abs_path = os.path.join(self.pathways_dir, fname)
                    html_rel_path = os.path.join("pathways", fname)

                    sub_ents = [e for e in entities if e.get("standard_name") in nodes]
                    vis = GraphVisualizer()
                    vis.generate_pathway_html(
                        pathway_entities=sub_ents if sub_ents else nodes,
                        pathway_relations=rels,
                        feedback_loops=loops,
                        cross_paper_bridges=bridges,
                        entity_annotations=self.entity_annotations,
                        relation_annotations=self.relation_annotations,
                        ontology=self.ontology,
                        output_file=html_abs_path,
                        output_lang=language,
                        height="480px",
                        layout_mode=layout_mode
                    )
                except Exception as e:
                    log(f"⚠️ 生成子图 HTML 遇到小异常: {e}")

            pathway_obj = {
                "id": f"pw_{cid}",
                "cluster_id": cid,
                "title": narrative.get("title", f"通路 {cid}" if not is_en else f"Pathway {cid}"),
                "summary": narrative.get("summary", ""),
                "nodes": nodes,
                "relations": rels,
                "containers": sub.get("containers", []),
                "layout_mode": layout_mode,
                "has_feedback_loop": len(loops) > 0,
                "feedback_loops": loops,
                "is_cross_paper": bridges.get("is_cross_paper", False),
                "cross_paper_bridges": bridges,
                "html_file": html_rel_path,
                "is_fallback": narrative.get("is_fallback", False)
            }
            results.append(pathway_obj)

        # 4. 全量落盘持久化
        prog(0.96, "正在持久化通路数据至项目专属目录..." if not is_en else "Persisting pathway data to project directory...")
        options = {
            "enable_loops": enable_loops,
            "enable_bridges": enable_bridges,
            "enable_subgraphs": enable_subgraphs,
            "max_pathways": max_pathways,
            "enable_diversity_penalty": enable_diversity_penalty,
            "diversity_penalty_rate": diversity_penalty_rate,
            "layout_mode": layout_mode
        }
        self.save_manifest(results, user_query=user_query, options=options)

        prog(1.0, f"🎉 成功解构出 {len(results)} 条核心信号通路！" if not is_en else f"🎉 Successfully modeled {len(results)} pathways!")
        log(f"🎉 信号通路拆解与建模流水线全部执行完毕，共生成 {len(results)} 条通路！" if not is_en else f"🎉 Pathway pipeline finished. Generated {len(results)} pathways.")
        return results

    def save_manifest(self, pathways: List[Dict[str, Any]], user_query: str = "", options: Optional[Dict[str, Any]] = None) -> str:
        """持久化保存分析结果到 projects/<active_pid>/pathways/pathways_manifest.json"""
        if not self.pathways_dir:
            return ""
        try:
            os.makedirs(self.pathways_dir, exist_ok=True)
            manifest_path = os.path.join(self.pathways_dir, "pathways_manifest.json")
            manifest_data = {
                "updated_at": int(time.time()),
                "user_query": user_query,
                "options": options or {},
                "pathway_count": len(pathways),
                "pathways": pathways
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, ensure_ascii=False, indent=2)
            return manifest_path
        except Exception as e:
            print(f"⚠️ [save_manifest] 持久化通路数据失败: {e}")
            return ""

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        """从 projects/<active_pid>/pathways/pathways_manifest.json 加载已有分析结果"""
        if not self.pathways_dir:
            return None
        manifest_path = os.path.join(self.pathways_dir, "pathways_manifest.json")
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ [load_manifest] 读取通路快照失败: {e}")
            return None
