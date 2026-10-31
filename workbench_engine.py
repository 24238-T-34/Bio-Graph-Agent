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
        """Round 1: 依据局部网络微上下文分类单节点 (严格适配系统语言)"""
        is_en = "en" in str(language).lower()
        cat_str = self._format_ontology_categories(language=language)

        if is_en:
            system_prompt = (
                "You are a senior computational biologist and biomedical ontology curator. Your task is to rigorously classify biomedical entities based on local graph topological micro-context.\n"
                "If the context provides sufficient evidence, classify the entity into one of the designated candidate categories. If evidence is insufficient, ambiguous, or lacks confidence, you MUST directly set primary_type to \"Uncertain\"."
            )
            neighbors_str = "\n".join(micro_ctx["neighbors"]) if micro_ctx["neighbors"] else "No direct records"
            evidence_str = "\n".join(micro_ctx["evidence"]) if micro_ctx["evidence"] else "No direct literature micro-evidence"

            user_content = (
                f"【Entity to Classify】: {node_name}\n"
                f"【Known Aliases】: {', '.join(micro_ctx['aliases']) if micro_ctx['aliases'] else 'None'}\n"
                f"【Network Neighbor Relations】:\n{neighbors_str}\n"
                f"【Local Literature Evidence】:\n{evidence_str}\n\n"
                f"【Candidate Entity Categories】:\n{cat_str}\n\n"
                "【Output Instructions】:\n"
                "Strictly output a single valid JSON object. All field values (sub_type, tags, reasoning) MUST be in English. Do not add markdown or explanation outside the JSON:\n"
                "{\n"
                "  \"primary_type\": \"<Candidate Category ID or Uncertain>\",\n"
                "  \"sub_type\": \"<Specific functional subtype or role in English>\",\n"
                "  \"tags\": [\"<Tag1>\", \"<Tag2>\"],\n"
                "  \"confidence\": 0.85,\n"
                "  \"reasoning\": \"<Concise justification in English, under 40 words>\"\n"
                "}"
            )
        else:
            system_prompt = (
                "你是一位资深的计算生物学与生物信息学本体专家。你的任务是根据图谱局部微上下文对生物医学实体进行严谨分类。\n"
                "若上下文证据足以判断，请归类到指定类别之一；若信息不足、缺乏把握或无法明确判定，务必直接输出 primary_type 为 \"Uncertain\"。"
            )
            neighbors_str = "\n".join(micro_ctx["neighbors"]) if micro_ctx["neighbors"] else "无直接记录"
            evidence_str = "\n".join(micro_ctx["evidence"]) if micro_ctx["evidence"] else "无直接原文微证据"

            user_content = (
                f"【待分析实体】: {node_name}\n"
                f"【实体已知别名】: {', '.join(micro_ctx['aliases']) if micro_ctx['aliases'] else '无'}\n"
                f"【网络邻居关系】:\n{neighbors_str}\n"
                f"【局部文献证据】:\n{evidence_str}\n\n"
                f"【候选实体类别集合】:\n{cat_str}\n\n"
                "【输出要求】:\n"
                "请严格以单条合法 JSON 格式输出，所有文本字段（sub_type, tags, reasoning）务必使用中文:\n"
                "{\n"
                "  \"primary_type\": \"<候选类别ID或Uncertain>\",\n"
                "  \"sub_type\": \"<细分子类别或具体生物学功能>\",\n"
                "  \"tags\": [\"<标签1>\", \"<标签2>\"],\n"
                "  \"confidence\": 0.85,\n"
                "  \"reasoning\": \"<简明推断理由，50字以内>\"\n"
                "}"
            )

        try:
            raw = self.llm_agent._ask_llm(system_prompt, user_content)
            res = self._parse_llm_json(raw)
            if isinstance(res, dict) and "primary_type" in res:
                return res
        except Exception as e:
            print(f"⚠️ [Round 1 Entity LLM Error] {node_name}: {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "LLM call error or invalid response" if is_en else "大模型调用异常或未能返回合法结构"
        }

    def _llm_classify_entity_round2(self, node_name: str, matched_sentences: List[str], language: str = "zh") -> Dict[str, Any]:
        """Round 2: 根据正则精准回溯原文语句进行判定 (严格适配系统语言)"""
        is_en = "en" in str(language).lower()
        cat_str = self._format_ontology_categories(language=language)
        sents_str = "\n".join([f"- {s}" for s in matched_sentences])

        if is_en:
            system_prompt = (
                "You are a senior biomedical researcher. The previous round could not determine this entity's type due to weak context. Targeted regex search from the original paper full text retrieved the following sentences.\n"
                "Re-evaluate and classify this entity based on these genuine literature sentences. If evidence remains insufficient or ambiguous, you MUST set primary_type to \"Uncertain\"."
            )
            user_content = (
                f"【Entity to Classify】: {node_name}\n"
                f"【Sentences Retrieved from Original Paper】:\n{sents_str}\n\n"
                f"【Candidate Entity Categories】:\n{cat_str}\n\n"
                "【Output Instructions】:\n"
                "Strictly output a single valid JSON object. All field values (sub_type, tags, reasoning) MUST be in English:\n"
                "{\n"
                "  \"primary_type\": \"<Candidate Category ID or Uncertain>\",\n"
                "  \"sub_type\": \"<Specific subtype in English>\",\n"
                "  \"tags\": [\"<Tag1>\", \"<Tag2>\"],\n"
                "  \"confidence\": 0.85,\n"
                "  \"reasoning\": \"<Literature-grounded evidence & rationale in English, under 40 words>\"\n"
                "}"
            )
        else:
            system_prompt = (
                "你是一位资深的生物医学专家。前轮分析因上下文微弱无法断定该实体类型，现通过原文献正则精准检索到以下关键句子。\n"
                "请根据这些真实文献语句再次进行分类。若依然缺乏把握或证据不足，请将 primary_type 设为 \"Uncertain\"。"
            )
            user_content = (
                f"【待分析实体】: {node_name}\n"
                f"【从原文精准检索出的上下文语句】:\n{sents_str}\n\n"
                f"【候选实体类别集合】:\n{cat_str}\n\n"
                "【输出要求】:\n"
                "请严格以单条合法 JSON 格式输出，所有文本字段（sub_type, tags, reasoning）务必使用中文:\n"
                "{\n"
                "  \"primary_type\": \"<候选类别ID或Uncertain>\",\n"
                "  \"sub_type\": \"<细分子类别>\",\n"
                "  \"tags\": [\"<标签1>\", \"<标签2>\"],\n"
                "  \"confidence\": 0.85,\n"
                "  \"reasoning\": \"<原文支持证据与理由，50字以内>\"\n"
                "}"
            )

        try:
            raw = self.llm_agent._ask_llm(system_prompt, user_content)
            res = self._parse_llm_json(raw)
            if isinstance(res, dict) and "primary_type" in res:
                return res
        except Exception as e:
            print(f"⚠️ [Round 2 Entity LLM Error] {node_name}: {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "Context in paper is insufficient to confirm category" if is_en else "文献上下文证据不足以明确分类"
        }

    def _llm_classify_entity_round3(self, node_name: str, language: str = "zh") -> Dict[str, Any]:
        """Round 3: 兜底通用生物知识判定 (严格适配系统语言)"""
        is_en = "en" in str(language).lower()
        cat_str = self._format_ontology_categories(language=language)

        if is_en:
            system_prompt = (
                "You are a senior bioinformatics expert. The entity has limited context in the literature.\n"
                "Adjudicate its category using your biomedical domain world knowledge. If confident, assign the proper candidate category; if it is an internal project code or unidentifiable, honestly output \"Uncertain\"."
            )
            user_content = (
                f"【Biomedical Entity Name】: {node_name}\n\n"
                f"【Candidate Entity Categories】:\n{cat_str}\n\n"
                "【Output Instructions】:\n"
                "Strictly output a single valid JSON object in English:\n"
                "{\n"
                "  \"primary_type\": \"<Candidate Category ID or Uncertain>\",\n"
                "  \"sub_type\": \"<Specific subtype in English>\",\n"
                "  \"tags\": [\"<Tag1>\"],\n"
                "  \"confidence\": 0.70,\n"
                "  \"reasoning\": \"<Domain knowledge reasoning in English, under 40 words>\"\n"
                "}"
            )
        else:
            system_prompt = (
                "你是一位资深的生物信息学专家。该实体在文献局部和原文语句中信息均不足。\n"
                "请仅根据你在通用生物医学领域的常识进行最终研判。如果你确信其属于某个类别，请输出对应类别；若该词汇为课题自命名代号或无法确认，请坦率输出 \"Uncertain\"。"
            )
            user_content = (
                f"【待分析生物医学实体名】: {node_name}\n\n"
                f"【候选实体类别集合】:\n{cat_str}\n\n"
                "【输出要求】:\n"
                "请严格以合法 JSON 输出，所有文本字段（sub_type, tags, reasoning）务必使用中文:\n"
                "{\n"
                "  \"primary_type\": \"<候选类别ID或Uncertain>\",\n"
                "  \"sub_type\": \"<细分子类别>\",\n"
                "  \"tags\": [\"<标签1>\"],\n"
                "  \"confidence\": 0.70,\n"
                "  \"reasoning\": \"<通用知识研判依据，50字以内>\"\n"
                "}"
            )

        try:
            raw = self.llm_agent._ask_llm(system_prompt, user_content)
            res = self._parse_llm_json(raw)
            if isinstance(res, dict) and "primary_type" in res:
                return res
        except Exception as e:
            print(f"⚠️ [Round 3 Entity LLM Error] {node_name}: {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "Features remain insufficient to classify after multi-stage analysis" if is_en else "经多级综合研判，实体特征仍不足以明确分类"
        }

    def _llm_classify_relation(self, source: str, target: str, orig_rel: str, evidence: str, language: str = "zh") -> Dict[str, Any]:
        """对单条关系进行深层机制分类与机制动词归纳 (严格适配系统语言)"""
        is_en = "en" in str(language).lower()
        rel_cat_str = self._format_relation_categories(language=language)
        src_type = self.entity_annotations.get(source, {}).get("primary_type", "Unknown" if is_en else "未知")
        tgt_type = self.entity_annotations.get(target, {}).get("primary_type", "Unknown" if is_en else "未知")

        if is_en:
            system_prompt = (
                "You are a molecular mechanism biologist. Your task is to classify biomedical knowledge graph relationships into deep mechanism categories "
                "and distill a precise, concise English mechanism verb (e.g., 'Phosphorylates', 'Competitively inhibits', 'Transcriptional activation', 'Targeted degradation', 'Binds', 'Infects')."
            )
            user_content = (
                f"【Subject Entity】: {source} (Classified as: {src_type})\n"
                f"【Object Entity】: {target} (Classified as: {tgt_type})\n"
                f"【Original Relation Predicate】: {orig_rel}\n"
                f"【Original Literature Evidence】: {evidence if evidence else 'No direct sentence available'}\n\n"
                f"【Candidate Mechanism Categories】:\n{rel_cat_str}\n\n"
                "【Output Instructions】:\n"
                "Strictly output a single valid JSON object. All field values (mechanism_verb, reasoning) MUST be in English:\n"
                "{\n"
                "  \"mechanism_category\": \"<Exact Candidate Category ID>\",\n"
                "  \"mechanism_verb\": \"<Precise concise English mechanism verb, 1-3 words>\",\n"
                "  \"confidence\": 0.90,\n"
                "  \"reasoning\": \"<Concise mechanism rationale in English, under 40 words>\"\n"
                "}"
            )
        else:
            system_prompt = (
                "你是一位分子机制生物学家。你的任务是对生物知识图谱中的关系进行深层机制分类，"
                "并归纳出一个精准精炼的机制动词谓词（如：催化磷酸化、竞争性结合、转录激活、靶向降解、受体识别、感染宿主等）。"
            )
            user_content = (
                f"【主体实体】: {source} (已分类为: {src_type})\n"
                f"【客体实体】: {target} (已分类为: {tgt_type})\n"
                f"【原始关系谓词】: {orig_rel}\n"
                f"【文献原文证据】: {evidence if evidence else '无具体原文句子'}\n\n"
                f"【候选关系机制类别】:\n{rel_cat_str}\n\n"
                "【输出要求】:\n"
                "请严格以合法 JSON 输出，不要包含任何多余文字，mechanism_verb 与 reasoning 务必使用中文:\n"
                "{\n"
                "  \"mechanism_category\": \"<候选类别ID>\",\n"
                "  \"mechanism_verb\": \"<精准精炼的机制动词，10字以内>\",\n"
                "  \"confidence\": 0.90,\n"
                "  \"reasoning\": \"<简明归类依据，50字以内>\"\n"
                "}"
            )

        try:
            raw = self.llm_agent._ask_llm(system_prompt, user_content)
            res = self._parse_llm_json(raw)
            if isinstance(res, dict) and "mechanism_category" in res:
                return res
        except Exception as e:
            print(f"⚠️ [Relation LLM Error] {source} -> {target}: {e}")

        return {
            "mechanism_category": "Association_Correlation",
            "mechanism_verb": orig_rel,
            "confidence": 0.5,
            "reasoning": "Analysis defaulted to general correlation" if is_en else "分析降级至默认关联"
        }

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

            # === Round 0: 快速规则匹配 (0 Token, 极速预过滤) ===
            r0_hits = 0
            if enable_round0_rules:
                log("⚡ [Stage 1: Nomenclature Recognition] Screening biomedical nomenclature & affixes..." if is_en else "⚡ [阶段一: 命名规范识别] 正在依据生物医学命名规范与权威词缀高速解析...")
                for idx, node in enumerate(sorted_nodes):
                    if not force_reanalyze and node in self.entity_annotations:
                        cur = self.entity_annotations[node]
                        if cur.get("status") == "resolved":
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
                            valid_cats = self.ontology.get("entity_categories", {})
                            if cat and cat != "Uncertain" and (not valid_cats or cat in valid_cats):
                                self.entity_annotations[node] = {
                                    **res,
                                    "round": 1,
                                    "status": "resolved"
                                }
                                self.stats["round1_hits"] += 1
                            else:
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
                                valid_cats = self.ontology.get("entity_categories", {})
                                if cat and cat != "Uncertain" and (not valid_cats or cat in valid_cats):
                                    self.entity_annotations[node] = {
                                        **res,
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
                            valid_cats = self.ontology.get("entity_categories", {})
                            if cat and cat != "Uncertain" and (not valid_cats or cat in valid_cats):
                                self.entity_annotations[node] = {
                                    **res,
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
            for rel in relations:
                src = rel.get("source")
                tgt = rel.get("target")
                orig_rel = rel.get("relation", "相关")
                edge_key = f"{src}|{tgt}|{orig_rel}"
                if not force_reanalyze and edge_key in self.relation_annotations:
                    cur = self.relation_annotations[edge_key]
                    if cur.get("status") == "resolved":
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
                        self.relation_annotations[edge_key] = {
                            **res,
                            "round": 1,
                            "status": "resolved"
                        }
                        self.stats["analyzed_relations"] += 1

            log(f"✅ [Relation Distillation Complete] Successfully distilled {self.stats['analyzed_relations']} mechanism verbs and categories." if is_en else f"✅ [关系机制分析完成] 成功提炼 {self.stats['analyzed_relations']} 条深层机制动词与类别。")

        # ----------------------------------------------------
        # 阶段 C: 统计与指标计算
        # ----------------------------------------------------
        resolved_cnt = sum(1 for a in self.entity_annotations.values() if a.get("status") == "resolved")
        uncertain_cnt = sum(1 for a in self.entity_annotations.values() if a.get("status") == "uncertain" or a.get("primary_type") == "Uncertain")
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
    工作台第二阶段：信号/关系通路拆解与建模引擎 (Pathway Decomposition Engine)
    功能：
    1. 候选节点的多维综合评分（拓扑中心度 + 类别柔性加权 + 意图语义相关度）
    2. 自由组合与最小连通骨干子网络诱导（Backbone Subgraphs）
    3. 有向闭环（Directed Simple Cycles）与自调节反馈极性代数判定（自促进/正反馈 vs 自抑制/负反馈）
    4. 跨文献跳跃桥接检测与新假说发现（Cross-Paper Pivot & Synthetic Bridges）
    5. 大模型驱动的生物学因果建模与结构化机制叙事
    6. 专属工程子目录 (projects/<pid>/pathways/) 落盘与一键独立导出
    """

    def __init__(self,
                 llm_agent=None,
                 ontology: Optional[Dict[str, Any]] = None,
                 entity_annotations: Optional[Dict[str, Any]] = None,
                 relation_annotations: Optional[Dict[str, Any]] = None,
                 project_dir: Optional[str] = None):
        self.llm_agent = llm_agent
        self.ontology = ontology or {}
        self.entity_annotations = entity_annotations or {}
        self.relation_annotations = relation_annotations or {}
        self.project_dir = project_dir
        self.pathways_dir = os.path.join(project_dir, "pathways") if project_dir else None

    def score_candidate_seeds(self,
                              entities: List[Dict[str, Any]],
                              relations: List[Dict[str, Any]],
                              user_query: str = "",
                              max_seeds: int = 15) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        多维综合评分候选节点发现 (Multi-Criteria Node Scoring)
        得分 = 拓扑度数中心度 + 拓扑介数中心度 + 生物学类别先验倾向 + 用户输入语义相关度
        """
        if not relations:
            return []

        G = nx.DiGraph()
        for r in relations:
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "").strip())
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

        q_terms = [w.lower() for w in re.split(r'[\s,，、;；。.!！?？]+', user_query) if len(w) >= 2]
        query_lower = user_query.strip().lower()

        scores = []
        for n in G.nodes():
            # 1. 拓扑得分 (归一化到 0~3)
            deg_val = deg_dict.get(n, 0)
            deg_norm = (deg_val / max_deg) * 2.5
            bet_norm = betweenness_dict.get(n, 0.0) * 2.5

            # 2. 类别柔性加权 (Soft Prior)
            cat_annot = self.entity_annotations.get(n, {})
            primary_type = str(cat_annot.get("primary_type", "")).lower()
            sub_type = str(cat_annot.get("sub_type", "")).lower()
            combined_type = f"{primary_type} {sub_type}"

            cat_weight = 0.0
            if any(k in combined_type for k in ["疾病", "disease", "表型", "phenotype", "pathology", "cancer", "tumor", "syndrome", "死", "凋亡", "自噬"]):
                cat_weight += 2.0  # 天然终点/效应
            elif any(k in combined_type for k in ["药物", "drug", "化学", "chemical", "small molecule", "compound", "inhibitor", "agonist"]):
                cat_weight += 1.8  # 天然起点/刺激
            elif any(k in combined_type for k in ["激酶", "kinase", "受体", "receptor", "转录因子", "transcription factor"]):
                cat_weight += 1.6  # 核心中继
            elif any(k in combined_type for k in ["基因", "gene", "蛋白", "protein"]):
                cat_weight += 1.2
            elif any(k in combined_type for k in ["过程", "process", "通路", "pathway"]):
                cat_weight += 1.4

            # 3. 语义意图匹配提权
            query_boost = 0.0
            if query_lower:
                n_lower = n.lower()
                if n_lower in query_lower or query_lower in n_lower:
                    query_boost += 3.5
                elif any(term in n_lower for term in q_terms):
                    query_boost += 2.0

            total_score = deg_norm + bet_norm + cat_weight + query_boost
            meta = {
                "degree": deg_val,
                "betweenness": betweenness_dict.get(n, 0.0),
                "primary_type": cat_annot.get("primary_type", "Uncertain"),
                "cat_weight": cat_weight,
                "query_boost": query_boost
            }
            scores.append((n, total_score, meta))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:max_seeds]

    def cluster_and_induce_subgraphs(self,
                                      relations: List[Dict[str, Any]],
                                      seeds: List[str],
                                      max_pathways: int = 3) -> List[Dict[str, Any]]:
        """
        自由组合与骨干子网诱导 (Backbone Subgraph Induction)
        提取各核心种子组合之间的最小连通骨干网
        """
        if not relations or not seeds:
            return []

        G = nx.DiGraph()
        for r in relations:
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "")).strip()
            if s and t and s != t:
                G.add_edge(s, t, rel_data=r)

        G_undir = G.to_undirected()

        # 若种子数量较少，直接构建单一整体通路
        if len(seeds) <= 6:
            seed_clusters = [seeds]
        else:
            seed_clusters = []
            remaining_seeds = list(seeds)
            target_num_clusters = min(max_pathways, max(1, len(seeds) // 4))

            while remaining_seeds and len(seed_clusters) < target_num_clusters:
                center = remaining_seeds.pop(0)
                cluster = [center]
                to_remove = []
                for s in remaining_seeds:
                    try:
                        if nx.has_path(G_undir, center, s):
                            dist = nx.shortest_path_length(G_undir, center, s)
                            if dist <= 2 and len(cluster) < 5:
                                cluster.append(s)
                                to_remove.append(s)
                    except Exception:
                        pass
                for s in to_remove:
                    remaining_seeds.remove(s)
                seed_clusters.append(cluster)

            for s in remaining_seeds:
                best_c = None
                best_d = 999
                for c in seed_clusters:
                    for c_node in c:
                        try:
                            if nx.has_path(G_undir, c_node, s):
                                d = nx.shortest_path_length(G_undir, c_node, s)
                                if d < best_d:
                                    best_d = d
                                    best_c = c
                        except Exception:
                            pass
                if best_c is not None and len(best_c) < 7:
                    best_c.append(s)

        subgraphs = []
        for cluster_idx, cluster in enumerate(seed_clusters):
            sub_nodes = set(cluster)
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    u = cluster[i]
                    v = cluster[j]
                    try:
                        if nx.has_path(G_undir, u, v):
                            p = nx.shortest_path(G_undir, u, v)
                            if len(p) <= 4:
                                sub_nodes.update(p)
                    except Exception:
                        pass

            if len(sub_nodes) > 10:
                deg_in_g = dict(G.degree())
                sorted_nodes = sorted(list(sub_nodes), key=lambda x: (x in cluster, deg_in_g.get(x, 0)), reverse=True)
                sub_nodes = set(sorted_nodes[:10])

            sub_relations = []
            for r in relations:
                src = str(r.get("source", "")).strip()
                tgt = str(r.get("target", "")).strip()
                if src in sub_nodes and tgt in sub_nodes and src != tgt:
                    sub_relations.append(r)

            if len(sub_nodes) >= 2 and len(sub_relations) >= 1:
                subgraphs.append({
                    "cluster_id": cluster_idx + 1,
                    "seed_nodes": cluster,
                    "nodes": sorted(list(sub_nodes)),
                    "relations": sub_relations
                })

        return subgraphs[:max_pathways]

    def detect_feedback_loops(self,
                              subgraph_nodes: List[str],
                              subgraph_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        有向闭环与自调节反馈极性代数判定 (Feedback Polarity Algebra)
        - 查找有向简单环路 (Cycle Length 2~5)
        - 统计抑制/负调控边数量，偶数条 -> 自促进/正反馈；奇数条 -> 自抑制/负反馈
        """
        sub_G = nx.DiGraph()
        rel_lookup = {}
        for r in subgraph_relations:
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "")).strip()
            if s and t and s != t:
                sub_G.add_edge(s, t)
                rel_lookup[(s, t)] = r

        try:
            raw_cycles = list(nx.simple_cycles(sub_G))
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
            neg_count = 0
            cycle_steps = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge_rel = rel_lookup.get((u, v), {})
                orig_rel = str(edge_rel.get("relation", edge_rel.get("relation_type", "")))

                edge_k = f"{u}|{v}|{orig_rel}"
                annot = self.relation_annotations.get(edge_k, {})
                mech_cat = str(annot.get("mechanism_category", "")).lower()
                mech_verb = str(annot.get("mechanism_verb", orig_rel)).lower()

                is_neg = (orig_rel == "负作用") or any(kw in (mech_cat + " " + mech_verb) for kw in [
                    "negative", "inhibit", "degrad", "suppress", "block", "repress",
                    "downregulat", "负", "抑制", "降解", "阻断", "下调", "去磷酸化", "衰减"
                ])
                if is_neg:
                    neg_count += 1
                cycle_steps.append((u, v, annot.get("mechanism_verb", orig_rel), is_neg))

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
        跨文献跳跃桥接检测 (Cross-Paper Bridge Tagging)
        识别连接不同文献来源的桥接中枢节点与跨文献假说
        """
        paper_sources = set()
        node_in_sources = {n: set() for n in subgraph_nodes}
        node_out_sources = {n: set() for n in subgraph_nodes}

        for r in subgraph_relations:
            src = str(r.get("source", "")).strip()
            tgt = str(r.get("target", "")).strip()
            raw_doc = str(r.get("doc_source", "")).strip()
            if raw_doc and raw_doc not in ["未知文献", "未知", "Unknown", "unknown", "None"]:
                clean_doc = os.path.basename(raw_doc)
                paper_sources.add(clean_doc)
                if src in node_out_sources:
                    node_out_sources[src].add(clean_doc)
                if tgt in node_in_sources:
                    node_in_sources[tgt].add(clean_doc)

        pivot_nodes = []
        for n in subgraph_nodes:
            in_s = node_in_sources.get(n, set())
            out_s = node_out_sources.get(n, set())
            if in_s and out_s and (in_s != out_s or len(in_s.union(out_s)) >= 2):
                pivot_nodes.append({
                    "node": n,
                    "in_papers": sorted(list(in_s)),
                    "out_papers": sorted(list(out_s))
                })

        return {
            "is_cross_paper": len(paper_sources) >= 2,
            "paper_sources": sorted(list(paper_sources)),
            "pivot_nodes": pivot_nodes,
            "paper_count": len(paper_sources)
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
        大模型驱动的生物学因果建模与结构化机制叙事
        生成包含主轴流动、自调节闭环效应与跨文献假说的标准报告
        """
        is_en = "en" in str(language).lower()

        # 整理输入上下文
        nodes_desc = []
        for n in subgraph_nodes:
            ann = self.entity_annotations.get(n, {})
            ptype = ann.get("primary_type", "Uncertain")
            stype = ann.get("sub_type", "")
            type_str = f"[{ptype}]" if ptype != "Uncertain" else ""
            if stype:
                type_str += f" ({stype})"
            nodes_desc.append(f"- {n} {type_str}")

        edges_desc = []
        for r in subgraph_relations:
            s = r.get("source")
            t = r.get("target")
            rel = r.get("relation", r.get("relation_type", "相关"))
            edge_k = f"{s}|{t}|{rel}"
            ann = self.relation_annotations.get(edge_k, {})
            verb = ann.get("mechanism_verb", rel)
            evi = r.get("evidence", r.get("reason", "无"))
            doc = os.path.basename(str(r.get("doc_source", "未知文献")))
            edges_desc.append(f"- {s} ➔ {t} : `{verb}` (原文依据: {evi[:80]}; 来源: {doc})")

        loops_desc = []
        for lp in feedback_loops:
            ltype = "🔴 自抑制负反馈回路 (Negative Feedback Loop)" if lp.get("is_self_inhibition") else "🟢 自促进正反馈回路 (Positive Feedback Loop)"
            loops_desc.append(f"- 闭环: {lp.get('cycle_str')} [{ltype}]")

        bridges_desc = []
        if cross_paper_info.get("is_cross_paper"):
            pivots = [p["node"] for p in cross_paper_info.get("pivot_nodes", [])]
            papers = cross_paper_info.get("paper_sources", [])
            bridges_desc.append(f"- 跨越文献: {', '.join(papers)}")
            bridges_desc.append(f"- 桥接中枢: {', '.join(pivots) if pivots else '局部连线拼接'}")

        # 组装 Prompt
        if is_en:
            sys_prompt = (
                "You are an expert in systems biology, signal transduction networks, and molecular medicine. "
                "Your task is to analyze a biological subnetwork module and synthesize a rigorous causal signaling pathway model. "
                "Highlight the core causal flow, feedback loops (self-inhibition for homeostasis, self-promotion for amplification), "
                "and cross-literature synthesis insights."
            )
            user_prompt = (
                f"【Pathway Module #{pathway_idx} Context】\n"
                f"【User Exploration Intent】: {user_query if user_query else 'Global autonomous discovery'}\n"
                f"【Key Entities in Subnetwork】:\n" + "\n".join(nodes_desc) + "\n\n"
                f"【Direct Mechanism Interactions】:\n" + "\n".join(edges_desc) + "\n\n"
                f"【Detected Feedback Loops】:\n" + ("\n".join(loops_desc) if loops_desc else "None (Linear cascade)") + "\n\n"
                f"【Cross-Literature Evidence】:\n" + ("\n".join(bridges_desc) if bridges_desc else "Intra-paper pathway") + "\n\n"
                "【Output Requirement】:\n"
                "Strictly return a single valid JSON object. All field values MUST be in English. No markdown outside the JSON:\n"
                "{\n"
                "  \"title\": \"<Concise scientific pathway title under 20 words>\",\n"
                "  \"main_causal_axis\": \"<Linear arrow flow, e.g. Trigger ➔ Mediator ➔ (Inhibit) Target ➔ Phenotype>\",\n"
                "  \"executive_summary\": \"<3-5 concise sentences explaining the molecular cascade and biological significance>\",\n"
                "  \"feedback_loop_explanation\": \"<Detailed analysis of self-regulation loops, or state linear cascade if none>\",\n"
                "  \"cross_paper_insight\": \"<Analysis of how combined papers reveal novel insights, or state single-source consistency>\"\n"
                "}"
            )
        else:
            sys_prompt = (
                "你是一位资深的系统生物学、信号传导网络与分子医学专家。你的任务是基于给定的局部知识子图，"
                "对其进行深度因果建模，提炼出严谨的信号通路机制报告。特别注意阐明核心因果主轴、"
                "自调节闭环（自抑制稳态机制 vs 自促进放大开关）以及跨文献综合提出的全新科学假说。"
            )
            user_prompt = (
                f"【第 {pathway_idx} 条通路子网络数据】\n"
                f"【学者探索意图/焦点】: {user_query if user_query else '全局自主发掘'}\n"
                f"【子网络关键实体】:\n" + "\n".join(nodes_desc) + "\n\n"
                f"【分子机制交互关系】:\n" + "\n".join(edges_desc) + "\n\n"
                f"【算法捕获的自调节闭环回路】:\n" + ("\n".join(loops_desc) if loops_desc else "无闭环 (呈单向线性级联)") + "\n\n"
                f"【跨文献证据分布】:\n" + ("\n".join(bridges_desc) if bridges_desc else "单篇文献内部通路") + "\n\n"
                "【输出要求】:\n"
                "请严格输出唯一的合法 JSON 对象，字段内容全部使用中文，严禁在 JSON 外输出额外 Markdown：\n"
                "{\n"
                "  \"title\": \"<精炼的学术通路标题，20字以内，如：AMPK-mTOR-ULK1 自噬调控轴与自抑回路>\",\n"
                "  \"main_causal_axis\": \"<清晰的箭头流向主轴，如：二甲双胍 ➔ AMPK ➔ (抑制) mTOR ➔ ULK1 ➔ 细胞自噬>\",\n"
                "  \"executive_summary\": \"<3~5句话精炼阐释该通路的分子传导逻辑与生理/病理效应>\",\n"
                "  \"feedback_loop_explanation\": \"<详细剖析自抑制或自促进回路的调控生物学意义；若无回路则说明其单向传导特征>\",\n"
                "  \"cross_paper_insight\": \"<深入阐述多篇文献拼接揭示的新机制或协同新假说；若单篇则说明证据一致性>\"\n"
                "}"
            )

        # 调用大模型
        if self.llm_agent:
            try:
                raw_resp = self.llm_agent._ask_llm(sys_prompt, user_prompt)
                parsed = self._parse_llm_json(raw_resp)
                if parsed and isinstance(parsed, dict) and "title" in parsed:
                    return parsed
            except Exception as e:
                print(f"⚠️ [PathwayDecompositionEngine] LLM 机制叙事调用异常: {e}")

        # 兜底生成逻辑 (无 LLM 或 LLM 失败时的纯拓扑保底)
        first_node = subgraph_nodes[0] if subgraph_nodes else "NodeA"
        last_node = subgraph_nodes[-1] if len(subgraph_nodes) > 1 else first_node
        default_title = f"{first_node} ➔ {last_node} 信号调控轴" if not is_en else f"{first_node} to {last_node} Signaling Axis"
        default_axis = " ➔ ".join(subgraph_nodes[:6])
        default_summary = (
            f"本通路整合了 {len(subgraph_nodes)} 个关键节点与 {len(subgraph_relations)} 条机制连线，展现了以 {first_node} 为核心向 {last_node} 的信号级联流动。"
            if not is_en else
            f"This pathway integrates {len(subgraph_nodes)} key nodes and {len(subgraph_relations)} interactions, delineating a cascade from {first_node} to {last_node}."
        )
        default_loops_exp = (
            f"检测到 {len(feedback_loops)} 个自反馈调节闭环回路。" if feedback_loops else "本通路呈单向线性级联传导，未检测到闭合反馈回路。"
            if not is_en else
            f"Detected {len(feedback_loops)} feedback loops." if feedback_loops else "Unidirectional cascade without closed feedback loop."
        )
        default_cross_exp = (
            f"跨越了 {cross_paper_info.get('paper_count', 1)} 篇文献进行机制综合。"
            if not is_en else
            f"Synthesizes evidence across {cross_paper_info.get('paper_count', 1)} literature sources."
        )

        return {
            "title": default_title,
            "main_causal_axis": default_axis,
            "executive_summary": default_summary,
            "feedback_loop_explanation": default_loops_exp,
            "cross_paper_insight": default_cross_exp
        }

    def run_decomposition(self,
                          entities: List[Dict[str, Any]],
                          relations: List[Dict[str, Any]],
                          user_query: str = "",
                          enable_loops: bool = True,
                          enable_bridges: bool = True,
                          enable_subgraphs: bool = True,
                          max_pathways: int = 3,
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

        # 1. 种子节点打分与挑选
        prog(0.15, "正在计算多维节点重要性得分..." if not is_en else "Computing multi-criteria node importance scores...")
        seed_scores = self.score_candidate_seeds(entities, relations, user_query=user_query, max_seeds=18)
        seed_names = [s[0] for s in seed_scores]
        log(f"🎯 成功挑选出 {len(seed_names)} 个核心候选种子实体: {', '.join(seed_names[:8])}..." if not is_en else f"🎯 Identified {len(seed_names)} candidate seed entities: {', '.join(seed_names[:8])}...")

        # 2. 自由组合与骨干子网诱导
        prog(0.30, "正在自由组合聚类与提取骨干子网络..." if not is_en else "Clustering seeds and inducing backbone subgraphs...")
        subgraphs = self.cluster_and_induce_subgraphs(relations, seed_names, max_pathways=max_pathways)
        log(f"📦 成功聚类诱导出 {len(subgraphs)} 个高密度通路子网络模块" if not is_en else f"📦 Successfully induced {len(subgraphs)} dense pathway subnetwork modules")

        if not subgraphs:
            log("⚠️ 未能提取出足够规模的连通子网络。" if not is_en else "⚠️ Unable to induce connected subgraphs.")
            prog(1.0, "分析完成" if not is_en else "Completed")
            return []

        # 3. 逐一深入拆解各通路
        results = []
        total_sub = len(subgraphs)

        for idx, sub in enumerate(subgraphs):
            cid = sub["cluster_id"]
            nodes = sub["nodes"]
            rels = sub["relations"]

            step_base = 0.35 + (0.55 * (idx / total_sub))
            prog(step_base, f"正在深度建模第 {idx+1}/{total_sub} 条通路: 包含 {len(nodes)} 节点, {len(rels)} 连线..." if not is_en else f"Modeling pathway {idx+1}/{total_sub} ({len(nodes)} nodes, {len(rels)} edges)...")

            # A. 闭环与反馈检测
            loops = []
            if enable_loops:
                loops = self.detect_feedback_loops(nodes, rels)
                if loops:
                    neg_l = [l for l in loops if l["is_self_inhibition"]]
                    pos_l = [l for l in loops if l["is_self_promotion"]]
                    log(f"  通路 {cid}: 捕获 {len(loops)} 个反馈闭环 (自抑制: {len(neg_l)}, 自促进: {len(pos_l)})" if not is_en else f"  Pathway {cid}: Captured {len(loops)} loops (Inhibition: {len(neg_l)}, Promotion: {len(pos_l)})")

            # B. 跨文献跳跃桥接检测
            bridges = {"is_cross_paper": False, "paper_sources": [], "pivot_nodes": [], "paper_count": 1}
            if enable_bridges:
                bridges = self.detect_cross_paper_bridges(nodes, rels)
                if bridges.get("is_cross_paper"):
                    log(f"  通路 {cid}: 识别为跨文献融合通路 (跨越 {bridges.get('paper_count')} 篇文献, 桥接点: {len(bridges.get('pivot_nodes', []))} 个)" if not is_en else f"  Pathway {cid}: Identified cross-paper synthesis ({bridges.get('paper_count')} papers)")

            # C. 大模型机制叙事
            prog(step_base + 0.1, f"正在调用大模型生成第 {idx+1} 条通路的因果机制报告..." if not is_en else f"Invoking LLM narrative for pathway {idx+1}...")
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

                    # 提取子图涉及的完整实体字典信息
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
                        height="480px"
                    )
                except Exception as e:
                    log(f"⚠️ 生成子图 HTML 遇到小异常: {e}")

            # E. 组装证据溯源列表
            evidence_list = []
            for r in rels:
                src = r.get("source")
                tgt = r.get("target")
                rel = r.get("relation", r.get("relation_type", "相关"))
                edge_k = f"{src}|{tgt}|{rel}"
                ann = self.relation_annotations.get(edge_k, {})
                verb = ann.get("mechanism_verb", rel)
                evidence_list.append({
                    "source": src,
                    "target": tgt,
                    "verb": verb,
                    "original_relation": rel,
                    "evidence": r.get("evidence", r.get("reason", "无")),
                    "doc_source": os.path.basename(str(r.get("doc_source", "未知文献")))
                })

            pathway_obj = {
                "id": f"pw_{cid}",
                "cluster_id": cid,
                "title": narrative.get("title", f"通路 {cid}" if not is_en else f"Pathway {cid}"),
                "main_causal_axis": narrative.get("main_causal_axis", " ➔ ".join(nodes[:5])),
                "executive_summary": narrative.get("executive_summary", ""),
                "feedback_loop_explanation": narrative.get("feedback_loop_explanation", ""),
                "cross_paper_insight": narrative.get("cross_paper_insight", ""),
                "nodes": nodes,
                "relations": rels,
                "feedback_loops": loops,
                "cross_paper_bridges": bridges,
                "evidence_list": evidence_list,
                "html_file": html_rel_path,
                "has_negative_loop": any(l.get("is_self_inhibition") for l in loops),
                "has_positive_loop": any(l.get("is_self_promotion") for l in loops),
                "is_cross_paper": bridges.get("is_cross_paper", False)
            }
            results.append(pathway_obj)

        # 4. 全量落盘持久化
        prog(0.96, "正在持久化通路数据至项目专属目录..." if not is_en else "Persisting pathway data to project directory...")
        options = {
            "enable_loops": enable_loops,
            "enable_bridges": enable_bridges,
            "enable_subgraphs": enable_subgraphs,
            "max_pathways": max_pathways
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
