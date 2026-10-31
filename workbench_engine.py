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


class RulePreFilter:
    """
    完全基于本体标准配置文件 (ontology_presets/*.json) 中 rules 定义的数据驱动规则分类器。
    无任何写死在代码中的领域字典或硬编码规则，严格根据当前加载的标准进行确定性匹配。
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
    def classify(cls, node_name: str, aliases: Optional[List[str]] = None,
                 ontology: Optional[Dict[str, Any]] = None, language: str = "zh") -> Optional[Dict[str, Any]]:
        """
        根据当前本体预设中显式声明的 rules 列表进行确定性匹配。
        - 优先检查 exact_matches (哈希 O(1) 匹配)
        - 其次检查 patterns 正则模式
        - 若当前标准未定义 rules 或均未命中，安全返回 None，平滑交由后续大模型阶段处理，不进行任何臆测推导
        """
        if not ontology or not isinstance(ontology, dict):
            return None

        rules = ontology.get("rules", [])
        if not rules or not isinstance(rules, list):
            return None

        raw_name = node_name.strip()
        lower_name = raw_name.lower()
        all_candidates = [lower_name]
        if aliases:
            all_candidates.extend([a.strip().lower() for a in aliases if a.strip()])

        is_en = "en" in str(language).lower()
        target_categories = set(ontology.get("entity_categories", {}).keys())

        for rule in rules:
            if not isinstance(rule, dict):
                continue

            cat_id = rule.get("category")
            # 严格保障：非当前标准原本合法类别的规则一律跳过，确保 100% 对齐原本类别
            if target_categories and cat_id not in target_categories:
                continue

            sub_type = rule.get("sub_type", "")
            sub_type_zh = rule.get("sub_type_zh", sub_type)
            tag = rule.get("tag", cat_id)
            conf = float(rule.get("confidence", 0.95))
            reason_zh = rule.get("reason_zh", f"匹配【{cat_id}】命名规范规则")
            reason_en = rule.get("reason_en", f"Matched [{cat_id}] nomenclature rule")

            # 1. 优先极速哈希匹配 exact_matches (O(1))
            exact_matches = rule.get("exact_matches", [])
            if exact_matches:
                exact_set = {str(w).strip().lower() for w in exact_matches if str(w).strip()}
                for cand in all_candidates:
                    if cand in exact_set or raw_name in exact_matches:
                        return {
                            "primary_type": cat_id,
                            "sub_type": sub_type if is_en else sub_type_zh,
                            "tags": [tag],
                            "confidence": conf,
                            "reasoning": reason_en if is_en else reason_zh,
                            "round": 0,
                            "status": "resolved"
                        }

            # 2. 正则模式匹配 patterns
            patterns = rule.get("patterns", [])
            if patterns:
                for pat_str in patterns:
                    compiled = cls._get_compiled_regex(pat_str)
                    if not compiled:
                        continue
                    for cand in all_candidates:
                        if compiled.match(cand):
                            match_reason_zh = f"{reason_zh} (模式: {pat_str})"
                            match_reason_en = f"{reason_en} (Pattern: {pat_str})"
                            return {
                                "primary_type": cat_id,
                                "sub_type": sub_type if is_en else sub_type_zh,
                                "tags": [tag],
                                "confidence": conf,
                                "reasoning": match_reason_en if is_en else match_reason_zh,
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
                     log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
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
