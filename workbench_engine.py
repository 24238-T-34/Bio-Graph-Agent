"""
workbench_engine.py - 图谱分析工作台核心引擎
包含：
1. RulePreFilter: 基于生物医学命名法与权威词缀的高速 0-Token 正则预过滤分流器
2. OntologyManager: 本体分类预设配置加载、管理与保存系统
3. WorkbenchAnalysisEngine: 分级升级状态机（Round 0-3）、热度优先级调度、微上下文组装、精准文献回溯、通用生物知识兜底及人工专家覆写
"""

import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Callable


class RulePreFilter:
    """基于生物医学命名规则与权威词缀的高速 0-Token 正则预分类器"""

    # 1. 药物与化学分子词缀 (INN 权威国际非专利药品名称规范)
    INN_DRUG_PATTERNS = [
        r'.*tinib$',     # 酪氨酸激酶抑制剂 (Imatinib, Gefitinib, Erlotinib)
        r'.*rafenib$',   # RAF激酶抑制剂 (Sorafenib, Vemurafenib)
        r'.*lisib$',     # PI3K抑制剂 (Alpelisib, Idelalisib)
        r'.*ciclib$',    # CDK抑制剂 (Palbociclib, Ribociclib)
        r'.*parib$',     # PARP抑制剂 (Olaparib, Rucaparib)
        r'.*zomib$',     # 蛋白酶体抑制剂 (Bortezomib, Carfilzomib)
        r'.*nib$',       # 通用激酶抑制剂 (Dasatinib, Sunatinib, Sunitinib)
        r'.*mab$',       # 单克隆抗体 (Trastuzumab, Pembrolizumab, Rituximab)
        r'.*cept$',      # 受体融合蛋白 (Etanercept, Aflibercept)
        r'.*statin$',    # 他汀类降脂药 (Simvastatin, Atorvastatin, Rosuvastatin)
        r'.*stat$',      # 酶抑制剂 (Cilastatin, Ubenimex)
        r'.*prazole$',   # 质子泵抑制剂 (Omeprazole, Pantoprazole)
        r'.*mycin$',     # 抗生素 (Streptomycin, Rapamycin)
        r'.*sartan$',    # ARB受体阻滞剂 (Losartan, Valsartan)
        r'.*dipine$',    # 钙通道阻滞剂 (Amlodipine, Nifedipine)
        r'.*afil$',      # PDE5抑制剂 (Sildenafil, Tadalafil)
        r'.*gliflozin$', # SGLT2抑制剂 (Empagliflozin, Dapagliflozin)
        r'.*glitazone$', # PPAR-γ激动剂 (Pioglitazone, Rosiglitazone)
        r'.*oxacin$',    # 喹诺酮类抗生素 (Ciprofloxacin, Levofloxacin)
        r'.*vir$',       # 抗病毒药 (Acyclovir, Remdesivir)
        r'.*vudine$',    # 核苷逆转录酶抑制剂 (Lamivudine, Zidovudine)
    ]

    CHEM_COMPOUND_PATTERNS = [
        r'.*acid$', r'.*酸$',
        r'.*ester$', r'.*酯$',
        r'.*phosphate$', r'.*磷酸盐$',
        r'.*sulfate$', r'.*硫酸盐$',
        r'.*chloride$', r'.*氯化物$',
        r'.*acetate$', r'.*乙酸盐$',
        r'.*oxide$', r'.*氧化物$',
        r'.*inhibitor$', r'.*抑制剂$',
        r'.*agonist$', r'.*激动剂$',
        r'.*antagonist$', r'.*拮抗剂$'
    ]

    METABOLITES_AND_IONS = {
        'atp', 'adp', 'amp', 'gtp', 'gdp', 'gmp', 'ctp', 'utp', 'camp', 'cgmp',
        'nadh', 'nad+', 'nadph', 'nadp+', 'fadh2', 'fad',
        'ros', 'no', 'ca2+', 'mg2+', 'k+', 'na+', 'fe2+', 'fe3+', 'zn2+',
        'glucose', 'glutamine', 'lactate', 'pyruvate', 'citrate', 'cholesterol',
        'lipid', 'fatty acid', 'glutathione', 'gsh', 'gssg', 'acetyl-coa'
    }

    # 2. 酶与蛋白质词缀
    ENZYME_PATTERNS = [
        r'.*kinase$', r'.*激酶$',
        r'.*phosphatase$', r'.*磷酸酶$',
        r'.*synthase$', r'.*合成酶$',
        r'.*synthetase$', r'.*合酶$',
        r'.*transferase$', r'.*转移酶$',
        r'.*polymerase$', r'.*聚合酶$',
        r'.*ligase$', r'.*连接酶$',
        r'.*hydrolase$', r'.*水解酶$',
        r'.*peptidase$', r'.*肽酶$',
        r'.*protease$', r'.*蛋白酶$',
        r'.*dehydrogenase$', r'.*脱氢酶$',
        r'.*oxygenase$', r'.*加氧酶$',
        r'.*reductase$', r'.*还原酶$',
        r'.*dismutase$', r'.*歧化酶$',
        r'.*isomerase$', r'.*异构酶$',
        r'.*acetylase$', r'.*乙酰化酶$',
        r'.*deacetylase$', r'.*去乙酰化酶$',
        r'.*methylase$', r'.*甲基化酶$',
        r'.*demethylase$', r'.*去甲基化酶$',
        r'.*caspase[-0-9]*$',
        r'.*proteasome.*$', r'.*蛋白酶体.*$'
    ]

    RECEPTOR_PATTERNS = [
        r'.*receptor$', r'.*受体$',
        r'.*gpcr$',
        r'^[a-z0-9]+r[0-9]*$',   # e.g., egfr, tgfbr1, ifngr1
        r'^[a-z0-9]+rec$',
    ]

    RNA_PATTERNS = [
        r'^(?:mir|mirna|microrna|lncrna|circrna|sncrna|sirna|pirna)[\-_].*',
        r'.*[\-_](?:mir|mirna|microrna|lncrna|circrna|sirna)$'
    ]

    # 经典人类高频基因/蛋白模式 (3-7个大写字母+可选数字组合)
    GENE_SYMBOL_REGEX = re.compile(r'^[A-Z][A-Z0-9]{1,8}(?:-[A-Z0-9]+)?$')
    KNOWN_MAJOR_GENES = {
        'TP53', 'EGFR', 'KRAS', 'NRAS', 'HRAS', 'MYC', 'STAT3', 'STAT1', 'CDK4', 'CDK6',
        'BRCA1', 'BRCA2', 'VEGFA', 'VEGFR', 'IL6', 'TNF', 'AKT1', 'AKT2', 'MTOR', 'BRAF',
        'PIK3CA', 'PTEN', 'MAPK1', 'MAPK3', 'CTNNB1', 'HIF1A', 'CASP3', 'BCL2', 'BAX',
        'JUN', 'FOS', 'NFKB1', 'RELA', 'SMAD3', 'SMAD4', 'ERBB2', 'MET', 'ALK', 'RET',
        'ATM', 'ATR', 'CHEK1', 'CHEK2', 'NOTCH1', 'JAK1', 'JAK2', 'TYK2', 'SRC', 'ABL1',
        'MAPK', 'ERK', 'JNK', 'P38', 'WNT', 'TGFB', 'IFNG', 'PDCD1', 'CD274', 'CTLA4'
    }

    # 3. 细胞与组织模式
    CELL_PATTERNS = [
        r'.*(?:cell|cells)$', r'.*细胞$',
        r'.*(?:lymphocyte|monocyte|macrophage|neutrophil|fibroblast|platelet)$',
        r'.*(?:淋巴细胞|单核细胞|巨噬细胞|中性粒细胞|成纤维细胞|血小板)$',
        r'.*(?:t[- ]?cell|b[- ]?cell|nk[- ]?cell|dendritic cell)$',
        r'.*(?:t细胞|b细胞|nk细胞|树突状细胞)$',
        r'.*(?:endothelial|epithelial|neuron|astrocyte|microglia)$',
        r'.*(?:内皮细胞|上皮细胞|神经元|星形胶质细胞|小胶质细胞)$'
    ]

    CELL_LINES = {
        'hela', 'hek293', 'hek293t', 'mcf-7', 'mcf7', 'a549', 'jurkat', 'pc3', 'u2os',
        'k562', 'thp-1', 'thp1', 'raw264.7', 'cos-7', 'cho', 'nih3t3', 'hl-60', 'sk-br-3',
        'hct116', 'sw480', 'mdamb231', 'mda-mb-231', 'b16f10', '4t1', 'panc-1', 'hep-g2', 'hepg2'
    }

    TISSUE_PATTERNS = [
        r'.*(?:tissue|organ|liver|lung|kidney|heart|brain|spleen|pancreas|colon|stomach|bone marrow|blood|serum|plasma|cortex|stroma)$',
        r'.*(?:组织|器官|肝脏|肺|肾脏|心脏|脑|脾脏|胰腺|结肠|胃|骨髓|血液|血清|血浆|皮质|基质)$'
    ]

    # 4. 疾病与表型模式
    DISEASE_PATTERNS = [
        r'.*(?:oma|itis|pathy|osis|emia|syndrome|carcinoma|sarcoma|melanoma|leukemia|lymphoma|failure|disorder|infection)$',
        r'.*(?:瘤|炎|病|综合征|癌|肉瘤|白血病|淋巴瘤|衰竭|障碍|感染)$',
        r'.*(?:cancer|tumor|neoplasm|hypertension|diabetes|asthma|stroke|infarction|alzheimer|parkinson|fibrosis|cirrhosis|sepsis|metastasis)$',
        r'.*(?:癌症|肿瘤|高血压|糖尿病|哮喘|中风|心肌梗死|阿尔茨海默病|帕金森病|纤维化|肝硬化|脓毒症|转移)$'
    ]

    # 5. 生物学过程与通路模式 (包含 -ptosis 等程序性死亡模式)
    PROCESS_PATTERNS = [
        r'.*ptosis$',   # apoptosis, ferroptosis, pyroptosis, necroptosis
        r'.*(?:pathway|signaling|cascade|apoptosis|autophagy|proliferation|differentiation|migration|invasion|angiogenesis|metastasis|glycolysis|metabolism|transcription|translation|ferroptosis|pyroptosis|cell cycle|immune response)$',
        r'.*(?:通路|信号转导|级联反应|凋亡|自噬|增殖|分化|迁移|侵袭|血管生成|糖酵解|代谢|转录|翻译|铁死亡|焦亡|细胞周期|免疫反应)$'
    ]

    # 6. 突变位点与序列变异模式
    VARIANT_PATTERNS = [
        r'^rs\d+$',
        r'^[a-z]\d+[a-z]$',          # e.g., v600e, t790m, g12d
        r'^p\.[a-z]\d+[a-z]$',        # e.g., p.v600e
        r'.*(?:exon|intron)\s*\d+.*',
        r'.*(?:mutation|mutant|variant|polymorphism|snp|indel|duplication|deletion)$',
        r'.*(?:突变|变异|多态性|缺失|重复)$'
    ]

    @classmethod
    def classify(cls, node_name: str, aliases: Optional[List[str]] = None, ontology: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        尝试使用基于医学规则的快速分类
        :param node_name: 实体主名称
        :param aliases: 实体已知别名列表
        :param ontology: 当前选定的本体预设字典
        :return: 匹配成功返回标注字典，若规则无法确定返回 None
        """
        raw_name = node_name.strip()
        lower_name = raw_name.lower()
        all_candidates = [lower_name]
        if aliases:
            all_candidates.extend([a.strip().lower() for a in aliases if a.strip()])

        target_categories = set(ontology.get("entity_categories", {}).keys()) if ontology else set()

        def make_res(cat_id: str, sub_type: str, tag: str, conf: float, reason: str) -> Dict[str, Any]:
            # 兼容特定预设的映射转换
            final_cat = cat_id
            if target_categories:
                if final_cat not in target_categories:
                    # 尝试转换到子预设类别
                    if "Kinase_Phosphatase" in target_categories and "kinase" in sub_type.lower():
                        final_cat = "Kinase_Phosphatase"
                    elif "Small_Molecule_Drug" in target_categories and cat_id == "Chemical_Drug":
                        final_cat = "Small_Molecule_Drug"
                    elif "Druggable_Target" in target_categories and ("receptor" in sub_type.lower() or "kinase" in sub_type.lower()):
                        final_cat = "Druggable_Target"
                    elif "Immune_Cell_Subset" in target_categories and cat_id == "Cell_Tissue":
                        final_cat = "Immune_Cell_Subset"
                    elif "Disease_Indication" in target_categories and cat_id == "Disease_Phenotype":
                        final_cat = "Disease_Indication"
                    elif "Pathway_Module" in target_categories and cat_id == "Biological_Process":
                        final_cat = "Pathway_Module"
                    elif cat_id not in target_categories:
                        # 找不到对应类别，交由大模型分级决策
                        return None

            return {
                "primary_type": final_cat,
                "sub_type": sub_type,
                "tags": [tag],
                "confidence": conf,
                "reasoning": reason,
                "round": 0,
                "status": "resolved"
            }

        # 检查是否为变异位点
        for cand in all_candidates:
            for pat in cls.VARIANT_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Sequence_Variant", "Sequence Mutation", "Gene Variant", 0.95, f"匹配权威基因变异/突变命名规则: {pat}")

        # 检查是否为代谢物或无机离子
        for cand in all_candidates:
            if cand in cls.METABOLITES_AND_IONS:
                return make_res("Chemical_Drug", "Metabolite / Ion", "Biochemical Molecule", 0.96, "匹配权威内源性代谢小分子/离子词库")

        # 检查 INN 药物词缀
        for cand in all_candidates:
            for pat in cls.INN_DRUG_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Chemical_Drug", "Targeted / Synthetic Drug", "INN Pharmaceutical", 0.96, f"匹配 INN 国际权威药名词缀规则: {pat}")

        # 检查通用化合物词缀
        for cand in all_candidates:
            for pat in cls.CHEM_COMPOUND_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Chemical_Drug", "Chemical Compound", "Compound", 0.88, f"匹配化学分子词缀: {pat}")

        # 检查非编码 RNA
        for cand in all_candidates:
            for pat in cls.RNA_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Gene_Protein", "Non-coding RNA", "RNA", 0.95, f"匹配非编码 RNA 命名规则: {pat}")

        # 检查酶/激酶
        for cand in all_candidates:
            for pat in cls.ENZYME_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Gene_Protein", "Enzyme / Kinase", "Catalytic Protein", 0.92, f"匹配酶学催化蛋白质命名词缀: {pat}")

        # 检查受体
        for cand in all_candidates:
            for pat in cls.RECEPTOR_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Gene_Protein", "Receptor Protein", "Receptor", 0.90, f"匹配生物膜/细胞受体命名规则: {pat}")

        # 检查标准知名基因符号 (TP53, EGFR 等)
        if raw_name in cls.KNOWN_MAJOR_GENES:
            return make_res("Gene_Protein", "Protein / Gene", "Canonical Gene", 0.98, "匹配权威人类核心基因/蛋白数据库列表")

        # 检查细胞系与细胞类型
        for cand in all_candidates:
            if cand in cls.CELL_LINES:
                return make_res("Cell_Tissue", "Cell Line", "Model Cell Line", 0.96, "匹配国际标准细胞系名称列表")
            for pat in cls.CELL_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Cell_Tissue", "Cell Subpopulation", "Cell Type", 0.92, f"匹配细胞类型规则: {pat}")
            for pat in cls.TISSUE_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Cell_Tissue", "Tissue / Organ", "Anatomy", 0.90, f"匹配组织解剖学器官词缀: {pat}")

        # 检查生物学过程与通路 (优先于疾病词缀以防 -ptosis/-osis 混淆)
        for cand in all_candidates:
            for pat in cls.PROCESS_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Biological_Process", "Biological Pathway / Process", "Process", 0.90, f"匹配生物信号传导/生理通路词缀: {pat}")

        # 检查疾病与病理表型
        for cand in all_candidates:
            for pat in cls.DISEASE_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Disease_Phenotype", "Clinical Disease / Pathology", "Pathology", 0.90, f"匹配医学病理学/疾病名称词缀: {pat}")

        # 规则无法明确判断，交由 LLM 状态机后续分轮处理
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
        # 尝试按文件名匹配
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
    """图谱分析工作台全流程分析与执行引擎"""

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
        # 1. 查找已知别名与来源
        aliases = []
        doc_source = "未知"
        for ent in entities:
            if ent.get("standard_name") == node_name:
                aliases = ent.get("aliases", [])
                doc_source = ent.get("doc_source", "未知")
                break

        # 2. 收集相邻关系谓词与关联实体
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
            "neighbors": connected_triplets[:8],  # 最多取前8个代表性拓扑连接
            "evidence": evidence_snippets[:3]     # 最多取前3段高代表性微证据
        }

    def _fetch_paper_sentences(self, node_name: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Round 2: 根据正则精准在原文献全文中截取命中关键句 (避免通读全文)"""
        matched_sentences = []
        if not self.paper_full_texts:
            return matched_sentences

        # 查找该实体关联的原文献名称
        target_doc = None
        aliases = []
        for ent in entities:
            if ent.get("standard_name") == node_name:
                target_doc = ent.get("doc_source", "")
                aliases = ent.get("aliases", [])
                break

        search_texts = []
        if target_doc:
            # 清理类似 [Figure: ...] 标记
            clean_doc = re.sub(r'\[.*?\]', '', target_doc).strip()
            for k, text in self.paper_full_texts.items():
                if clean_doc and (clean_doc in k or k in clean_doc):
                    search_texts.append(text)
        if not search_texts:
            # 若无法精确命中文件名，搜索所有可用文献全文
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

    def _format_ontology_categories(self) -> str:
        """格式化实体候选类别清单供 Prompt 注入"""
        cats = self.ontology.get("entity_categories", {})
        lines = []
        for k, v in cats.items():
            lines.append(f"- `{k}` ({v.get('label', k)}): {v.get('description', '')}")
        return "\n".join(lines)

    def _format_relation_categories(self) -> str:
        """格式化关系机制候选类别清单供 Prompt 注入"""
        cats = self.ontology.get("relation_categories", {})
        lines = []
        for k, v in cats.items():
            lines.append(f"- `{k}` ({v.get('label', k)})")
        return "\n".join(lines)

    def _parse_llm_json(self, raw_str: str) -> Optional[Dict[str, Any]]:
        """清洗并严格反序列化大模型返回的 JSON"""
        if not raw_str:
            return None
        cleaned = raw_str.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            # 尝试正则截取首个大括号块
            match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def _llm_classify_entity_round1(self, node_name: str, micro_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Round 1: 依据局部网络微上下文分类单节点"""
        system_prompt = (
            "你是一位资深的计算生物学与生物信息学本体专家。你的任务是根据图谱局部微上下文对生物医学实体进行严谨分类。\n"
            "若上下文证据足以判断，请归类到指定类别之一；若信息不足、缺乏把握或无法明确判定，务必直接输出 primary_type 为 \"Uncertain\"。"
        )
        cat_str = self._format_ontology_categories()
        neighbors_str = "\n".join(micro_ctx["neighbors"]) if micro_ctx["neighbors"] else "无直接记录"
        evidence_str = "\n".join(micro_ctx["evidence"]) if micro_ctx["evidence"] else "无直接原文微证据"

        user_content = (
            f"【待分析实体】: {node_name}\n"
            f"【实体已知别名】: {', '.join(micro_ctx['aliases']) if micro_ctx['aliases'] else '无'}\n"
            f"【网络邻居关系】:\n{neighbors_str}\n"
            f"【局部文献证据】:\n{evidence_str}\n\n"
            f"【候选实体类别集合】:\n{cat_str}\n\n"
            "【输出要求】:\n"
            "请严格以单条合法 JSON 格式输出，不要包含任何多余解释:\n"
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
            "reasoning": "大模型调用异常或未能返回合法结构"
        }

    def _llm_classify_entity_round2(self, node_name: str, matched_sentences: List[str]) -> Dict[str, Any]:
        """Round 2: 根据正则精准回溯原文语句进行判定"""
        system_prompt = (
            "你是一位资深的生物医学专家。前轮分析因上下文微弱无法断定该实体类型，现通过原文献正则精准检索到以下关键句子。\n"
            "请根据这些真实文献语句再次进行分类。若依然缺乏把握或证据不足，请将 primary_type 设为 \"Uncertain\"。"
        )
        cat_str = self._format_ontology_categories()
        sents_str = "\n".join([f"- {s}" for s in matched_sentences])

        user_content = (
            f"【待分析实体】: {node_name}\n"
            f"【从原文精准检索出的上下文语句】:\n{sents_str}\n\n"
            f"【候选实体类别集合】:\n{cat_str}\n\n"
            "【输出要求】:\n"
            "请严格以单条合法 JSON 格式输出:\n"
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
            "reasoning": "Round 2 原文回溯分析未果"
        }

    def _llm_classify_entity_round3(self, node_name: str) -> Dict[str, Any]:
        """Round 3: 兜底通用生物知识判定"""
        system_prompt = (
            "你是一位资深的生物信息学专家。该实体在文献局部和原文语句中信息均不足。\n"
            "请仅根据你在通用生物医学领域的常识进行最终研判。如果你确信其属于某个类别，请输出对应类别；若该词汇为课题自命名代号或无法确认，请坦率输出 \"Uncertain\"。"
        )
        cat_str = self._format_ontology_categories()

        user_content = (
            f"【待分析生物医学实体名】: {node_name}\n\n"
            f"【候选实体类别集合】:\n{cat_str}\n\n"
            "【输出要求】:\n"
            "请严格以合法 JSON 输出:\n"
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
            "reasoning": "Round 3 通用知识兜底研判未果"
        }

    def _llm_classify_relation(self, source: str, target: str, orig_rel: str, evidence: str) -> Dict[str, Any]:
        """对单条关系进行深层机制分类与机制动词归纳"""
        system_prompt = (
            "你是一位分子机制生物学家。你的任务是对生物知识图谱中的关系进行深层机制分类，"
            "并归纳出一个精准精炼的机制动词谓词（如：催化磷酸化、竞争性结合、转录激活、靶向降解等）。"
        )
        rel_cat_str = self._format_relation_categories()
        src_type = self.entity_annotations.get(source, {}).get("primary_type", "未知")
        tgt_type = self.entity_annotations.get(target, {}).get("primary_type", "未知")

        user_content = (
            f"【主体实体】: {source} (已分类为: {src_type})\n"
            f"【客体实体】: {target} (已分类为: {tgt_type})\n"
            f"【原始关系谓词】: {orig_rel}\n"
            f"【文献原文证据】: {evidence if evidence else '无具体原文句子'}\n\n"
            f"【候选关系机制类别】:\n{rel_cat_str}\n\n"
            "【输出要求】:\n"
            "请严格以合法 JSON 输出，不要包含任何多余文字:\n"
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
            "reasoning": "分析降级至默认关联"
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
                     progress_callback: Optional[Callable[[float, str], None]] = None,
                     log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        执行完整的工作台分级分析工作流
        :param entities: 实体列表
        :param relations: 关系三元组列表
        :param enable_entities: 是否开启实体分析
        :param enable_relations: 是否开启关系分析
        :param enable_round0_rules: 是否开启 Round 0 快速规则预过滤 (0 Token)
        :param enable_round1_context: 是否开启 Round 1 微上下文分析
        :param enable_round2_regex: 是否开启 Round 2 原文正则回溯
        :param enable_round3_world: 是否开启 Round 3 通用生物常识兜底
        :param force_reanalyze: 是否忽略已有缓存强制重新分析
        :param progress_callback: 进度回调 (0.0 ~ 1.0, 状态文字)
        :param log_callback: 日志输出回调
        """
        def log(msg: str):
            if log_callback:
                log_callback(msg)
            print(f"[WorkbenchEngine] {msg}")

        log("🚀 开始启动图谱分析工作台处理流水线...")

        # 1. 计算热度与优先级
        degrees = self.compute_node_degrees(relations)
        all_node_names = [e.get("standard_name") for e in entities if e.get("standard_name")]
        # 按热度降序排序
        sorted_nodes = sorted(all_node_names, key=lambda n: degrees.get(n, 0), reverse=True)

        total_nodes = len(sorted_nodes)
        total_rels = len(relations)
        self.stats["total_nodes"] = total_nodes
        self.stats["total_relations"] = total_rels

        # ----------------------------------------------------
        # 阶段 A: 实体分级升级状态机 (Round 0 -> 1 -> 2 -> 3)
        # ----------------------------------------------------
        if enable_entities and total_nodes > 0:
            log(f"📊 实体分析队列已就绪，共计 {total_nodes} 个实体，已按拓扑中心度降序排列。")

            # === Round 0: 快速规则匹配 (0 Token, 极速预过滤) ===
            r0_hits = 0
            if enable_round0_rules:
                log("⚡ [Round 0: 快速规则预分流] 正在执行无 Token 规则词缀初筛...")
                for idx, node in enumerate(sorted_nodes):
                    if not force_reanalyze and node in self.entity_annotations:
                        cur = self.entity_annotations[node]
                        if cur.get("status") == "resolved":
                            continue

                    # 查找别名
                    aliases = []
                    for ent in entities:
                        if ent.get("standard_name") == node:
                            aliases = ent.get("aliases", [])
                            break

                    rule_res = RulePreFilter.classify(node, aliases, self.ontology)
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
                                "reasoning": "待进一步分级判断",
                                "round": 0,
                                "status": "uncertain"
                            }
                self.stats["round0_hits"] = r0_hits
                log(f"✅ [Round 0 完成] 极速命中 {r0_hits} 个实体，节省大量 LLM Token！")

            # === Round 1: 微上下文分析 ===
            if enable_round1_context and self.llm_agent:
                r1_candidates = [n for n in sorted_nodes if self.entity_annotations.get(n, {}).get("status") == "uncertain"]
                log(f"🧠 [Round 1: 局部微上下文分析] 共有 {len(r1_candidates)} 个待决实体进入大模型分析...")
                for idx, node in enumerate(r1_candidates):
                    if progress_callback:
                        progress_callback(0.1 + 0.3 * (idx / max(1, len(r1_candidates))), f"Round 1: 正在分析实体 ({idx+1}/{len(r1_candidates)}): {node}")

                    micro_ctx = self._extract_micro_context(node, entities, relations)
                    res = self._llm_classify_entity_round1(node, micro_ctx)
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
                log(f"✅ [Round 1 完成] 成功解析 {self.stats['round1_hits']} 个实体。")

            # === Round 2: 文献正则精准回溯 ===
            if enable_round2_regex and self.llm_agent and self.paper_full_texts:
                r2_candidates = [n for n in sorted_nodes if self.entity_annotations.get(n, {}).get("status") == "uncertain"]
                log(f"📖 [Round 2: 文献精准原文回溯] 共有 {len(r2_candidates)} 个顽固未决实体进入文献检索...")
                for idx, node in enumerate(r2_candidates):
                    if progress_callback:
                        progress_callback(0.4 + 0.2 * (idx / max(1, len(r2_candidates))), f"Round 2: 正在回溯原文 ({idx+1}/{len(r2_candidates)}): {node}")

                    matched_sents = self._fetch_paper_sentences(node, entities)
                    if matched_sents:
                        res = self._llm_classify_entity_round2(node, matched_sents)
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
                log(f"✅ [Round 2 完成] 文献回溯成功破解 {self.stats['round2_hits']} 个实体。")

            # === Round 3: 通用常识兜底 ===
            if enable_round3_world and self.llm_agent:
                r3_candidates = [n for n in sorted_nodes if self.entity_annotations.get(n, {}).get("status") == "uncertain"]
                log(f"🌐 [Round 3: 通用生物医学常识兜底] 共有 {len(r3_candidates)} 个未决实体尝试领域常识研判...")
                for idx, node in enumerate(r3_candidates):
                    if progress_callback:
                        progress_callback(0.6 + 0.15 * (idx / max(1, len(r3_candidates))), f"Round 3: 常识兜底研判 ({idx+1}/{len(r3_candidates)}): {node}")

                    res = self._llm_classify_entity_round3(node)
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
                log(f"✅ [Round 3 完成] 领域常识兜底解析出 {self.stats['round3_hits']} 个实体。")

        # ----------------------------------------------------
        # 阶段 B: 关系机制深层分析与机制动词归纳
        # ----------------------------------------------------
        if enable_relations and total_rels > 0 and self.llm_agent:
            log(f"🔗 关系分析队列已就绪，共计 {total_rels} 条关系...")
            for idx, rel in enumerate(relations):
                src = rel.get("source")
                tgt = rel.get("target")
                orig_rel = rel.get("relation", "相关")
                evidence = rel.get("evidence", "")
                edge_key = f"{src}|{tgt}|{orig_rel}"

                if not force_reanalyze and edge_key in self.relation_annotations:
                    cur = self.relation_annotations[edge_key]
                    if cur.get("status") == "resolved":
                        continue

                if progress_callback:
                    progress_callback(0.75 + 0.25 * (idx / max(1, total_rels)), f"正在分析关系机制 ({idx+1}/{total_rels}): {src} -> {tgt}")

                res = self._llm_classify_relation(src, tgt, orig_rel, evidence)
                self.relation_annotations[edge_key] = {
                    **res,
                    "round": 1,
                    "status": "resolved"
                }
                self.stats["analyzed_relations"] += 1

            log(f"✅ [关系机制分析完成] 成功提炼 {self.stats['analyzed_relations']} 条深层机制动词与类别。")

        # ----------------------------------------------------
        # 阶段 C: 统计与指标计算
        # ----------------------------------------------------
        resolved_cnt = sum(1 for a in self.entity_annotations.values() if a.get("status") == "resolved")
        uncertain_cnt = sum(1 for a in self.entity_annotations.values() if a.get("status") == "uncertain" or a.get("primary_type") == "Uncertain")
        self.stats["resolved_nodes"] = resolved_cnt
        self.stats["uncertain_nodes"] = uncertain_cnt

        if progress_callback:
            progress_callback(1.0, "分析全部完成！正在渲染工作台图谱...")

        log("🎉 [图谱分析工作台流水线全部执行完毕！]")
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
