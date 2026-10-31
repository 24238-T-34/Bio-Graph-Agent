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
    """基于生物医学命名规则与权威词缀的高速 0-Token 正则预分类器"""

    # 1. 物种、宿主与模式生物 (Species / Organism)
    SPECIES_PATTERNS = [
        r'^(?:human|humans|patient|patients|host|hosts|homo sapiens)$',
        r'^(?:mouse|mice|mus musculus|murine)$',
        r'^(?:rat|rats|rattus|rattus norvegicus)$',
        r'^(?:zebrafish|danio rerio)$',
        r'^(?:drosophila|drosophila melanogaster|fruit fly|fruit flies)$',
        r'^(?:c\. ?elegans|caenorhabditis elegans|nematode|nematodes)$',
        r'^(?:yeast|saccharomyces|saccharomyces cerevisiae|s\. ?cerevisiae)$',
        r'^(?:primate|primates|non-human primate|nhp|macaque|rhesus|chimpanzee)$',
        r'^(?:pig|pigs|swine|porcine|rabbit|rabbits|canine|dog|dogs|bovine|cow|cows)$',
        r'^(?:人类|患者|宿主|小鼠|大鼠|斑马鱼|果蝇|线虫|酵母|灵长类|恒河猴|猕猴|黑猩猩|猪|兔|犬|牛)$'
    ]

    KNOWN_SPECIES = {
        'human', 'homo sapiens', 'mouse', 'mus musculus', 'rat', 'rattus',
        'zebrafish', 'danio rerio', 'drosophila', 'c. elegans', 'yeast',
        's. cerevisiae', 'macaque', 'rhesus', 'pig', 'rabbit', 'dog',
        '人类', '小鼠', '大鼠', '斑马鱼', '果蝇', '线虫', '酵母', '灵长类'
    }

    # 2. 微生物、病原体、细菌与病毒 (Microbe / Pathogen)
    MICROBE_PATTERNS = [
        r'.*(?:virus|viral|viridae)$',
        r'.*病毒$',
        r'.*(?:bacteria|bacterium|bacillus|coccus|spirillum|mycoplasma|chlamydia)$',
        r'.*(?:microbiome|microbiota|gut flora|pathogen|pathogens)$',
        r'.*(?:细菌|杆菌|球菌|真菌|支原体|衣原体|菌群|病原体|寄生虫)$',
        r'^[a-z]\. ?(?:coli|aureus|pylori|pneumoniae|tuberculosis)$'
    ]

    KNOWN_PATHOGENS = {
        'sars-cov-2', 'covid-19', 'coronavirus', 'hiv', 'hbv', 'hcv', 'hpv', 'ebv', 'cmv',
        'influenza', 'influenza virus', 'zika', 'dengue', 'hsv', 'vzv', 'rsv',
        'e. coli', 'escherichia coli', 's. aureus', 'staphylococcus', 'staphylococcus aureus',
        'h. pylori', 'helicobacter pylori', 'm. tuberculosis', 'mycobacterium tuberculosis',
        'candida albicans', 'c. albicans', 'aspergillus', 'salmonella', 'pseudomonas',
        'pseudomonas aeruginosa', 'streptococcus', 'streptococcus pneumoniae',
        'plasmodium', 'plasmodium falciparum', 'gut microbiota', 'microbiome'
    }

    # 3. 组织器官、解剖结构与宏观生理系统 (Organ / Tissue / Anatomy)
    ORGAN_PATTERNS = [
        r'.*(?:tissue|organ|liver|lung|lungs|kidney|kidneys|heart|brain|spleen|pancreas|colon|stomach|bone marrow|blood|serum|plasma|cortex|stroma|endothelium|epithelium|system|tract|barrier|vessel|artery|vein|lymph node|gut|intestine|skin|retina)$',
        r'.*(?:组织|器官|肝脏|肺|肾脏|心脏|脑|大脑|小脑|脾脏|胰腺|结肠|胃|骨髓|血液|血清|血浆|皮质|基质|系统|管腔|通路|屏障|血管|动脉|静脉|淋巴结|肠道|肠|皮肤|视网膜)$',
        r'.*(?:blood-brain barrier|bbb|central nervous system|cns|cardiovascular system|immune system|circulatory system|digestive system)$',
        r'.*(?:血脑屏障|中枢神经系统|心血管系统|免疫系统|循环系统|消化系统)$'
    ]

    # 4. 细胞亚群与模式细胞系 (Cell / Cell Line)
    CELL_PATTERNS = [
        r'.*(?:cell|cells)$', r'.*细胞$',
        r'.*(?:lymphocyte|monocyte|macrophage|neutrophil|fibroblast|platelet|splenocyte)$',
        r'.*(?:淋巴细胞|单核细胞|巨噬细胞|中性粒细胞|成纤维细胞|血小板)$',
        r'.*(?:t[- ]?cell|b[- ]?cell|nk[- ]?cell|dendritic cell|treg|th1|th2|th17)$',
        r'.*(?:t细胞|b细胞|nk细胞|树突状细胞|调节性t细胞)$',
        r'.*(?:neuron|astrocyte|microglia|myocyte|adipocyte|hepatocyte|podocyte)$',
        r'.*(?:神经元|星形胶质细胞|小胶质细胞|肌细胞|脂肪细胞|肝实质细胞|足细胞)$'
    ]

    CELL_LINES = {
        'hela', 'hek293', 'hek293t', 'mcf-7', 'mcf7', 'a549', 'jurkat', 'pc3', 'u2os',
        'k562', 'thp-1', 'thp1', 'raw264.7', 'cos-7', 'cho', 'nih3t3', 'hl-60', 'sk-br-3',
        'hct116', 'sw480', 'mdamb231', 'mda-mb-231', 'b16f10', '4t1', 'panc-1', 'hep-g2', 'hepg2'
    }

    # 5. 药物与化学小分子 (INN 权威国际非专利药品规范)
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

    # 6. 酶、蛋白质与受体
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
        r'^[a-z0-9]{2,6}r[0-9]+$',   # e.g., tgfbr1, ifngr1, il6r2
        r'^[a-z0-9]+rec$',
    ]

    RNA_PATTERNS = [
        r'^(?:mir|mirna|microrna|lncrna|circrna|sncrna|sirna|pirna)[\-_].*',
        r'.*[\-_](?:mir|mirna|microrna|lncrna|circrna|sirna)$'
    ]

    GENE_SYMBOL_REGEX = re.compile(r'^[A-Z][A-Z0-9]{1,8}(?:-[A-Z0-9]+)?$')
    KNOWN_MAJOR_GENES = {
        'TP53', 'EGFR', 'KRAS', 'NRAS', 'HRAS', 'MYC', 'STAT3', 'STAT1', 'CDK4', 'CDK6',
        'BRCA1', 'BRCA2', 'VEGFA', 'VEGFR', 'IL6', 'TNF', 'AKT1', 'AKT2', 'MTOR', 'BRAF',
        'PIK3CA', 'PTEN', 'MAPK1', 'MAPK3', 'CTNNB1', 'HIF1A', 'CASP3', 'BCL2', 'BAX',
        'JUN', 'FOS', 'NFKB1', 'RELA', 'SMAD3', 'SMAD4', 'ERBB2', 'MET', 'ALK', 'RET',
        'ATM', 'ATR', 'CHEK1', 'CHEK2', 'NOTCH1', 'JAK1', 'JAK2', 'TYK2', 'SRC', 'ABL1',
        'MAPK', 'ERK', 'JNK', 'P38', 'WNT', 'TGFB', 'IFNG', 'PDCD1', 'CD274', 'CTLA4'
    }

    # 7. 疾病与病理表型模式
    DISEASE_PATTERNS = [
        r'.*(?:oma|itis|pathy|osis|emia|syndrome|carcinoma|sarcoma|melanoma|leukemia|lymphoma|failure|disorder|infection)$',
        r'.*(?:瘤|炎|病|综合征|癌|肉瘤|白血病|淋巴瘤|衰竭|障碍|感染)$',
        r'.*(?:cancer|tumor|neoplasm|hypertension|diabetes|asthma|stroke|infarction|alzheimer|parkinson|fibrosis|cirrhosis|sepsis|metastasis)$',
        r'.*(?:癌症|肿瘤|高血压|糖尿病|哮喘|中风|心肌梗死|阿尔茨海默病|帕金森病|纤维化|肝硬化|脓毒症|转移)$'
    ]

    # 8. 生物学过程与通路模式 (包含 -ptosis 等程序性死亡模式)
    PROCESS_PATTERNS = [
        r'.*ptosis$',   # apoptosis, ferroptosis, pyroptosis, necroptosis
        r'.*(?:pathway|signaling|cascade|apoptosis|autophagy|proliferation|differentiation|migration|invasion|angiogenesis|metastasis|glycolysis|metabolism|transcription|translation|ferroptosis|pyroptosis|cell cycle|immune response)$',
        r'.*(?:通路|信号转导|级联反应|凋亡|自噬|增殖|分化|迁移|侵袭|血管生成|糖酵解|代谢|转录|翻译|铁死亡|焦亡|细胞周期|免疫反应)$'
    ]

    # 9. 突变位点与序列变异模式
    VARIANT_PATTERNS = [
        r'^rs\d+$',
        r'^[a-z]\d+[a-z]$',          # e.g., v600e, t790m, g12d
        r'^p\.[a-z]\d+[a-z]$',        # e.g., p.v600e
        r'.*(?:exon|intron)\s*\d+.*',
        r'.*(?:mutation|mutant|variant|polymorphism|snp|indel|duplication|deletion)$',
        r'.*(?:突变|变异|多态性|缺失|重复)$'
    ]

    @classmethod
    def classify(cls, node_name: str, aliases: Optional[List[str]] = None, ontology: Optional[Dict[str, Any]] = None, language: str = "zh") -> Optional[Dict[str, Any]]:
        """
        尝试使用基于医学规则的快速分类 (覆盖宏观物种、微生物、器官、细胞至分子突变，支持中英双语推理理由输出)
        """
        raw_name = node_name.strip()
        lower_name = raw_name.lower()
        all_candidates = [lower_name]
        if aliases:
            all_candidates.extend([a.strip().lower() for a in aliases if a.strip()])

        target_categories = set(ontology.get("entity_categories", {}).keys()) if ontology else set()
        is_en = "en" in str(language).lower()

        def make_res(cat_id: str, sub_type: str, tag: str, conf: float, reason_zh: str, reason_en: str) -> Optional[Dict[str, Any]]:
            final_cat = cat_id
            if target_categories:
                if final_cat not in target_categories:
                    # 兼容性映射
                    if cat_id in ["Cell_CellLine", "Organ_Tissue"] and "Cell_Tissue" in target_categories:
                        final_cat = "Cell_Tissue"
                    elif "Species_Organism" in target_categories and cat_id == "Species_Organism":
                        final_cat = "Species_Organism"
                    elif "Microbe_Pathogen" in target_categories and cat_id == "Microbe_Pathogen":
                        final_cat = "Microbe_Pathogen"
                    elif "Kinase_Phosphatase" in target_categories and "kinase" in sub_type.lower():
                        final_cat = "Kinase_Phosphatase"
                    elif "Small_Molecule_Drug" in target_categories and cat_id == "Chemical_Drug":
                        final_cat = "Small_Molecule_Drug"
                    elif "Druggable_Target" in target_categories and ("receptor" in sub_type.lower() or "kinase" in sub_type.lower()):
                        final_cat = "Druggable_Target"
                    elif "Immune_Cell_Subset" in target_categories and cat_id == "Cell_CellLine":
                        final_cat = "Immune_Cell_Subset"
                    elif "Disease_Indication" in target_categories and cat_id == "Disease_Phenotype":
                        final_cat = "Disease_Indication"
                    elif "Pathway_Module" in target_categories and cat_id == "Biological_Process":
                        final_cat = "Pathway_Module"
                    elif cat_id not in target_categories:
                        return None

            return {
                "primary_type": final_cat,
                "sub_type": sub_type,
                "tags": [tag],
                "confidence": conf,
                "reasoning": reason_en if is_en else reason_zh,
                "round": 0,
                "status": "resolved"
            }

        # 1. 检查是否为物种/宿主/模式生物 (Species / Organism)
        for cand in all_candidates:
            if cand in cls.KNOWN_SPECIES:
                return make_res("Species_Organism", "Model Organism / Species", "Species", 0.98,
                                "匹配权威生物物种与模式生物词库", "Matched canonical model organism / species database")
            for pat in cls.SPECIES_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Species_Organism", "Host / Organism", "Species", 0.95,
                                    f"匹配生物学物种命名规则: {pat}", f"Matched binomial species nomenclature rule: {pat}")

        # 2. 检查是否为微生物/病原体/细菌/病毒 (Microbe / Pathogen)
        for cand in all_candidates:
            if cand in cls.KNOWN_PATHOGENS:
                return make_res("Microbe_Pathogen", "Microorganism / Pathogen", "Pathogen", 0.98,
                                "匹配权威病原微生物/细菌/病毒词库", "Matched authoritative pathogen / microbe vocabulary")
            for pat in cls.MICROBE_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Microbe_Pathogen", "Microbe / Pathogen", "Microbiology", 0.92,
                                    f"匹配微生物/病毒命名规则: {pat}", f"Matched microbe / virus naming pattern: {pat}")

        # 3. 检查是否为变异位点
        for cand in all_candidates:
            for pat in cls.VARIANT_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Sequence_Variant", "Sequence Mutation", "Gene Variant", 0.95,
                                    f"匹配权威基因变异/突变命名规则: {pat}", f"Matched canonical sequence variant / mutation rule: {pat}")

        # 4. 检查是否为代谢物或无机离子
        for cand in all_candidates:
            if cand in cls.METABOLITES_AND_IONS:
                return make_res("Chemical_Drug", "Metabolite / Ion", "Biochemical Molecule", 0.96,
                                "匹配权威内源性代谢小分子/离子词库", "Matched endogenous metabolite / biochemical ion library")

        # 5. 检查 INN 药物词缀
        for cand in all_candidates:
            for pat in cls.INN_DRUG_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Chemical_Drug", "Targeted / Synthetic Drug", "INN Pharmaceutical", 0.96,
                                    f"匹配 INN 国际权威药名词缀规则: {pat}", f"Matched INN international pharmaceutical stem rule: {pat}")

        # 6. 检查通用化合物词缀
        for cand in all_candidates:
            for pat in cls.CHEM_COMPOUND_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Chemical_Drug", "Chemical Compound", "Compound", 0.88,
                                    f"匹配化学分子词缀: {pat}", f"Matched chemical compound nomenclature affix: {pat}")

        # 7. 检查非编码 RNA
        for cand in all_candidates:
            for pat in cls.RNA_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Gene_Protein", "Non-coding RNA", "RNA", 0.95,
                                    f"匹配非编码 RNA 命名规则: {pat}", f"Matched non-coding RNA naming convention: {pat}")

        # 8. 检查酶/激酶
        for cand in all_candidates:
            for pat in cls.ENZYME_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Gene_Protein", "Enzyme / Kinase", "Catalytic Protein", 0.92,
                                    f"匹配酶学催化蛋白质命名词缀: {pat}", f"Matched enzymatic catalytic protein suffix: {pat}")

        # 9. 检查受体
        for cand in all_candidates:
            for pat in cls.RECEPTOR_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Gene_Protein", "Receptor Protein", "Receptor", 0.90,
                                    f"匹配生物膜/细胞受体命名规则: {pat}", f"Matched membrane / cellular receptor nomenclature: {pat}")

        # 10. 检查标准知名基因符号 (TP53, EGFR 等)
        if raw_name in cls.KNOWN_MAJOR_GENES:
            return make_res("Gene_Protein", "Protein / Gene", "Canonical Gene", 0.98,
                            "匹配权威人类核心基因/蛋白数据库列表", "Matched canonical human core gene / protein database")

        # 11. 检查细胞系与细胞类型 (Cell / CellLine)
        for cand in all_candidates:
            if cand in cls.CELL_LINES:
                return make_res("Cell_CellLine", "Cell Line", "Model Cell Line", 0.96,
                                "匹配国际标准细胞系名称列表", "Matched international standard cell line registry")
            for pat in cls.CELL_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Cell_CellLine", "Cell Subpopulation", "Cell Type", 0.92,
                                    f"匹配细胞类型规则: {pat}", f"Matched cell subpopulation nomenclature pattern: {pat}")

        # 12. 检查宏观组织器官与解剖结构 (Organ / Tissue)
        for cand in all_candidates:
            for pat in cls.ORGAN_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Organ_Tissue", "Organ / Tissue / Anatomy", "Anatomy", 0.92,
                                    f"匹配组织器官解剖系统规则: {pat}", f"Matched anatomical organ / tissue system pattern: {pat}")

        # 13. 检查生物学过程与通路 (优先于疾病词缀以防 -ptosis/-osis 混淆)
        for cand in all_candidates:
            for pat in cls.PROCESS_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Biological_Process", "Biological Pathway / Process", "Process", 0.90,
                                    f"匹配生物信号传导/生理通路词缀: {pat}", f"Matched biological pathway / signaling process suffix: {pat}")

        # 14. 检查疾病与病理表型
        for cand in all_candidates:
            for pat in cls.DISEASE_PATTERNS:
                if re.match(pat, cand, re.IGNORECASE):
                    return make_res("Disease_Phenotype", "Clinical Disease / Pathology", "Pathology", 0.90,
                                    f"匹配医学病理学/疾病名称词缀: {pat}", f"Matched clinical pathology / disease suffix: {pat}")

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
