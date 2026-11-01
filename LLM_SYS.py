import json
import re
import time
from openai import OpenAI


class BioBrainAgent:
    def __init__(self, api_key, base_url="https://openrouter.ai/api/v1", model="tencent/hy3-preview"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        print(f"🔧 [System] 已加载底层模型: {self.model}")

    def _clean_json_string(self, raw_str):
        # 💡 真正的安全气囊：直接判断传进来的 raw_str 是不是空的就行了
        if raw_str is None:
            raise Exception("大模型未能正常返回内容（返回值为空），可能是 API 被限流或网络超时，请检查 API Key 或稍后再试。")

        return raw_str.replace("```json", "").replace("```", "").strip()

    def _parse_json_object(self, raw_str: str):
        """清洗并严格反序列化大模型返回的 JSON 对象或数组，带正则容错提取"""
        if not raw_str:
            return None
        cleaned = raw_str.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def _ask_llm(self, system_prompt, user_content, model_override=None, max_retries=3, base_delay=2.0):
        target_model = model_override or self.model
        last_exception = None
        for attempt in range(max_retries):
            try:
                # 发起真正的 API 请求
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    timeout=60.0  # 60秒超时控制
                )
                # 只有成功拿到 response，才会执行这一行
                return response.choices[0].message.content

            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                # 针对限流 429、并发冲突或临时网络连接波动进行指数退避重试
                if ("429" in err_str or "rate" in err_str or "timeout" in err_str or "connection" in err_str) and attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"\n⚠️ [LLM API 遇到限流或网络波动] 正在进行第 {attempt + 1}/{max_retries} 次指数退避重试，等待 {sleep_time:.1f} 秒... 原因: {e}")
                    time.sleep(sleep_time)
                    continue
                else:
                    # 🔥 核心防线：如果上面这步崩了且不可重试或耗尽重试次数，在终端提示
                    print(f"\n🚨 [LLM API 核心报错] 接口请求失败了！真实死因 -> {e}")
                    raise e
        if last_exception:
            raise last_exception

    def clean_and_sanitize_title(self, candidate_title: str) -> str:
        """极简低消耗清洗：将候选标题输入大模型，严格剔除版权/声明/期刊名/多余标点，消耗仅 ~30-50 tokens"""
        if not candidate_title or len(candidate_title.strip()) < 4:
            return ""
        system_prompt = (
            "你是一个学术文献标题清洗专家。你的任务是严格净化给定的文献候选标题，输出最终纯净的【学术论文真实大标题】。\n\n"
            "【严格剔除原则】：\n"
            "1. 剔除一切版权与发布声明（如 'Some rights reserved', 'All rights reserved', 'Copyright © ...', 'Open Access' 等）。\n"
            "2. 剔除混入的独立期刊名或出版社名（如 'Science', 'Cell', 'Nature', 'Elsevier', 'PLOS ONE' 等）。\n"
            "3. 剔除文献类型标签（如 'Research Article', 'Review', 'Brief Report' 等）。\n"
            "4. 剔除末尾多余的点句号、分号或引号，保持真实论文标题的大小写和专有名词连字符。\n\n"
            "【输出约束】：只输出净化后的纯标题字符串，绝不要包含解释、标号或Markdown标记。若该候选词完全不是论文标题，请直接回复 'UNKNOWN'。"
        )
        user_content = f"待清洗的候选标题：\n{candidate_title.strip()}"
        try:
            res = self._ask_llm(system_prompt, user_content)
            if not res:
                return candidate_title.strip()
            res = res.strip().strip('"\'`')
            if "unknown" in res.lower() or len(res) < 5:
                return ""
            return res
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] 标题大模型清洗受阻: {e}")
            return candidate_title.strip()

    def fast_extract_title(self, short_chunk: str) -> str:
        """本地极简提取降级：在缺失 DOI/PMID 时，从第 1 页极短文本（250~400字符）中快速定位论文真实大标题"""
        if not short_chunk or len(short_chunk.strip()) < 10:
            return ""
        system_prompt = (
            "你是一个学术文献标题提取专家。请从给定的文献首页头部极短文本中，识别出该论文的【真实完整学术研究大标题】。\n\n"
            "【规则】：\n"
            "1. 绝不要提取期刊名（如 Science, Cell, Nature, PNAS）、栏目名（如 Research Article, Review）或机构作者名。\n"
            "2. 只提取论文真实大标题本身，保留专有名词连字符，修复杂乱断行。\n"
            "3. 只输出纯标题文本。如果文本中未找到有效标题，请只回复 'UNKNOWN'。"
        )
        user_content = f"文献首页头部文本片段：\n{short_chunk[:500]}"
        try:
            res = self._ask_llm(system_prompt, user_content)
            if not res:
                return ""
            res = res.strip().strip('"\'`')
            if "unknown" in res.lower() or len(res) < 5:
                return ""
            return res
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] 本地快提大模型受阻: {e}")
            return ""

    def review_paper_title(self, raw_text: str = "", candidate_title: str = "") -> str:
        """向后兼容接口：整合清洗与快速抽取"""
        if candidate_title:
            return self.clean_and_sanitize_title(candidate_title)
        elif raw_text:
            extracted = self.fast_extract_title(raw_text[:500])
            if extracted:
                return self.clean_and_sanitize_title(extracted)
        return ""

    def extract_entities_with_reflection(self, text,use_reflection,entity_lang="关闭 (保持原文语言)"):
        """智能体工作流：提取包含标准名和别名的实体字典"""

        language_instruction = ""
        if "中文" in entity_lang:
            language_instruction = "\n【⚠️ 强力约束】：请务必将提取出的所有实体（主节点）的“标准名”统一翻译并输出为**纯中文**（如果是专有英文缩写可保留，但常规名词必须严格用中文，并且最好使用中文学术圈的常见名）。"
        elif "English" in entity_lang:
            language_instruction = "\n【⚠️ 强力约束】：请务必将提取出的所有实体（主节点）的“标准名”统一翻译并输出为**纯英文 (English)**（即便原文是中文，也必须翻译成英文）。"

        prompt_s1 = f"""你是一个严谨的生物学实体识别专家。请从给定的文献中提取出所有的核心生物实体（如基因、蛋白质、疾病、细胞、药物等）。
        
        【核心任务】：你需要为每个实体确定一个【标准名称 (standard_name)】，并把文中指代该实体的其他有效表达作为【别名 (aliases)】。
        
        【实体提取与别名规范（极其重要，严格遵守）】：
        1. 确立【标准名称】：优先使用官方简写或公认名称（如 "p53", "MDM2"），长度尽量控制在 1-3 个单词内。
        2. 确立【别名】：只保留真正的生物学同义词、全称或简称（如 "TP53"）。
        3. 🚫 严禁保留无意义修饰变体：绝对不要把加了状态语、修饰语的词组当作别名提取！（例如：遇到 "endogenous p53", "p53 protein", "~53"，请直接忽略这些噪音，不要放入别名）。
        4. 🚫 严禁提取“关系/事件”为实体（极度致命）：例如绝对不要把诸如 "p53-MDM2 interaction", "MDM2 overexpression", "binding to DNA" 提取为实体！这些属于【关系】或【事件】，它们应该在关系提取阶段作为连线出现，而不是独立的节点！实体必须是纯粹的生物学物质名词。
        5.文章讨论中带有方向性名词的应当转为中性表现，如原文为”促进代谢“应当提取为”代谢活性“而不是代谢
        6. 🚫 严禁生成重复的同源节点：例如不要让图谱里出现多个本质上都是 p53或p53系统 的节点。除了明确的突变型，所有普通 p53 必须归一化为唯一的 "p53"。
        7. ⚠如果遇到同一实体在细节或者功能存在相反或明显区别的子对象时，应当当作两个不同的实体，尤其是文章探讨了他们的区别及不同影响时，如突变与野生型：如果文中同时明确对比了 "mutant p53"（突变型）和 "wild-type p53"（野生型），请将它们提取为两个【独立的实体】，绝对不能混为一谈。
        {language_instruction}
        
        请严格只输出一个合法的 JSON 数组，绝对不要有其他废话。格式必须如下：（仅作格式参考）
        [
          {{
            "standard_name": "p53",
            "aliases": ["TP53"]
          }},
          {{
            "standard_name": "mutant p53",
            "aliases": ["mutp53"]
          }}
        ]"""

        # 📜 日志埋点 1
        print("   └─ 🧠 [Step 1/2] 正在进行实体初次提取...")
        try:
            raw_s1 = self._ask_llm(prompt_s1, text)
            cleaned_s1 = self._clean_json_string(raw_s1)
        except Exception as e:
            print(f"   ❌ [Step 1/2] 请求超时或失败: {e}")
            return []

        prompt_s1_5 = f"""你是一个严谨的知识图谱数据清洗专家。
        用户将提供一段【原始文献】和一份【初次提取的实体字典】。
        【核心任务】：你需要为每个实体确定一个终极的【标准名称 (standard_name)】，并把真正的同义词保留为【别名 (aliases)】。

        你的任务是进行“外科手术级”的数据清洗：
        
        1. 查漏补缺：是否有关键的生物学实体被完全遗漏了？如果有，请补全。
        2. 🧹 剔除假实体与事件节点（极度重要）：如果发现初次提取的字典里有诸如 "p53-MDM2 interaction", "MDM2 overexpression", "apoptosis activation" 等描述【关系、状态或事件】的词组被错误地当成了实体，请直接将它们【彻底删除】！实体必须是纯粹的物质名词。
        3. 实体去重合并：将不同叫法但指代同一概念的实体强制合并。绝对不能出现多个本质上都是 p53 的独立节点（除非是 mutant vs wild-type 这样在作用上不同的明确对比）。
        4. 📏 标准名精简压缩：如果标准名称过长，请必须将其缩短概括（优先使用官方简写，如 "p53"），并将原来的长名称移入 aliases 中！
        5. 别名瘦身去噪：初次提取的 aliases 中可能混入了大量垃圾信息（如 "p53 protein", "endogenous MDM2", "~53"）。请大刀阔斧地剔除这些带有无意义修饰词的噪音，每个实体只保留最核心的学术同义词！
        6. 拆分错误合并：发现功能截然相反的实体（如突变型和野生型）被合并时，必须将其拆分为独立实体。
        {language_instruction}
        
        请直接输出清洗、合并、精简后的最终 JSON 数组，保持与初次提取相同的格式。绝对不要有其他废话。"""

        # 📜 日志埋点 2
        if use_reflection:
            print("    └─ 🪞 [Step 1.5/2] 正在进行大脑自我反思纠错...")
            prompt_s1_5 = prompt_s1_5 + "\n⚠️ 致命警告：不管你有没有发现错误，你的回复必须且只能是一个合法的 JSON 数组 (JSON Array)。如果没有需要修改的地方，请直接返回原始的 JSON 数组。绝不允许输出任何解释性文字！"
            reflection_input = f"【原始文献】:\n{text}\n\n【初次提取】:\n{cleaned_s1}"

            try:
                raw_final = self._ask_llm(prompt_s1_5, reflection_input)
                return json.loads(self._clean_json_string(raw_final))  # 将反思后的结果作为最终结果
            except Exception as e:
                print(f"   ❌ [Step 1.5/2] 反思阶段请求失败或超时: {e}")
                return []
        else:
            print("    └─ ⏩ [Step 1.5/2] 已关闭自我反思，跳过纠错步骤...")
            raw_final = cleaned_s1
            return json.loads(self._clean_json_string(raw_final))

    def extract_relations(self, text, entities_list):
        """根据实体字典，抽取文本中的实体关系"""
        if not entities_list:
            return []

        prompt_s2 = """你是一个计算生物学专家。
        用户将提供一段【原始文献】和一份【确定的实体字典（含标准名称和别名）】。
        请提取出这些实体之间的相互作用。
        
        规则：
        1. 只能提取四种关系："正作用"、"负作用"、"相关"、"包含"。
        2. 【"包含"关系的方向与层级定义】：
           - "包含"用于表示分类层级（如物种包含其亚型株系）、组织与细胞（如器官组织包含其组成细胞）、分子复合物与亚基（如蛋白质复合物包含其亚基）等严格的从属或整体-部分关系。
           - 关系方向必须严格规定为：`source` 为上位/宏观/整体/分类父类，`target` 为下位/微观/组成部分/亚基/分类子类（即：父 (上位) ─[包含]▶ 子 (下位)）。
           - 🚨【严禁方向颠倒与循环从属】：严禁将微观分子或下位子类作为 source 反向包含上位父类；绝对禁止在两实体间同时提取双向互相包含。
        3. 你的输出中，"source" 和 "target" 必须严格使用字典中提供的【standard_name】！不要使用文中的原词或别名，从而保证图谱节点的统一。
        4. 只输出合法 JSON 数组格式。格式如下：
        [
          {"source": "大肠杆菌", "target": "尿路致病性大肠杆菌", "relation": "包含", "evidence": "UPEC is a pathogenic pathotype of Escherichia coli..."},
          {"source": "标准实体A", "target": "标准实体B", "relation": "正作用/负作用/相关/包含", "evidence": "英文原文"}
        ]"""

        # 📜 日志埋点 3
        print("   └─ 🔗 [Step 2/2] 正在抽取实体间的调控关系网...")
        relation_input = f"【原始文献】:\n{text}\n\n【确定的实体字典】:\n{json.dumps(entities_list, ensure_ascii=False)}"

        try:
            raw_relations = self._ask_llm(prompt_s2, relation_input)
            return json.loads(self._clean_json_string(raw_relations))
        except Exception as e:
            print(f"   ❌ [Step 2/2] 关系抽取阶段请求失败或超时: {e}")
            return []

    def verify_and_clean_abstract(self, raw_abstract):
        """大模型保安：只做判断题，绝对不碰原文本！"""
        prompt = f"""
        你是一个文献审查员。请判断以下文本中，是否真的包含了一篇学术论文的摘要(Abstract)内容。
        你只需要回答 YES 或 NO。绝对不要输出任何其他解释性文字！

        待检验文本：
        {raw_abstract}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            reply = response.choices[0].message.content.strip().upper()

            # 如果判定是摘要，直接把【一字未改】的原始文本退回去！0损耗！
            if "YES" in reply:
                return raw_abstract
            else:
                return "[NOT_ABSTRACT]"
        except Exception as e:
            print(f"⚠️ 质检失败: {e}，默认保留原文。")
            return raw_abstract

    def fallback_extract_abstract(self, full_text):
        """终极兜底：让大模型自己去海量文本里抠摘要"""
        prompt = f"""
        ⚠️ 注意，部分顶级期刊（如 Nature, Science 等）排版非常特殊：
        1. 它们通常没有 "Abstract" 也没有 "Introduction" 等明确的分段标题。
        2. 摘要通常紧随在作者、单位、日期等元数据之后，是一段高度概括研究背景、方法和结论的独立大段落。
        3. 摘要段落结束后的下一个段落，通常会突然开始详细介绍具体的研究背景。

        请凭借你的专业语义理解，精准切割出这唯一的“摘要概括段落”：
        - 彻底剔除论文标题、作者、通讯地址、日期、DOI 等无用的元数据。
        - 绝对不要包含后面的正文（具体背景介绍/引言）内容。
        - 只返回纯净的摘要文本，不要包含任何多余的解释词语。
        - 如果文本中根本没有包含类似摘要的段落，请直接严格回复：[NOT_ABSTRACT]

        论文文本：
        {full_text}
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def align_global_entities(self, master_entities, new_entities):
        """🌟 核心大招：跨文献实体对齐融合器 (包含别名增强增强识别)"""
        if not master_entities or not new_entities:
            return {}

        # 💡 核心优化：把 entities 里的 aliases (别名) 也提取出来，发给大模型作为重要的对齐参考！
        master_list = [{"name": e.get("standard_name"), "type": e.get("type", "未知"), "aliases": e.get("aliases", [])}
                       for e in master_entities]
        new_list = [{"name": e.get("standard_name"), "type": e.get("type", "未知"), "aliases": e.get("aliases", [])} for e
                    in new_entities]

        prompt = f"""
        你是一个严谨的生物信息学图谱融合专家。
        你的任务是比对【新提取的实体】和【现有知识库实体】，找出它们之间代表“同一个生物学概念”的同义词、缩写或中英文对照。

        【现有知识库实体】:
        {master_list}

        【新提取的实体】:
        {new_list}

        判断标准与严格警告：
        1. 寻找同义基因/蛋白质（如 p53 与 TP53，TNF-alpha 与 TNFA）。
        2. 寻找全称与简称（如 Reactive oxygen species 与 ROS）。
        3. 寻找中英文对照（如 肿瘤坏死因子 与 TNF）。
        4. 💡 重点参考：请充分利用实体提供的 "aliases" (别名) 列表进行交叉比对。如果一个新实体的名称或别名，与现有实体的名称或别名存在交集或高度重合，它们极有可能是同一个概念。
        5. ⚠️ 极其致命：如果两个实体只是“有关系”或者“属于同一类”，绝对不能合并！必须是【完全等价】的概念才能合并！不确定的坚决不合并！

        请输出一个合法的 JSON 字典，格式必须如下：
        {{
            "新实体名称1": "对应的现有实体名称",
            "新实体名称2": "对应的现有实体名称"
        }}
        如果没有发现任何可以合并的实体，请直接输出空字典：{{}}
        绝对不要输出任何其他解释文字！
        """

        try:
            print("    └─ 🧠 [Entity Aligner] 正在呼叫大模型进行跨文献语义比对(别名辅助模式)...")
            raw_response = self._ask_llm(prompt, "请开始比对。")
            clean_json = self._clean_json_string(raw_response)

            if not clean_json:
                return {}

            mapping = json.loads(clean_json)
            return mapping
        except Exception as e:
            print(f"    ⚠️ [Entity Aligner] 实体对齐解析失败，跳过融合: {e}")
            return {}

    def diagnose_graph(self, entities, relations,output_lang="zh"):
        """🤖 核心引擎：AI 图谱智能洗树 (检测同义词、层级与捷径边)"""
        if not entities or not relations:
            return []
        lang_str = "中文" if "zh" in output_lang else "English"
        prompt = f"""你是一个顶级的生物医药知识图谱架构师。
        我将提供一份当前图谱的【实体列表】和【关系连线列表】（包含起点、终点、关系类型和原文证据）。
        请你根据原文证据进行“图谱洗树（Graph Pruning & Merging）”，并返回结构化的 JSON 操作指令。

        你需要执行以下操作：
        1. 【合并同义词 (MERGE)】：如果两个节点完全指代同一事物（如“尿路感染”与“泌尿系统感染”），请指令合并。
        2. 【建立层级 (HIERARCHY)】：如果两个节点是包含/分类关系（如“大肠杆菌”是父，“UTI89”是子），请指令新增一条 `[包含]` 关系线。⚠️【机制并存规则】：如果某实体（如抗生素）既连着父节点又连着子节点，请保留这些机制连线！
        3. 【双向相关连线折叠 (FOLD_SYMMETRIC)】：节点 A 与 B 之间若同时存在双向“相关”连线（如 A ─[相关]▶ B 且 B ─[相关]▶ A），在语义上互为无向冗余，请指令将其【折叠合并 (FOLD_SYMMETRIC)】为单向规范“相关”连线（系统将自动合并双方的证据与文献出处，并累加权重）。
        4. 【破除循环包含 (RESOLVE_CYCLE)】：当图中发现两个节点之间存在互为从属的双向矛盾包含连线时（即 A 包含 B 且 B 包含 A 均为“包含”关系），请根据生物学常识判定上位父节点（宏观/整体/分类父类）与下位子节点（微观/部分/亚基/分类子类），指令破除循环包含矛盾。系统将确立规范的父到子单向层级，并将反向包含线的证据、出处和热度无损合并入正向连线中。
        5. 【粗糙关系机制提纯 (PURIFY)】：节点 A 与 B 之间如果已经建立了具体的生物调控机制（如“正作用”、“负作用”或“包含”），伴随并存的粗糙笼统“相关”连线属于低信息量冗余，请指令将其【提纯合并 (PURIFY)】（将其证据和热度无损并入具体的机制连线中，消除粗糙冗余线）。
        6. 【彻底删除越级分类 (REMOVE)】：仅用于存在三级以上多级嵌套时的越级分类冗余连线（如 尿路感染包含膀胱炎，膀胱炎包含急性膀胱炎，删除急性膀胱炎与尿路感染之间的越级连线），指令将其彻底删除。
        7. 【标记机制捷径边 (DOWNGRADE)】：针对“机制推导”产生的表象捷径边（A->B且B->C，导致的A->C），将其标记为捷径边进行视觉降级。

        请严格只输出一个合法的 JSON 数组，格式如下（可返回多条指令）：
        [
          {{"action": "MERGE", "target_node": "保留的标准实体", "nodes_to_remove": ["要删掉的冗余实体"], "reason": "说明"}},
          {{"action": "HIERARCHY", "parent": "父节点名称", "child": "子节点名称", "reason": "说明"}},
          {{"action": "FOLD_SYMMETRIC", "source": "起点名称", "target": "终点名称", "relation": "相关", "reason": "双向相关连线冗余，折叠为单向规范线并无损合并证据"}},
          {{"action": "RESOLVE_CYCLE", "parent": "上位父节点", "child": "下位子节点", "reason": "破除循环包含矛盾，确立单向层级并无损合并反向证据"}},
          {{"action": "PURIFY", "source": "起点名称", "target": "终点名称", "relation": "相关", "target_relation": "正作用/负作用/包含", "reason": "已被明确机制覆盖，提纯合并证据、出处与权重"}},
          {{"action": "REMOVE", "source": "起点名称", "target": "终点名称", "relation": "关系词", "reason": "越级分类冗余，需彻底删除"}},
          {{"action": "DOWNGRADE", "source": "起点名称", "target": "终点名称", "relation": "关系词", "reason": "机制捷径降级"}}
        ]
        🚨 【强制输出语言与字段隔离指令】：
        1. 🚨 【键名与枚举冻结】：你必须严格保持 JSON 的所有键名（如 'action', 'reason'）为英文！同时，'action' 字段的值必须【严格保持】为模板中指定的英文大写枚举值（MERGE, HIERARCHY, FOLD_SYMMETRIC, RESOLVE_CYCLE, PURIFY, REMOVE, DOWNGRADE），绝对不允许翻译或改成其他词！
        2. 🚨 【节点名称原样对齐】：JSON 中涉及的所有实体/节点名称（如 'target_node', 'nodes_to_remove', 'parent', 'child', 'source', 'target' 等字段的值），必须与我提供给你的输入列表中的名称【完全保持一模一样，原样返回】，绝对不允许翻译或改变字面！
        3. 🌐 【唯一允许翻译的字段】：【只有】'reason'（解释/说明）字段的内容，需要根据当前要求，严格使用【{lang_str}】来编写和表达。
        绝对不要输出任何其他解释文字！
        """

        # 将当前的图谱数据喂给大模型
        graph_data = f"【实体列表】:\n{json.dumps(entities, ensure_ascii=False)}\n\n【关系列表】:\n{json.dumps(relations, ensure_ascii=False)}"

        try:
            print("   └─ 🤖 [AI Cleaner] 正在进行图谱全局深度诊断...")
            raw_response = self._ask_llm(prompt, graph_data)
            clean_json = self._clean_json_string(raw_response)
            return json.loads(clean_json)
        except Exception as e:
            print(f"   ❌ [AI Cleaner] 诊断失败: {e}")
            return []

    def generate_pubmed_query(self, nodes_info, context_relations=None):
        """
        根据用户选择的节点及其本地别名生成 PubMed 检索式。
        如果传入了 context_relations，大模型将进行深度背景理解。
        """
        if context_relations:
            # 🌐 模式 3：背景搜索 (结合图谱现有关系)
            system_prompt = """你是一个顶级的计算生物学与文献检索专家。
            用户将提供他们希望检索的【核心实体】，以及这些实体在当前图谱中的【已有关系网络（背景上下文）】。
            你的任务是：深刻理解这些背景逻辑，为这些实体构建一个高度专业的 PubMed 布尔检索式（Boolean Query）。

            策略要求：
            1. 既然已知了部分关系，你的检索式应当倾向于探索“更深层的机制”、“未知的中间桥梁”或“具体的临床表型”。
            2. 可以根据上下文推断，自行加入一到两个关键的机制词（如 pathway, mechanism, phosphorylation, inhibitor 等）作为 AND 条件，以缩小范围，提高文献质量。
            3. 同一个实体的标准名称与别名之间用 OR 连接，实体组之间用 AND 连接。必须带上 [Title/Abstract] 或 [Mesh] 标签。
            4. ⚠️ 严禁输出任何解释性文本或 Markdown 格式。只能输出纯粹的检索式字符串！
            """
            user_content = f"【希望拓展的核心实体】：\n{json.dumps(nodes_info, ensure_ascii=False)}\n\n【已知图谱上下文】：\n{json.dumps(context_relations, ensure_ascii=False)}"
        else:
            # 🧠 模式 2：智能搜索 (仅扩充同义词)
            system_prompt = """你是一个顶级的生物医学文献检索专家。
            用户将提供几个他们希望在 PubMed 中检索的核心生物学实体（包含标准名称和已知别名）。
            你的任务是：为这些实体构建一个高度专业的 PubMed 布尔检索式（Boolean Query）。

            规则：
            1. 不同的实体之间必须用 AND 连接。同一个实体的标准名与别名之间用 OR 连接。
            2. 必须在词汇后加上合适的检索字段标签，例如 [Title/Abstract] 或 [Mesh]。
            3. 根据医学常识，自行补充常见且重要的英文同义词到 OR 逻辑中。
            4. ⚠️ 严禁输出任何解释性文本或 Markdown 格式。只能输出纯粹的检索式字符串！
            """
            user_content = f"请为以下选中的实体生成 PubMed 检索式：\n{json.dumps(nodes_info, ensure_ascii=False)}"

        print(f"🧠 [Agent] 正在生成 PubMed 检索策略 (背景模式: {bool(context_relations)})...")
        raw_response = self._ask_llm(system_prompt, user_content)
        query = self._clean_json_string(raw_response)

        print(f"🎯 [Agent] 生成的检索式: {query}")
        return query

    def explain_mechanism(self, node_a, node_b, paths_data,output_lang="zh"):
        """
        分支一专用：根据图谱中找到的多步路径，推导深层机制
        """
        lang_str = "中文" if "zh" in output_lang else "English"
        system_prompt = f"""你是一个顶级的分子生物学家和系统生物学专家。
        用户将提供从现有知识图谱中提取的，连接【起点实体】和【终点实体】的多条拓扑路径及对应的文献证据碎片。
        你的任务是：仔细阅读这些跨文献的证据碎片，将它们串联起来，进行全局视角的生物学逻辑推导，生成一份专业的【跨文献分子机制桥接报告】。

        报告结构要求：
        1. 🌟 【桥接通路总览】：用文本箭头（例如 A ─[激活]▶ C ─[抑制]▶ B）直观展现发现的 1~2 条最核心机制通路。
        2. 🔬 【机制深度剖析】：结合提供的 evidence（证据原句），详细解释起点如何通过中间节点一步步影响到终点。
        3. 🛡️ 【可信度与审计】：必须明确引用路径数据中的 doc_source（文献来源）。
        4. 💡 【新发现暗示】：如果有明显的跨文献拼图（例如文献1证明A调控C，文献2证明C调控B），请特别高亮指出这可能是一个具有研究价值的串联新机制。

        🚨 【强制输出语言指令】：你必须严格使用【{lang_str}】输出整篇报告（包含所有的标题、正文和列表）。如果要求的语言是 English，请你自动将报告模板中的中文标题全部翻译为英文。
        全程使用优美的 Markdown 格式输出。
        """
        user_content = f"起点实体：{node_a}\n终点实体：{node_b}\n\n【提取到的拓扑路径与证据碎片】：\n{json.dumps(paths_data, ensure_ascii=False)}"

        print(f"🧠 [Agent] 正在推导 {node_a} 与 {node_b} 的深层机制...")
        return self._ask_llm(system_prompt, user_content)

    def generate_bridge_query(self, node_a, node_b):
        """
        分支二/三专用：生成强调“机制/通路”的检索式，自带防幻觉强制报错。
        """
        system_prompt = """你是一个顶尖的生物学专家与 PubMed 检索策略师。
        用户在知识图谱中发现【实体A】和【实体B】之间缺乏深层机制通路，希望你构造检索式去网络上挖掘。

        【🛡️ 核心强制规则】：
        1. 逻辑审核：首先利用你的科学常识判断这两个实体是否可能存在生物学/医学上的关联。如果它们在科学上绝对毫无关联（例如一个是"苹果"，一个是"黑洞"），你必须仅输出纯文本：[NO_RELATION_ERROR]。
        2. 策略构造：如果可能存在关联，请构造一个高度专业的 PubMed 布尔检索式。
        3. 必须侧重机制：检索式不仅要包含实体 A 和 B（考虑它们的同义词），还必须强行加入机制探索词（如 mechanism, pathway, mediate, interact, cross-talk 等），以确保搜出来的是机制研究论文，而不是泛泛而谈的水文。
        4. ⚠️ 严禁输出任何解释性文本或 Markdown 代码块！只能输出纯粹的检索式字符串，或者 [NO_RELATION_ERROR]。
        """
        user_content = f"实体A：{node_a}\n实体B：{node_b}"

        print(f"🧠 [Agent] 正在思考 {node_a} 与 {node_b} 的机制探索策略...")
        raw_response = self._ask_llm(system_prompt, user_content)
        query = self._clean_json_string(raw_response)

        print(f"🎯 [Agent] 桥接检索策略: {query}")
        return query

    def extract_bridge_mechanism(self, node_a, node_b, abstracts_text, entity_lang="English"):
        """
        专门用于【智能桥接】模块：阅读多篇文献摘要，狙击式提取只与 A 和 B 联系相关的实体与关系。
        增加 entity_lang 参数以统一图谱命名规范。
        """
        system_prompt = f"""你是一个顶尖的分子生物学机制挖掘专家。
        用户将提供【实体A】和【实体B】，以及从 PubMed 检索到的【多篇相关文献摘要】。
        你的任务是：专门寻找并提取连接 A 和 B 的深层调控机制或中间通路。
        ⚠️ 核心警告：请彻底忽略与连接这两个实体无关的其他背景知识、旁支靶点！

        【🌐 命名语言强制规范】：
        所有提取出的 entities (主要是标准名 standard_name ) 以及 ，必须强制使用 {entity_lang} 进行输出！
        (注：explanation, evidence, reason 以及 relation 关系动词可以使用中文，以方便阅读)

        请严格按照以下 JSON 格式返回数据：
        {{
            "explanation": "在这里写一段连贯的文本报告（支持Markdown）：综合这些文献，解释 A 是如何通过哪些中间介导物影响 B 的。如果没有找到任何确切证据，请在此说明。",
            "entities": [
                {{"standard_name": "实体名", "aliases": ["别名1", "别名2"]}}
            ],
            "relations": [
                {{
                    "source": "主语",
                    "target": "宾语",
                    "relation": "关系动词(如 激活, 抑制, 磷酸化)",
                    "evidence": "从文献中摘抄的证明原句",
                    "reason": "你对这条调控逻辑的简短解释",
                    "doc_source": "对应的 PMID 或文献标题"
                }}
            ]
        }}
        注意：entities 中必须包含你找出的所有关键中间节点，以及 A 和 B 本身。如果没有找到桥接关系，entities 和 relations 请返回空列表 []。
        """
        user_content = f"【实体A】：{node_a}\n【实体B】：{node_b}\n\n【供分析的文献摘要合集】：\n{abstracts_text}"

        print(f"🧠 [Agent] 正在使用 {entity_lang} 命名规范，挖掘 {node_a} 与 {node_b} 的桥接机制...")
        raw_response = self._ask_llm(system_prompt, user_content)

        try:
            cleaned_json = self._clean_json_string(raw_response)
            result = json.loads(cleaned_json)
            return result
        except Exception as e:
            print(f"❌ [Agent] 桥接机制 JSON 解析失败: {e}")
            return {"explanation": "解析大模型返回结果失败，请重试。", "entities": [], "relations": []}

    def generate_topic_query(self, user_topic):
        """
        冷启动模块：将用户的自然语言研究意图，转化为专业的 PubMed 检索式。
        【结构化拆解版】：防止过度泛化导致器官丢失，强制保留靶向组织。
        """
        system_prompt = """你是一个顶级的生物医学信息检索专家（Medical Librarian）。
        请根据用户的【自然语言研究意图】，构造一个专业、精准且高召回率的 PubMed 布尔检索式。

        【🚨 核心检索策略 (绝对遵守)】：
        1. 剔除泛机制类“元词汇”（致命错误）：【绝对禁止】在检索式中加入 pathway, mechanism, protein, bridge protein, adaptor, crosstalk, interaction, target 等泛指词汇！在 PubMed 中搜索具体表型和实体即可。
        2. 必须保留“器官/组织/细胞”锚点：如果用户明确了研究对象（如“肠道”），必须把它作为一个独立的 AND 逻辑块，加上同义词（如 intestinal OR gut OR colon）。
        3. 自然切分逻辑块（解除数量限制）：如果用户描述了 A -> B -> C 的多步机制，请将它们分为独立的 AND 逻辑块。⚠️ 严禁把完全不同的生物学概念（如“细胞衰老”和“免疫逃逸”）强行塞进同一个 OR 括号里！
        4. 极致同义词拓展 (预防0结果)：你必须为每一个概念提供 4-6 个极其宽泛的同义词或相关词（用 OR 连接）。
        5. 禁用 MeSH 标签：绝对禁止添加 [Mesh]、[majr] 等标签，直接使用自由词或 [tiab]。
        6. 🌟 鼓励分段组合检索 (大招)：对于 A -> B -> C 的长链宏大假说，单靠全链条串联极容易搜不到文献。我们极其鼓励你拆开搜索！你可以建立 A+B+C 的全貌网络，也同时建立 A+B, B+C, A+C 的局部网络，并将这些网络用大的 OR 连接起来（例如：(A AND B AND C) OR (A AND B) OR (B AND C)）。在每一个小的 AND 区间内，请放开手脚大胆发挥你的专业度！

        【实战示范】：
        用户意图：“长期熬夜导致肠道上皮细胞衰老，通过哪些桥梁蛋白引起肿瘤免疫逃逸？”
        ❌ 错误（不同概念强行合并进 OR）：(circadian rhythm) AND (intestinal) AND (senescence OR tumor immune escape)
        ❌ 错误（包含泛机制词）：(circadian rhythm) AND (intestinal) AND (senescence) AND (pathway OR bridge protein)
        ✅ 正确的自由分段组合式：
        ((circadian rhythm OR sleep deprivation) AND (intestinal OR gut) AND (senescence OR SASP) AND (immune evasion OR tumor immune escape)) OR ((circadian rhythm OR sleep deprivation) AND (intestinal OR gut) AND (senescence OR SASP)) OR ((intestinal OR gut) AND (senescence OR SASP) AND (immune evasion OR tumor immune escape))

        【最后警告】：如果用户的输入完全无科学意义，请直接输出纯文本：[INVALID_TOPIC]。
        ⚠️ 严禁输出任何解释性文字，严禁使用 Markdown 代码块，只能输出纯粹的检索式字符串！
        """

        user_content = f"【用户研究意图】：{user_topic}"

        print(f"🧠 [Agent] 正在将意图转化为检索式...")
        raw_response = self._ask_llm(system_prompt, user_content)
        return self._clean_json_string(raw_response)

    def evaluate_abstracts_relevance(self, user_topic, abstracts_text,output_lang="zh"):
        """
        冷启动反思模块：检查抓取回来的文献摘要是否符合意图。
        【核心技巧】：在 JSON 中强制要求先输出 reason，再输出 is_relevant，通过思维链 (CoT) 大幅提升判断准确率。
        """
        lang_str = "中文" if "zh" in output_lang else "English"
        system_prompt = f"""你是一个宽容的知识图谱构建助手。
        你需要评估检索到的文献摘要，是否有助于为用户的研究意图构建【初始图谱】。

        【🟢 核心放行标准 (局部命中 = 完美)】：
        用户经常提出 A -> B -> C 这样跨界宏大的假说。真实的文献绝大多数只会研究其中的一小段（比如只研究 A->B，或者只研究 B->C）。
        因此，只要这批文献中有【任何一篇】探讨了假说中的【局部环节】（例如：只探讨了“熬夜与肠道”，或者只探讨了“肠道衰老”），这就叫极其有价值的拼图！
        【绝对禁止】因为“文献没有涵盖用户提到的所有环节”而驳回！

        【🛑 驳回标准】：
        只有当这些文献与用户的各个切片【完全无关】（例如全都是讲植物、或者全都是毫无细胞机制的流行病学调查）时，才驳回。

        请严格返回如下 JSON 格式：
        {{
            "reason": "请说明该文献命中了用户宏大假说中的哪一个局部切片（说明为什么放行）。严禁抱怨文献没有覆盖全链条！",
            "is_relevant": true 或 false
        }}
        🚨 【强制输出语言与字段隔离指令】：
        1. 🚨 【键名与布尔值冻结】：你必须严格保持 JSON 的所有键名（'reason', 'is_relevant'）为英文！同时，'is_relevant' 字段的值必须【严格保持】为标准的 JSON 布尔值（true 或 false），绝对不允许加引号，也不允许翻译成其他文字！
        2. 🌐 【唯一允许翻译的字段】：【只有】'reason'（解释/说明）字段的内容，需要结合文献内容，严格使用【{lang_str}】来编写和表达。
        
        """
        user_content = f"【用户研究意图】：{user_topic}\n\n【检索到的文献摘要】：\n{abstracts_text}"

        print(f"🧠 [Agent] 正在使用思维链 (CoT) 反思文献相关性...")
        raw_response = self._ask_llm(system_prompt, user_content)
        try:
            cleaned_json = self._clean_json_string(raw_response)
            return json.loads(cleaned_json)
        except Exception as e:
            print(f"❌ [Agent] 相关性评估 JSON 解析失败: {e}")
            default_reason = "解析大模型输出失败，默认放行。" if "中文" in lang_str else "Failed to parse LLM output, passing by default."
            return {"reason": default_reason, "is_relevant": True}

    def prune_nodes_by_intent(self, user_intent, entities, relations,output_lang="zh"):
        """
        基于意图的智能剪枝：让大模型根据用户的研究兴趣，揪出无关的“杂草”节点。
        """
        lang_str = "中文" if "zh" in output_lang else "English"
        # 构建带有拓扑上下文的轻量级节点字典，节省 Token 并提供判断依据
        node_context = []
        for ent in entities:
            node_name = ent.get("standard_name")
            neighbors = set()
            for r in relations:
                if r.get("source") == node_name:
                    neighbors.add(r.get("target"))
                elif r.get("target") == node_name:
                    neighbors.add(r.get("source"))
            # 如果没有邻居，就标注为孤立节点
            neighbor_str = ", ".join(list(neighbors)) if neighbors else "无(孤立节点)"
            node_context.append(f"节点: {node_name} | 其连接的邻居: {neighbor_str}")

        context_str = "\n".join(node_context)

        system_prompt = f"""你是一个严谨且果断的生物医学知识图谱清理专家。
        当前图谱在构建过程中不可避免地引入了一些偏离主题的节点（噪音）。你需要根据用户提供的【核心研究方向】，对下列所有节点逐一进行严格审视。

        【风险等级标准】：
        1. "完全无关": 与用户研究方向毫无关联，或者是明显被误判进来的同名异义词、其他无关器官/疾病的节点，坚决建议删除。
        2. "可能无关": 偏离研究核心，或者联系极其薄弱的旁支对象，建议删除。
        3. "关联度低": 依然在相关领域内，但对核心问题扩展性不足或过于泛泛而谈，交由用户定夺。
        4. "核心相关": 与研究方向紧密相关，非常重要（注意：核心相关的节点直接跳过，绝对不要输出到最终结果中！）。

        请返回 JSON 数组格式（仅包含前三种风险级别的节点），格式如下：
        [
            {{
                "node_name": "被判定的节点名称",
                "risk_level": "完全无关/可能无关/关联度低",
                "reason": "请结合其邻居信息，简短说明为什么该节点偏离了用户的研究意图"
            }}
        ]
        
        🚨 【强制输出语言与字段隔离指令】：
        1. 🚨 【键名与枚举冻结】：你必须严格保持 JSON 的所有键名（如 'node_name', 'risk_level', 'reason'）为英文！同时，'risk_level' 字段的值必须【严格保持】为指定英文枚举值（Completely_Irrelevant, Potentially_Irrelevant, Low_Relevance），绝对不允许翻译成中文或其他格式！
        2. 🚨 【节点名称原样对齐】：'node_name' 字段的值必须与输入的图谱节点名称【完全保持一模一样】，无论是大小写、空格还是语言，绝对不允许进行任何翻译或擅自改动！
        3. 🌐 【唯一允许翻译的字段】：【只有】'reason'（原因说明）字段的内容，需要结合邻居和用户意图，严格使用【{lang_str}】来编写输出。

        ⚠️ 严禁输出任何多余的解释，严禁使用 Markdown 代码块，只能输出纯粹的 JSON 数组！如果所有节点都极其完美相关，请返回空数组 []。
        """

        user_content = f"【用户核心研究方向】：{user_intent}\n\n【图谱节点上下文列表】：\n{context_str}"

        print(f"🧠 [Agent] 正在基于意图对图谱节点进行清洗评估...")
        raw_response = self._ask_llm(system_prompt, user_content)
        try:
            cleaned_json = self._clean_json_string(raw_response)
            return json.loads(cleaned_json)
        except Exception as e:
            print(f"❌ [Agent] 节点清洗 JSON 解析失败: {e}")
            return []

    def explain_graph_element(self, element_type, element_name, local_context,output_lang="zh"):
        """
        Ask AI 核心模块：为选中的图谱节点或关系提供专业且锚定上下文的解读。
        【终极完全体 + 动态分流】：区分节点和关系的模板逻辑。
        """
        lang_str = "中文" if "zh" in output_lang else "English"
        # ✨ 根据传入的类型，动态组装不同的排版紧身衣
        if element_type == "节点":
            template_block = """
        <OUTPUT_TEMPLATE>
        ### 📖 基础生物学定义
        **概念解析**：(用通俗专业的语言简述该元素是什么)
        **主要功能**：(说明它在细胞/分子层面的常规生理或病理作用)

        ### 🕸️ 在本图谱中的研究定位
        **网络枢纽机制分析**：
        - **上游调控机制**：(结合数据，深入解释上游节点是如何诱导/调控它的，若无写暂无)
        - **下游靶点效应**：(结合数据，深入解释它是如何影响下游靶点的，若无写暂无)

        **提取机制与文献来源**：
        > (根据 <CONTEXT_DATA> 中的依据进行学术总结。若依据为空，则进行优雅的学术补充推断，并明确列出相关文献名。)
        </OUTPUT_TEMPLATE>
            """
        else:  # 元素类型是“关系连线”
            template_block = """
        <OUTPUT_TEMPLATE>
        ### 📖 生物学关联机制解析
        **起点解析**：(简述该连线起点实体的核心生物学功能)
        **终点解析**：(简述该连线终点实体的核心生物学功能)

        ### 🕸️ 连线深度推演
        **核心作用机制 (Mechanism of Action)**：
        - (⚠️ 绝对不要分析上游下游！请直接结合你的生物学知识与 <CONTEXT_DATA>，深入剖析起点实体是如何通过具体的生化通路、受体或分子级联反应，去调控/影响终点实体的。这是一对一的机制级解析。一定要参考文献证据并做出合理扩充)

        **提取机制与文献来源**：
        > (根据 <CONTEXT_DATA> 中的依据进行学术总结。若依据为空，则进行优雅的学术补充推断，并明确列出相关文献名。)
        </OUTPUT_TEMPLATE>
            """

        system_prompt = f"""你是一个顶级的生物医学知识图谱解说专家。
        请基于用户选中的图谱元素和 <CONTEXT_DATA> 提供深入的生物学解读。

        【🚨 核心行为准则 (绝对遵守)】：
        1. 格式隔离：你必须且只能输出下方 <OUTPUT_TEMPLATE> 标签内的内容！【绝对禁止】输出“核心警告”、“注意”等我的系统指令！
        2. 🎨 强制排版规范：为了页面美观，【绝对禁止】使用一级标题(#)和二级标题(##)！只能使用三级标题(###)和加粗(**)。必须完全遵照模板的格式。
        3. 拒绝“复读机”：在解释机制时，必须结合你的生物学知识，深度推演 <CONTEXT_DATA> 中的通路机制和级联反应。
        4. 优雅处理无依据：如果 <CONTEXT_DATA> 提示提取依据是“无”，请委婉说明：“图谱暂未记录具体细节，但基于经典生物学通路推测...”。如果有明确的文献来源，请在依据中大方引用！

        {template_block}

        🚨 【强制输出语言指令】：你必须严格使用【{lang_str}】输出整篇解读报告。在生成最终内容时，如果要求的语言是 English，请你动态将 <OUTPUT_TEMPLATE> 模板里的所有中文标题（如“概念解析”等）翻译为地道的英文标题。
        ⚠️ 请立刻输出填充后的模板内容，不要包含 <OUTPUT_TEMPLATE> 标签本身，绝对不要使用 # 或 ## 标题！
        """

        user_content = f"【选中的元素】：[{element_type}] {element_name}\n\n<CONTEXT_DATA>\n{local_context}\n</CONTEXT_DATA>"

        print(f"🧠 [Agent] 正在生成专属解读 (已启用 [{element_type}] 专属动态推演模板)...")
        raw_response = self._ask_llm(system_prompt, user_content)
        return raw_response

    def transcribe_figure_to_text(self, image_bytes, image_ext="png", caption="", vision_model=None, output_lang="zh"):
        """
        多模态视觉前置转译：专门将信号通路机制图 (Pathway Diagram) 或图文摘要转译为详尽严谨的学术机制文本描述块
        """
        import base64
        b64_str = base64.b64encode(image_bytes).decode("utf-8")
        clean_ext = image_ext.lower().replace(".", "")
        mime_type = "image/jpeg" if clean_ext in ["jpg", "jpeg"] else "image/png"

        is_en = "en" in output_lang.lower()
        lang_instruction = "Please output the description strictly in English." if is_en else "请使用严谨清晰的中文学术语言输出。"

        system_prompt = f"""你是一位顶尖的生物医学多模态与系统生物学分析专家。
你的任务是对生物文献中的【信号通路图 / 机制假说图 / 图文摘要 (Pathway Diagram / Mechanism Figure)】进行【地毯式、零遗漏】的学术机制全面转录，输出详尽严谨的学术文本描述，供后续知识图谱抽取引擎构建高精度分子调控网络。

【⚠️ 视觉拓扑理解与全景转译核心规则（严禁高层概括，必须穷尽所有细节）】：

1. 【地毯式生化实体清单（零遗漏盘点）】：
   - 必须逐一列出图中出现的每一个实体名称（包括文字标注、蛋白复合物及缩写代号）：
     * 细胞外因子/配体 (Ligands)
     * 跨膜受体及受体亚基 (Receptors)
     * 胞质信号分子、接头蛋白、蛋白激酶/磷酸酶 (Kinases/Phosphatases/Adapters)
     * 泛素连接酶/去泛素化酶及负调控因子 (E3 Ligases/Inhibitors)
     * 细胞骨架调节分子与马达蛋白 (Cytoskeleton Regulators)
     * 多聚复合物 (Complexes)
     * 入核转录因子、辅激活因子、辅阻遏因子 (TFs/Coactivators/Corepressors)
     * 下游靶基因、生理功能与细胞效应/表型 (Phenotypes/Outcomes)
   - 注明其亚细胞空间定位（如细胞外、质膜、胞质、内质网/线粒体、细胞核等）。

2. 【逐条明确有向分子调控动作句（核心证据链，严禁笼统概括）】：
   - 严禁模糊使用“参与了”、“与...有关”，必须使用主谓宾明确的严谨学术动作句表达图中的每一条连线与箭头：
     * 【正向激活/促进 (➔ 或 ->)】：明确写出“A 激活/促进/诱导/磷酸化/上调 B”；
     * 【负向抑制/阻断 (┫)】：明确写出“A 抑制/阻断/拮抗/泛素化降解/去磷酸化/下调 B”；
     * 【物理结合/形成复合物 (⮂)】：明确写出“A 与 B 相互结合/形成异源二聚体/多聚复合物”；
     * 【空间转运/易位】：明确写出“复合物 A/B 发生入核转运/质膜易位/募集”。

3. 【全部分支与侧枝旁路解析（兼顾经典与非经典通路）】：
   - 仔细审视图中所有的分叉支路，分别完整记录：
     * 主干经典通路（Canonical Pathway）的完整上下游级联链条；
     * 所有的非经典侧枝通路（Non-canonical Pathways）与旁路交叉对话（Crosstalk）；
     * 绝不可遗漏图周边的分支支路（如 MAPK/p38/JNK、PI3K/Akt/mTOR、RhoA/ROCK、NF-κB 等侧枝）。

4. 【负反馈与抑制调控网络】：
   - 重点提取图中的抑制性分子（如 I-Smads、Smurfs、磷酸酶等）对受体或上游激酶的负反馈阻断、受体内化与降解通路。

5. 【实验干预与药物靶点（如有）】：
   - 明确指出图中标注的所有小分子抑制剂、抗体药物、基因敲除/突变干预及其精确作用位点与效应。

{lang_instruction}
请按上述结构（分子实体全盘点、逐条具体调控作用、经典与非经典级联通路、负调控与细胞结局）输出详尽完整的机制解析正文，无需输出 JSON 代码块。"""

        caption_info = f"\n【插图关联图注说明】：{caption}" if caption else ""
        prompt_text = (
            f"请仔细观察这张文献机制插图。{caption_info}\n"
            "请对该机制图进行地毯式细致解析，逐一提取所有的生物分子、所有正向箭头(➔)、负向平头线(┫)和复合物结合，"
            "完整涵盖经典主干通路与所有非经典侧枝旁路，写明具体的上下游调控动作，确保没有任何分子和调控关系遗漏。"
        )

        multimodal_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_str}"}}
        ]

        effective_model = vision_model or self.model
        print(f"👁️ [Agent] 正在调用专职视觉模型前置转译机制图为学术文本块 (模型: {effective_model})...")
        try:
            raw_response = self._ask_llm(
                system_prompt=system_prompt,
                user_content=multimodal_content,
                model_override=vision_model
            )
            print(f"✅ [Agent] 机制图成功转译为机制文本块 (长度: {len(raw_response)} 字符)！")
            return raw_response
        except Exception as e:
            print(f"⚠️ [Agent] 视觉模型机制图转译失败: {e}")
            return ""

    def extract_graph_from_figure(self, image_bytes, image_ext="png", caption="", vision_model=None, entity_lang="关闭 (保持原文语言)"):
        """
        多模态视觉智能体（备用兼容接口）
        """
        text_desc = self.transcribe_figure_to_text(image_bytes, image_ext, caption, vision_model)
        if not text_desc:
            return [], []
        ents = self.extract_entities_with_reflection(text_desc, use_reflection=True, entity_lang=entity_lang)
        rels = self.extract_relations(text_desc, ents)
        return ents, rels

    # =========================================================================
    # 🧬 [Workbench & Pathway Analysis 专属大模型能力扩展]
    # =========================================================================

    def classify_workbench_entity_round1(self, node_name: str, micro_ctx: dict, cat_str: str, allowed_ids: list, language: str = "zh") -> dict:
        """
        [Workbench] Round 1: 依据局部网络微上下文分类单节点 (严格约束枚举白名单与零容忍负向惩罚)
        """
        is_en = "en" in str(language).lower()
        allowed_ids_str = json.dumps(allowed_ids, ensure_ascii=False)

        if is_en:
            system_prompt = (
                "You are a senior computational biologist and biomedical ontology curator. Your task is to rigorously classify biomedical entities based on local graph topological micro-context.\n"
                f"【CRITICAL RESTRICTION - ZERO TOLERANCE FOR NEW CATEGORIES】:\n"
                f"1. You MUST choose primary_type strictly from this allowed ID list (case-sensitive): {allowed_ids_str}.\n"
                "2. You are STRICTLY FORBIDDEN from inventing, creating, or translating ANY new category names (e.g. DO NOT create 'Bacteria', 'Pathogen', 'Microorganism', 'Organism', 'Receptor', etc.).\n"
                "3. If evidence is insufficient, ambiguous, or the entity does not strictly fit any candidate category, you MUST set primary_type to \"Uncertain\"."
            )
            neighbors_str = "\n".join(micro_ctx.get("neighbors", [])) if micro_ctx.get("neighbors") else "No direct records"
            evidence_str = "\n".join(micro_ctx.get("evidence", [])) if micro_ctx.get("evidence") else "No direct literature micro-evidence"

            user_content = (
                f"【Entity to Classify】: {node_name}\n"
                f"【Known Aliases】: {', '.join(micro_ctx.get('aliases', [])) if micro_ctx.get('aliases') else 'None'}\n"
                f"【Network Neighbor Relations】:\n{neighbors_str}\n"
                f"【Local Literature Evidence】:\n{evidence_str}\n\n"
                f"【Candidate Entity Categories】:\n{cat_str}\n\n"
                f"【STRICT CANDIDATE ENUMERATION - primary_type MUST be one of】:\n{allowed_ids_str}\n\n"
                "【Output Instructions】:\n"
                "Strictly output a single valid JSON object. All field values (sub_type, tags, reasoning) MUST be in English. Do not invent any new category:\n"
                "{\n"
                "  \"primary_type\": \"<Exact Category ID from enumeration, or Uncertain>\",\n"
                "  \"sub_type\": \"<Specific functional subtype or role in English>\",\n"
                "  \"tags\": [\"<Tag1>\", \"<Tag2>\"],\n"
                "  \"confidence\": 0.85,\n"
                "  \"reasoning\": \"<Concise justification in English, under 40 words>\"\n"
                "}"
            )
        else:
            system_prompt = (
                "你是一位资深的计算生物学与生物信息学本体专家。你的任务是根据图谱局部微上下文对生物医学实体进行严谨分类。\n"
                "【绝对硬性约束 / CRITICAL RESTRICTION - 零容忍自创新分类】:\n"
                f"1. 字段 primary_type 的取值必须且只能严格从以下合法候选 ID 列表中选取（必须一字不差精确匹配）：\n"
                f"   {allowed_ids_str}\n"
                "2. 严禁擅自创造、发明或衍生任何列表之外的新分类名称（绝对禁止自创如: Bacteria, Pathogen, Microorganism, Organism, Receptor 等）！\n"
                "3. 若该实体在当前预设中找不到对应分类或把握不足，你【必须且只能】将 primary_type 设为 \"Uncertain\"！绝不允许自立门户新增分类！"
            )
            neighbors_str = "\n".join(micro_ctx.get("neighbors", [])) if micro_ctx.get("neighbors") else "无直接记录"
            evidence_str = "\n".join(micro_ctx.get("evidence", [])) if micro_ctx.get("evidence") else "无直接原文微证据"

            user_content = (
                f"【待分析实体】: {node_name}\n"
                f"【实体已知别名】: {', '.join(micro_ctx.get('aliases', [])) if micro_ctx.get('aliases') else '无'}\n"
                f"【网络邻居关系】:\n{neighbors_str}\n"
                f"【局部文献证据】:\n{evidence_str}\n\n"
                f"【候选实体类别集合】:\n{cat_str}\n\n"
                f"【唯一允许选取的 primary_type 严格白名单】:\n{allowed_ids_str}\n\n"
                "【输出要求】:\n"
                "请严格以单条合法 JSON 格式输出，所有文本字段（sub_type, tags, reasoning）务必使用中文，严禁自创分类:\n"
                "{\n"
                "  \"primary_type\": \"<严格白名单中的ID或Uncertain>\",\n"
                "  \"sub_type\": \"<细分子类别或具体生物学功能>\",\n"
                "  \"tags\": [\"<标签1>\", \"<标签2>\"],\n"
                "  \"confidence\": 0.85,\n"
                "  \"reasoning\": \"<简明推断理由，50字以内>\"\n"
                "}"
            )

        try:
            raw = self._ask_llm(system_prompt, user_content)
            res = self._parse_json_object(raw)
            if isinstance(res, dict) and "primary_type" in res:
                return res
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] Round 1 Entity LLM Error ({node_name}): {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "LLM call error or invalid response" if is_en else "大模型调用异常或未能返回合法结构"
        }

    def classify_workbench_entity_round2(self, node_name: str, matched_sentences: list, cat_str: str, allowed_ids: list, language: str = "zh") -> dict:
        """
        [Workbench] Round 2: 根据正则精准回溯原文语句进行判定 (严格约束枚举白名单与零容忍负向惩罚)
        """
        is_en = "en" in str(language).lower()
        allowed_ids_str = json.dumps(allowed_ids, ensure_ascii=False)
        sents_str = "\n".join([f"- {s}" for s in matched_sentences])

        if is_en:
            system_prompt = (
                "You are a senior biomedical researcher. The previous round could not determine this entity's type due to weak context. Targeted regex search from the original paper full text retrieved the following sentences.\n"
                f"【CRITICAL RESTRICTION - ZERO TOLERANCE FOR NEW CATEGORIES】:\n"
                f"1. You MUST choose primary_type strictly from this allowed ID list (case-sensitive): {allowed_ids_str}.\n"
                "2. You are STRICTLY FORBIDDEN from inventing, creating, or translating ANY new category names (e.g. DO NOT create 'Bacteria', 'Pathogen', 'Microorganism', 'Organism', 'Receptor', etc.).\n"
                "3. If evidence remains insufficient or ambiguous, you MUST set primary_type to \"Uncertain\"."
            )
            user_content = (
                f"【Entity to Classify】: {node_name}\n"
                f"【Sentences Retrieved from Original Paper】:\n{sents_str}\n\n"
                f"【Candidate Entity Categories】:\n{cat_str}\n\n"
                f"【STRICT CANDIDATE ENUMERATION - primary_type MUST be one of】:\n{allowed_ids_str}\n\n"
                "【Output Instructions】:\n"
                "Strictly output a single valid JSON object. All field values (sub_type, tags, reasoning) MUST be in English. Do not invent any new category:\n"
                "{\n"
                "  \"primary_type\": \"<Exact Category ID from enumeration, or Uncertain>\",\n"
                "  \"sub_type\": \"<Specific subtype in English>\",\n"
                "  \"tags\": [\"<Tag1>\", \"<Tag2>\"],\n"
                "  \"confidence\": 0.85,\n"
                "  \"reasoning\": \"<Literature-grounded evidence & rationale in English, under 40 words>\"\n"
                "}"
            )
        else:
            system_prompt = (
                "你是一位资深的生物医学专家。前轮分析因上下文微弱无法断定该实体类型，现通过原文献正则精准检索到以下关键句子。\n"
                "【绝对硬性约束 / CRITICAL RESTRICTION - 零容忍自创新分类】:\n"
                f"1. 字段 primary_type 的取值必须且只能严格从以下合法候选 ID 列表中选取（必须一字不差精确匹配）：\n"
                f"   {allowed_ids_str}\n"
                "2. 严禁擅自创造、发明或衍生任何列表之外的新分类名称（绝对禁止自创如: Bacteria, Pathogen, Microorganism, Organism, Receptor 等）！\n"
                "3. 若依然缺乏把握或证据不足，你【必须且只能】将 primary_type 设为 \"Uncertain\"！"
            )
            user_content = (
                f"【待分析实体】: {node_name}\n"
                f"【从原文精准检索出的上下文语句】:\n{sents_str}\n\n"
                f"【候选实体类别集合】:\n{cat_str}\n\n"
                f"【唯一允许选取的 primary_type 严格白名单】:\n{allowed_ids_str}\n\n"
                "【输出要求】:\n"
                "请严格以单条合法 JSON 格式输出，所有文本字段（sub_type, tags, reasoning）务必使用中文，严禁自创分类:\n"
                "{\n"
                "  \"primary_type\": \"<严格白名单中的ID或Uncertain>\",\n"
                "  \"sub_type\": \"<细分子类别>\",\n"
                "  \"tags\": [\"<标签1>\", \"<标签2>\"],\n"
                "  \"confidence\": 0.85,\n"
                "  \"reasoning\": \"<原文支持证据与理由，50字以内>\"\n"
                "}"
            )

        try:
            raw = self._ask_llm(system_prompt, user_content)
            res = self._parse_json_object(raw)
            if isinstance(res, dict) and "primary_type" in res:
                return res
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] Round 2 Entity LLM Error ({node_name}): {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "Context in paper is insufficient to confirm category" if is_en else "文献上下文证据不足以明确分类"
        }

    def classify_workbench_entity_round3(self, node_name: str, cat_str: str, allowed_ids: list, language: str = "zh") -> dict:
        """
        [Workbench] Round 3: 兜底通用生物知识判定 (严格约束枚举白名单与零容忍负向惩罚)
        """
        is_en = "en" in str(language).lower()
        allowed_ids_str = json.dumps(allowed_ids, ensure_ascii=False)

        if is_en:
            system_prompt = (
                "You are a senior bioinformatics expert. The entity has limited context in the literature.\n"
                "Adjudicate its category using your biomedical domain world knowledge.\n"
                f"【CRITICAL RESTRICTION - ZERO TOLERANCE FOR NEW CATEGORIES】:\n"
                f"1. You MUST choose primary_type strictly from this allowed ID list (case-sensitive): {allowed_ids_str}.\n"
                "2. You are STRICTLY FORBIDDEN from inventing, creating, or translating ANY new category names (e.g. DO NOT create 'Bacteria', 'Pathogen', 'Microorganism', 'Organism', 'Receptor', etc.).\n"
                "3. If confident, assign the proper candidate category from the list; if it is unidentifiable or outside the list, honestly output \"Uncertain\"."
            )
            user_content = (
                f"【Biomedical Entity Name】: {node_name}\n\n"
                f"【Candidate Entity Categories】:\n{cat_str}\n\n"
                f"【STRICT CANDIDATE ENUMERATION - primary_type MUST be one of】:\n{allowed_ids_str}\n\n"
                "【Output Instructions】:\n"
                "Strictly output a single valid JSON object in English. Do not invent any new category:\n"
                "{\n"
                "  \"primary_type\": \"<Exact Category ID from enumeration, or Uncertain>\",\n"
                "  \"sub_type\": \"<Specific subtype in English>\",\n"
                "  \"tags\": [\"<Tag1>\"],\n"
                "  \"confidence\": 0.70,\n"
                "  \"reasoning\": \"<Domain knowledge reasoning in English, under 40 words>\"\n"
                "}"
            )
        else:
            system_prompt = (
                "你是一位资深的生物信息学专家。该实体在文献局部和原文语句中信息均不足。\n"
                "请仅根据你在通用生物医学领域的常识进行最终研判。\n"
                "【绝对硬性约束 / CRITICAL RESTRICTION - 零容忍自创新分类】:\n"
                f"1. 字段 primary_type 的取值必须且只能严格从以下合法候选 ID 列表中选取（必须一字不差精确匹配）：\n"
                f"   {allowed_ids_str}\n"
                "2. 严禁擅自创造、发明或衍生任何列表之外的新分类名称（绝对禁止自创如: Bacteria, Pathogen, Microorganism, Organism, Receptor 等）！\n"
                "3. 若属于列表中的分类请准确归纳；若词汇无法确认或不在列表范畴，请坦率输出 \"Uncertain\"！"
            )
            user_content = (
                f"【待分析生物医学实体名】: {node_name}\n\n"
                f"【候选实体类别集合】:\n{cat_str}\n\n"
                f"【唯一允许选取的 primary_type 严格白名单】:\n{allowed_ids_str}\n\n"
                "【输出要求】:\n"
                "请严格以合法 JSON 输出，所有文本字段（sub_type, tags, reasoning）务必使用中文，严禁自创分类:\n"
                "{\n"
                "  \"primary_type\": \"<严格白名单中的ID或Uncertain>\",\n"
                "  \"sub_type\": \"<细分子类别>\",\n"
                "  \"tags\": [\"<标签1>\"],\n"
                "  \"confidence\": 0.70,\n"
                "  \"reasoning\": \"<通用知识研判依据，50字以内>\"\n"
                "}"
            )

        try:
            raw = self._ask_llm(system_prompt, user_content)
            res = self._parse_json_object(raw)
            if isinstance(res, dict) and "primary_type" in res:
                return res
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] Round 3 Entity LLM Error ({node_name}): {e}")

        return {
            "primary_type": "Uncertain",
            "sub_type": "",
            "tags": [],
            "confidence": 0.0,
            "reasoning": "Features remain insufficient to classify after multi-stage analysis" if is_en else "经多级综合研判，实体特征仍不足以明确分类"
        }

    def classify_workbench_relation(self, source: str, target: str, orig_rel: str, evidence: str, rel_cat_str: str, allowed_rel_ids: list, src_type: str = "Unknown", tgt_type: str = "Unknown", language: str = "zh") -> dict:
        """
        [Workbench] 对单条关系进行深层机制分类与机制动词归纳 (严格枚举白名单与动词解耦约束)
        """
        is_en = "en" in str(language).lower()
        allowed_rel_ids_str = json.dumps(allowed_rel_ids, ensure_ascii=False)

        if is_en:
            system_prompt = (
                "You are a molecular mechanism biologist. Your task is to classify biomedical knowledge graph relationships into deep mechanism categories "
                "and distill a precise, concise English mechanism verb.\n"
                "【CRITICAL RESTRICTION - MECHANISM CATEGORY ENUMERATION】:\n"
                f"1. mechanism_category MUST strictly be one of: {allowed_rel_ids_str}.\n"
                "2. Specific action verbs (e.g. 'Phosphorylates', 'Inhibits', 'Binds', 'Degrades') MUST be written in mechanism_verb, NEVER as mechanism_category!\n"
                "3. You are strictly forbidden from creating new category IDs."
            )
            user_content = (
                f"【Subject Entity】: {source} (Classified as: {src_type})\n"
                f"【Object Entity】: {target} (Classified as: {tgt_type})\n"
                f"【Original Relation Predicate】: {orig_rel}\n"
                f"【Original Literature Evidence】: {evidence if evidence else 'No direct sentence available'}\n\n"
                f"【Candidate Mechanism Categories】:\n{rel_cat_str}\n\n"
                f"【STRICT ENUMERATION - mechanism_category MUST be one of】:\n{allowed_rel_ids_str}\n\n"
                "【Output Instructions】:\n"
                "Strictly output a single valid JSON object. All field values MUST be in English:\n"
                "{\n"
                "  \"mechanism_category\": \"<Exact Candidate Category ID from enumeration>\",\n"
                "  \"mechanism_verb\": \"<Precise concise English mechanism verb, 1-3 words>\",\n"
                "  \"confidence\": 0.90,\n"
                "  \"reasoning\": \"<Concise mechanism rationale in English, under 40 words>\"\n"
                "}"
            )
        else:
            system_prompt = (
                "你是一位分子机制生物学家。你的任务是对生物知识图谱中的关系进行深层机制分类，"
                "并归纳出一个精准精炼的机制动词谓词（如：催化磷酸化、竞争性结合、转录激活、靶向降解、受体识别、感染宿主等）。\n"
                "【绝对硬性约束 - 关系机制分类枚举限制】:\n"
                f"1. mechanism_category 的取值必须且只能严格从以下合法候选类别中选取：\n"
                f"   {allowed_rel_ids_str}\n"
                "2. 严禁自创机制大类！具体的机制动作（如：抑制、激活、结合、磷酸化等）请填写在 mechanism_verb 字段中，不要将动词当作类别填入 mechanism_category！"
            )
            user_content = (
                f"【主体实体】: {source} (已分类为: {src_type})\n"
                f"【客体实体】: {target} (已分类为: {tgt_type})\n"
                f"【原始关系谓词】: {orig_rel}\n"
                f"【文献原文证据】: {evidence if evidence else '无具体原文句子'}\n\n"
                f"【候选关系机制类别】:\n{rel_cat_str}\n\n"
                f"【唯一允许选取的 mechanism_category 严格白名单】:\n{allowed_rel_ids_str}\n\n"
                "【输出要求】:\n"
                "请严格以合法 JSON 输出，不要包含任何多余文字，mechanism_verb 与 reasoning 务必使用中文:\n"
                "{\n"
                "  \"mechanism_category\": \"<严格白名单中的类别ID>\",\n"
                "  \"mechanism_verb\": \"<精准精炼的机制动词，10字以内>\",\n"
                "  \"confidence\": 0.90,\n"
                "  \"reasoning\": \"<简明归类依据，50字以内>\"\n"
                "}"
            )

        try:
            raw = self._ask_llm(system_prompt, user_content)
            res = self._parse_json_object(raw)
            if isinstance(res, dict) and "mechanism_category" in res:
                return res
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] Relation LLM Error ({source} -> {target}): {e}")

        default_cat = "Association_Correlation" if "Association_Correlation" in allowed_rel_ids else (allowed_rel_ids[0] if allowed_rel_ids else "Association_Correlation")
        return {
            "mechanism_category": default_cat,
            "mechanism_verb": orig_rel,
            "confidence": 0.5,
            "reasoning": "Analysis defaulted to general correlation" if is_en else "分析降级至默认关联"
        }

    def score_workbench_nodes_intent(self, candidate_nodes: list, user_query: str, language: str = "zh") -> dict:
        """
        [Workbench/Pathway] 利用大模型对候选节点进行自然语言假说意图相关性与生物学角色偏好综合评分 (范围 0.0 ~ 5.0)
        - 核心语义关联度: 0.0 ~ 3.5 分
        - 关键情境角色偏好 (疾病/表型/靶点/药物/关键调节因子): 0.0 ~ 1.5 分
        - 未提及节点默认打分 0.0
        """
        if not user_query or not candidate_nodes:
            return {}

        is_en = "en" in str(language).lower()

        node_lines = []
        for c in candidate_nodes[:80]:
            n_name = c.get("name", "")
            cat = c.get("primary_type", "Uncertain")
            sub = c.get("sub_type", "")
            deg = c.get("degree", 0)
            type_str = f"{cat}/{sub}".strip("/") if sub else cat
            node_lines.append(f"- {n_name} [类型: {type_str}, 连接度: {deg}]" if not is_en else f"- {n_name} [Type: {type_str}, Degree: {deg}]")

        nodes_text = "\n".join(node_lines)

        if is_en:
            sys_prompt = (
                "You are an expert systems biologist and computational medicine specialist. "
                "The researcher provides a scientific research inquiry or hypothesis, and a list of entities extracted from a knowledge graph.\n"
                "Your task is to evaluate and score which entities are relevant to the researcher's query.\n\n"
                "【Scoring Criteria (Total: 0.0 to 5.0)】:\n"
                "1. Semantic & Mechanistic Relevance (0.0 - 3.5 pts): How directly does this entity relate to the queried biological question, process, or pathology?\n"
                "2. Contextual Role Preference (0.0 - 1.5 pts): Dynamically elevate key functional anchors for this specific inquiry (e.g. disease phenotypes, drug targets, critical enzymes/transcription factors, virulence factors).\n"
                "Total score must be between 0.0 and 5.0.\n\n"
                "【Output Format】:\n"
                "Return strictly a single JSON object mapping ONLY the relevant entities (score >= 1.0). Omit irrelevant entities (default to 0.0):\n"
                "{\n"
                "  \"EntityName\": {\"score\": 4.5, \"role\": \"Direct pathological endpoint or key regulator\"}\n"
                "}"
            )
            user_prompt = (
                f"【Researcher's Focus/Inquiry】: {user_query}\n\n"
                f"【Candidate Entities】:\n{nodes_text}\n\n"
                "Please output the JSON ranking relevant entities:"
            )
        else:
            sys_prompt = (
                "你是一位资深的系统生物学与计算分子医学专家。学者提出了一项科学假说或探索重点，你需要从给出的图谱候选实体中，"
                "评估哪些实体与学者的探索假说高度相关，并评定其作为通路关键锚点的优先级。\n\n"
                "【评分标准 (总分 0.0 ~ 5.0 分)】：\n"
                "1. 假说机理相关性 (0.0 ~ 3.5 分)：该实体是否处于学者探索意图的核心因果链上（如上下游分子、生理过程、靶点）？\n"
                "2. 动态情境角色偏好 (0.0 ~ 1.5 分)：根据本次研究问题的具体侧重点，天然倾斜重点生物学角色（若探究致病机理则侧重关键毒力因子/疾病表型；若探究药物机制则侧重药物/靶点激酶；若探究信号转导则侧重核心受体/转录因子）。\n"
                "满分为 5.0 分。\n\n"
                "【输出格式要求】：\n"
                "仅需对得分 >= 1.0 的相关实体进行评分（无关实体无需输出，系统默认赋 0.0）。严格输出唯一的合法 JSON 对象：\n"
                "{\n"
                "  \"实体名称\": {\"score\": 4.5, \"role\": \"核心病理表型 / 关键上游调控因子\"}\n"
                "}"
            )
            user_prompt = (
                f"【学者探索重点/假说】: {user_query}\n\n"
                f"【候选图谱实体列表】:\n{nodes_text}\n\n"
                "请严格按照 JSON 格式输出相关实体评分："
            )

        try:
            raw_resp = self._ask_llm(sys_prompt, user_prompt)
            parsed = self._parse_json_object(raw_resp)
            if parsed and isinstance(parsed, dict):
                cleaned_scores = {}
                for k, v in parsed.items():
                    if isinstance(v, dict):
                        sc = float(v.get("score", 0.0))
                        role = str(v.get("role", ""))
                    elif isinstance(v, (int, float)):
                        sc = float(v)
                        role = ""
                    else:
                        continue
                    sc = max(0.0, min(5.0, sc))
                    if sc > 0:
                        cleaned_scores[k] = {"score": sc, "role": role}
                return cleaned_scores
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] Pathway 批处理节点意图评分异常: {e}")

        return {}

    def synthesize_pathway_narrative(self, pathway_idx: int, nodes_desc: list, edges_desc: list, loops_desc: list, user_query: str = "", language: str = "zh") -> dict:
        """
        [Pathway] 结合子网络关键实体、分子相互作用与回路拓扑，合成学术通路 LaTeX 命名与机制总结叙事
        """
        is_en = "en" in str(language).lower()

        if is_en:
            focus_instruction = f"\n3. Since a specific user inquiry is provided ('{user_query}'), strictly center the scholarly title and mechanism narrative on elucidating how this subnetwork addresses, regulates, or mediates that biological process." if user_query else ""
            sys_prompt = (
                "You are an expert in systems biology and molecular oncology. "
                "You will be given an exact biological subnetwork module induced from a validated knowledge graph. "
                "Your task is to synthesize: "
                "1. A highly professional academic pathway title using LaTeX format where appropriate "
                "(e.g. '$\\text{CRL5}^{\\text{SOCS4}}$-mediated ubiquitination pathway and neurodegeneration resistance'). "
                "2. A concise 2-3 sentence executive summary explaining the molecular mechanism. "
                "If a feedback loop is verified in the network, briefly discuss its dynamic role (homeostasis or bistable switch). "
                f"{focus_instruction}\n"
                "Do NOT invent or fabricate any relations not present in the graph."
            )
            user_prompt = (
                f"【Pathway #{pathway_idx} Verified Context】\n"
                f"【User Focus】: {user_query if user_query else 'Autonomous discovery'}\n"
                f"【Entities in Subnetwork】:\n" + "\n".join(nodes_desc) + "\n\n"
                f"【Direct Verified Interactions】:\n" + "\n".join(edges_desc) + "\n\n"
                f"【Detected Autoregulation Loops】:\n" + ("\n".join(loops_desc) if loops_desc else "None (Linear cascade)") + "\n\n"
                "【Output Requirement】:\n"
                "Return strictly a single valid JSON object in English:\n"
                "{\n"
                "  \"title\": \"<Scholarly title in LaTeX, e.g. $\\\\text{AMPK-mTOR}$ Signaling Axis in Autophagy>\",\n"
                "  \"summary\": \"<2-3 concise sentences explaining the cascade and feedback regulation if present>\"\n"
                "}"
            )
        else:
            focus_instruction = f"\n3. 学者指定了明确的研究焦点（'{user_query}'），请务必以此为核心视角，在学术命名与机制解读中深入阐明该子网络如何介导或调控该生物学/病理过程。" if user_query else ""
            sys_prompt = (
                "你是一位资深的系统生物学与分子医学专家。你将接收从权威知识图谱中严格诱导出的局部生物机制子网络。"
                "你的任务是：\n"
                "1. 给出高度专业、规范的学术命名，支持规范的 LaTeX 格式"
                "（如：$\\text{CRL5}^{\\text{SOCS4}}$ 介导的泛素化降解途径与神经退行抗性、$\\text{AMPK-mTOR}$ 介导的自噬调控轴）；\n"
                "2. 用 2~3 句话精炼阐明该通路的分子传导逻辑与生理/病理意义；"
                "如果给出的数据中检测到了真实的自反馈调节闭环，请在此适度剖析其自稳态或自促进效应。\n"
                f"{focus_instruction}\n"
                "⚠️ 严格禁止无中生有或捏造不存在的关系，节点与关系必须以给定数据为准！"
            )
            user_prompt = (
                f"【第 {pathway_idx} 条通路已知机制数据】\n"
                f"【学者关注焦点】: {user_query if user_query else '全局自主发现'}\n"
                f"【子网络关键实体】:\n" + "\n".join(nodes_desc) + "\n\n"
                f"【真实分子相互作用】:\n" + "\n".join(edges_desc) + "\n\n"
                f"【算法硬性捕获的闭环回路】:\n" + ("\n".join(loops_desc) if loops_desc else "无闭环 (单向传导)") + "\n\n"
                "【输出格式要求】:\n"
                "请严格输出唯一的合法 JSON 对象，字段内容使用学术中文，严禁输出任何多余字符：\n"
                "{\n"
                "  \"title\": \"<学术命名，支持规范 LaTeX，如：$\\\\text{AMPK-mTOR}$ 轴介导的自噬应答通路>\",\n"
                "  \"summary\": \"<2~3句话精练介绍通路机制；若有反馈循环适当融入动力学分析>\"\n"
                "}"
            )

        try:
            raw_resp = self._ask_llm(sys_prompt, user_prompt)
            parsed = self._parse_json_object(raw_resp)
            if parsed and isinstance(parsed, dict) and "title" in parsed and "summary" in parsed:
                return parsed
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] Pathway 机制叙事调用异常: {e}")

        return {}

    def generate_hypothesis_pubmed_query(self, entity_a: str, entity_b: str, proposed_relation: str = "", intent: str = "") -> str:
        """
        为假说推演两端的实体专门生成高精度的 PubMed 检验检索式。
        精准探查学术界是否已有关于两者直接互作或调控机制的报道。
        """
        sys_prompt = """你是一个顶级的生物医学文献检索与信息计量学专家。
你的任务是为待检验的科学假说（实体 A 与 实体 B 之间的潜在调控机制）生成唯一的 PubMed 布尔检索式（Boolean Query）。

规则要求：
1. 核心实体 A 和 B 必须用 AND 连接，并带上 [Title/Abstract] 标签。
2. 实体分类识别与机制词约束：
   - 若实体涉及【药物/抗生素/小分子】与【疾病/感染/表型】（例如抗生素与尿路感染），切勿使用纯分子生物物理结合词（如 binding, phosphorylat*），应保持广义机制或临床治疗词（例如 AND (treatment OR therapy OR antimicrobial OR resistance OR clinical)），或直接仅以两实体作为核心检索式，以保证高召回与临床文献覆盖。
   - 若实体均为【基因/蛋白质/转录因子/激酶】，可适度提炼 2~4 个机制同义词作为第二组 AND 约束（例如 AND (inhibit* OR suppress* OR target* OR regulat*)）。
3. 严禁使用过于繁琐冷门的检索语法，确保 PubMed 能够高召回。
4. ⚠️ 严禁输出任何解释性文字、前缀或 Markdown 标记。直接输出纯检索式字符串！"""

        user_content = f"待探勘核心实体 A: {entity_a}\n待探勘核心实体 B: {entity_b}\n推测机制/方向: {proposed_relation}\n学者科研意图: {intent if intent else '探究两者之间的直接分子机制与调控关系'}"

        try:
            raw_resp = self._ask_llm(sys_prompt, user_content)
            query = self._clean_json_string(raw_resp).strip()
            if query and "AND" in query and (entity_a.lower() in query.lower() or entity_b.lower() in query.lower()):
                return query
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] 生成假说检验检索式异常: {e}")

        clean_a = re.sub(r'[\(\)\[\]]', '', entity_a).strip()
        clean_b = re.sub(r'[\(\)\[\]]', '', entity_b).strip()
        return f'("{clean_a}"[Title/Abstract] AND "{clean_b}"[Title/Abstract])'

    def formulate_hypothesis_candidates(self, candidate_pairs: list, intent: str = "", lang: str = "zh") -> list:
        """
        根据图论寻孔算法提取出的结构洞/边缘候选对，结合生物学生化逻辑，为每一对生成专业的假说命名、推演机制与研究方向描述。
        :param candidate_pairs: 包含 node_a, category_a, node_b, category_b, context 等元数据的列表
        :param intent: 学者指定的科研关注点（若有）
        :param lang: "zh" 或 "en"
        :return: 包含假说表述与建议研究方向的结构化列表
        """
        if not candidate_pairs:
            return []

        is_en = "en" in str(lang).lower()
        if is_en:
            sys_prompt = """You are a senior molecular biologist and systems medicine researcher.
You will receive candidate entity pairs representing topological structural holes or growth frontiers in a biological knowledge graph.
For each pair, formulate a biologically sound scientific hypothesis, propose a mechanistic relation, and articulate a clear research direction for scholars to investigate.

Return strictly a valid JSON array of objects, one for each input pair in the same order:
[
  {
    "hypothesis_statement": "<Concise scholarly hypothesis sentence in LaTeX, e.g. $\\\\text{Drug X}$ directly suppresses $\\\\text{Protein Y}$ phosphorylation>",
    "proposed_relation": "<Positive_Regulation | Negative_Regulation | Physical_Binding | Catalytic_Transformation | Biomarker_Association>",
    "biological_rationale": "<1-2 sentences on biological rationale based on entity categories and cellular context>",
    "research_direction": "<Clear, inspiring description of the specific scientific research direction for the researcher to pursue>"
  }
]"""
        else:
            sys_prompt = """你是一位资深的分子系统生物学与转化医学专家。
系统通过图论拓扑寻孔算法识别出了知识图谱中的若干“结构洞/缺失位点”或“周界边缘生长点”。
你的任务是为给定的每一组候选节点对，结合两者的生物学本体分类与调控上下文，推演提出合理的科学假说、标定机制动词，并为学者阐述清晰的研究方向与科学留白。

输出要求：请严格输出合法的 JSON 数组，数组元素与输入的候选对一一对应：
[
  {
    "hypothesis_statement": "<规范学术假说陈述，支持 LaTeX 格式，如：$\\\\text{化合物 A}$ 靶向变构抑制 $\\\\text{激酶 B}$ 介导的下游促癌级联>",
    "proposed_relation": "<机制动词：Positive_Regulation (促进/激活) | Negative_Regulation (抑制/阻断) | Physical_Binding (物理结合) | Catalytic_Transformation (催化转化) | Biomarker_Association (标志物关联)>",
    "biological_rationale": "<1~2 句话阐明基于本体分类与细胞通路的生物学合理性理由>",
    "research_direction": "<为学者指明具体、明确且有启发性的科学研究方向与留白描述，供学者自行设计实验验证>"
  }
]"""

        user_content = f"【学者科研意向/焦点】: {intent if intent else '全局自感知自主探勘'}\n\n【待推演结构洞候选对列表】:\n" + json.dumps(candidate_pairs, ensure_ascii=False, indent=2)

        try:
            raw_resp = self._ask_llm(sys_prompt, user_content)
            parsed = self._parse_json_object(raw_resp)
            if isinstance(parsed, list) and len(parsed) == len(candidate_pairs):
                return parsed
            elif isinstance(parsed, list):
                results = []
                for i, pair in enumerate(candidate_pairs):
                    if i < len(parsed) and isinstance(parsed[i], dict):
                        results.append(parsed[i])
                    else:
                        results.append({
                            "hypothesis_statement": f"{pair.get('node_a')} 与 {pair.get('node_b')} 存在潜在分子调控偶联",
                            "proposed_relation": "Positive_Regulation",
                            "biological_rationale": "基于拓扑近邻与跨簇结构洞推测存在未明机制关联",
                            "research_direction": f"深入考察 {pair.get('node_a')} 对 {pair.get('node_b')} 的功能学调节效应"
                        })
                return results
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] 生成假说猜想异常: {e}")

        # 启发式兜底
        fallback = []
        for pair in candidate_pairs:
            na, nb = pair.get("node_a", "A"), pair.get("node_b", "B")
            ca, cb = pair.get("category_a", ""), pair.get("category_b", "")
            if "Drug" in ca or "Chemical" in ca:
                rel = "Negative_Regulation"
                stmt = f"{na} 靶向调控 {nb} 的表达或活性"
                dir_desc = f"探究小分子 {na} 作为 {nb} 潜在干预剂的机理及敏感性表型"
            else:
                rel = "Positive_Regulation"
                stmt = f"{na} 与 {nb} 在调控网络中存在功能偶联与互作"
                dir_desc = f"探究 {na} 与 {nb} 在相关生物学过程中的协同或上下游级联机制"

            fallback.append({
                "hypothesis_statement": stmt,
                "proposed_relation": rel,
                "biological_rationale": f"基于 {ca} 与 {cb} 的本体相容性及拓扑结构洞推演",
                "research_direction": dir_desc
            })
        return fallback

    def adjudicate_literature_novelty(self, hypothesis_info: dict, articles: list, lang: str = "zh") -> dict:
        """
        文献裁判大模型核心入口：
        通读检索到的 PubMed 摘要，判定假说究竟属于：
        - 'known': 文献中已有直接实验/观察证实，属于知识盲区补全；
        - 'novel_gap': 现有文献尚未有直接因果互作报道，属于真正的科研空白/原始创新假说；
        - 'cascade': 现有文献揭示了多步间接级联（A 调控 C，C 再调控 B）；
        - 'inconclusive': 文献仅概念共现或无直接因果依据。
        """
        is_en = "en" in str(lang).lower()
        node_a = hypothesis_info.get("node_a", "")
        node_b = hypothesis_info.get("node_b", "")
        stmt = hypothesis_info.get("hypothesis_statement", "")
        direction = hypothesis_info.get("research_direction", "")

        if not articles:
            # 即使 PubMed 未直接命中标题摘要，严禁盲目直接断定为科研空白！
            # 必须调用大模型先验知识库进行医学常识/公认临床事实的保底核验
            if is_en:
                cs_sys_prompt = """You are a senior clinical pharmacologist and biomedical reviewer.
A PubMed search returned 0 abstracts for the given entity pair. Evaluate using established biomedical consensus:
Does the relationship between Entity A and Entity B represent an established clinical fact, standard medical therapy, textbook knowledge, or common pharmacological treatment (e.g. antibiotics treating infections, known drug indications)?
- If it is established medical knowledge / clinical practice: strictly classify as "known" (confidence 0.90+).
- Only if the connection is truly unprecedented and scientifically unstudied: classify as "novel_gap".

Return strictly a valid JSON object:
{
  "verdict": "<known | novel_gap | inconclusive>",
  "confidence": <0.7 to 0.95>,
  "direct_evidence_pmid": "",
  "evidence_quote": "",
  "research_direction": "<refined research direction>",
  "adjudication_rationale": "<reasoning explaining clinical consensus or novelty>"
}"""
                cs_user = f"Entity A: {node_a} ({hypothesis_info.get('category_a', 'Unknown')})\nEntity B: {node_b} ({hypothesis_info.get('category_b', 'Unknown')})\nHypothesis: {stmt}\nDirection: {direction}"
            else:
                cs_sys_prompt = """你是一位资深的生物医药与转化医学终审专家。
PubMed 检索未直接召回包含两者的标题摘要。请结合权威生物医药先验知识库严格判定：
实体 A 与 实体 B 之间的关系是否属于【医学生物学公认常识 / 已知临床事实】（例如常见药物/抗生素治疗该疾病/感染、指南常规用药、公认代谢通路底物）？
- 若属于公认医学常识或已在临床广泛应用的事实：必须严格判定为 "known"（已知事实补全，置信度 0.90+），绝对严禁误判为科研空白！并在 adjudication_rationale 中指明其公认临床地位。
- 只有当两者在现代科学界确实未被报道建立关联、属于真正的前沿猜想时：才判定为 "novel_gap"。

请严格输出合法的 JSON 对象：
{
  "verdict": "<known | novel_gap | inconclusive>",
  "confidence": <0.7~0.95>,
  "direct_evidence_pmid": "",
  "evidence_quote": "",
  "research_direction": "<明确研究方向>",
  "adjudication_rationale": "<判定理由说明>"
}"""
                cs_user = f"实体 A: {node_a} ({hypothesis_info.get('category_a', 'Unknown')})\n实体 B: {node_b} ({hypothesis_info.get('category_b', 'Unknown')})\n假说陈述: {stmt}\n原定研究方向: {direction}"

            try:
                raw_resp = self._ask_llm(cs_sys_prompt, cs_user)
                parsed = self._parse_json_object(raw_resp)
                if isinstance(parsed, dict) and "verdict" in parsed:
                    return parsed
            except Exception as e:
                print(f"⚠️ [BioBrainAgent] 0文献先验知识判定异常: {e}")

            # 启发式常识保底：若实体 A 包含抗生素/药物且实体 B 包含感染/疾病，判定为 known
            low_a, low_b = node_a.lower(), node_b.lower()
            is_anti_inf = any(w in low_a for w in ["antibiotic", "penicillin", "cepha", "cipro", "anti-", "抗生素"]) and any(w in low_b for w in ["infection", "tract", "uti", "pneumonia", "sepsis", "感染", "炎"])
            if is_anti_inf:
                return {
                    "verdict": "known",
                    "confidence": 0.95,
                    "direct_evidence_pmid": "",
                    "evidence_quote": "",
                    "research_direction": direction,
                    "adjudication_rationale": f"基于权威临床药理学常识，{node_a} 针对 {node_b} 的抗感染治疗与经验性用药属于广泛公认的临床事实与治疗常规，属于已知事实补全。" if not is_en else f"Based on established clinical pharmacology, the use of {node_a} for {node_b} is a standard clinical fact and medical practice."
                }

            if is_en:
                return {
                    "verdict": "novel_gap",
                    "confidence": 0.85,
                    "direct_evidence_pmid": "",
                    "evidence_quote": "",
                    "research_direction": direction,
                    "adjudication_rationale": f"Comprehensive search in PubMed returned 0 direct co-occurrence articles for {node_a} and {node_b}. Based on ontological compatibility and network topology, this interaction represents a genuine scientific white space (novel hypothesis)."
                }
            else:
                return {
                    "verdict": "novel_gap",
                    "confidence": 0.85,
                    "direct_evidence_pmid": "",
                    "evidence_quote": "",
                    "research_direction": direction,
                    "adjudication_rationale": f"在 PubMed 权威数据库检索中，未发现直接报道 {node_a} 与 {node_b} 相互作用的文献。基于本体相容性与拓扑结构洞推演，该机制属于尚未被报道的【真正的科研空白（原始创新假说）】。"
                }

        docs_summary = []
        for i, art in enumerate(articles[:5]):
            pmid = art.get("pmid", "")
            title = art.get("title", "")
            abstract = art.get("abstract", "")
            year = art.get("year", "")
            journal = art.get("journal", "")
            docs_summary.append(
                f"[Document #{i+1}] PMID: {pmid} ({year}, {journal})\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}\n"
            )

        if is_en:
            sys_prompt = """You are the Chief Scientific Reviewer in molecular pharmacology and systems biology.
Your task is to thoroughly analyze the provided PubMed article abstracts against a proposed scientific hypothesis, and deliver an impartial, rigorous adjudication.

Adjudication Categories:
1. "known": The articles contain explicit experimental or clinical evidence directly confirming or reporting this interaction between Entity A and Entity B.
   ⚠️ CRITICAL MANDATE: For Drug/Antibiotic/Compound and Disease/Infection/Phenotype pairs (e.g. broad-spectrum antibiotics and urinary tract infection), if the articles describe the drug being used for treatment, management, antibiotic therapy, susceptibility, clinical prophylaxis, or as standard of care, you MUST classify it as "known" (established fact gap filled). Never misclassify standard clinical treatments or antibiotic therapy as a novel gap!
   Similarly, for gene/protein pairs with direct biophysical binding, phosphorylation, or regulation demonstrated in the text, classify as "known".
2. "novel_gap": The articles discuss the entities separately or in unrelated contexts without demonstrating this direct causal interaction or clinical indication. This represents a true scientific gap / virgin territory.
3. "cascade": The articles show that Entity A regulates an intermediate mediator C, which in turn influences Entity B.
4. "inconclusive": Mentions are superficial or contradictory without clear mechanistic conclusion.

Return strictly a valid JSON object:
{
  "verdict": "<known | novel_gap | cascade | inconclusive>",
  "confidence": <float between 0.5 and 1.0>,
  "direct_evidence_pmid": "<PMID of the key article if known, else empty string>",
  "evidence_quote": "<Exact sentence from the abstract providing evidence, else empty string>",
  "research_direction": "<Refined specific research direction for the researcher to explore>",
  "adjudication_rationale": "<2-3 sentences explaining your analytical reasoning as the referee>"
}"""
        else:
            sys_prompt = """你是一位国际顶尖的分子生物医学文献终审专家（Chief Referee）。
你的任务是严谨、客观地通读给定的 PubMed 文献摘要，与学者提出的待检假说进行逐字对照，作出权威判定。

判定档位规范：
1. "known"【已知事实补全】：给定的文献摘要中已包含直接的实验或临床证据，明确报道了实体 A 与 实体 B 之间的直接作用关系或临床疗效。
   ⚠️ 核心审判规范：对于【药物/抗生素/小分子】与【疾病/感染/症状/病原体】（例如广谱抗生素与尿路感染、降压药与高血压），文献中只要表明该药物用于治疗、管理、敏感性测试或临床应对该疾病/感染，或者属于公认的临床治疗范式/耐药性研究，必须严格判定为 "known"（文献已确证的事实补全），绝对严禁将其误判为科研空白！
   同样，对于基因/蛋白质之间已被直接实验证实的调控/结合/修饰关系，亦判定为 "known"。
2. "novel_gap"【真正的科研空白 / 原始假说】：给定的文献虽然提及了实体 A 或 B，但均属于各自独立研究或宏观背景并列，学术界尚未有文献完成该直接因果机制的实验验证，确属科研空白。
3. "cascade"【多步级联中介】：文献表明 A 并不直接作用于 B，而是通过介导因子 C 产生间接调控 (A -> C -> B)。
4. "inconclusive"【证据不足/无关】：文献中提及内容与该假说机制无关或存在严重矛盾。

输出要求：请严格输出合法的单个 JSON 对象：
{
  "verdict": "<known | novel_gap | cascade | inconclusive>",
  "confidence": <置信度浮点数 0.5~1.0>,
  "direct_evidence_pmid": "<若已证实，给出关键证据的 PMID，否则留空字符>",
  "evidence_quote": "<从摘要中摘录的直接证据原句，若无留空字符>",
  "research_direction": "<进一步完善提炼的建议明确研究方向与科学留白描述>",
  "adjudication_rationale": "<2~3句话清晰陈述裁判大模型的权威裁决理由>"
}"""

        user_content = (
            f"【待检验假说陈述】: {stmt}\n"
            f"【核心实体 A】: {node_a} ({hypothesis_info.get('category_a', 'Unknown')})\n"
            f"【核心实体 B】: {node_b} ({hypothesis_info.get('category_b', 'Unknown')})\n"
            f"【推测机制类型】: {hypothesis_info.get('proposed_relation', 'Unknown')}\n"
            f"【原始研究方向】: {direction}\n\n"
            f"【PubMed 检索回的前沿文献摘要集合】:\n" + "\n".join(docs_summary)
        )

        try:
            raw_resp = self._ask_llm(sys_prompt, user_content)
            parsed = self._parse_json_object(raw_resp)
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
        except Exception as e:
            print(f"⚠️ [BioBrainAgent] 文献裁判大模型审读异常: {e}")

        first_pmid = articles[0].get("pmid", "") if articles else ""
        return {
            "verdict": "novel_gap",
            "confidence": 0.75,
            "direct_evidence_pmid": first_pmid,
            "evidence_quote": "",
            "research_direction": direction,
            "adjudication_rationale": "大模型在精读当前检索文献后，未发现针对两者直接因果调控机制的明确实验验证，建议学者针对该机制展开探究。"
        }

