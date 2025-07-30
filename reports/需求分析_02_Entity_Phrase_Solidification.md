### **需求文档：实体与短语的半自动化固化流程**

**项目目标：** 从筛选后的中国新闻数据中，通过“算法发现 + 人工审核”的闭环流程，构建一个高质量的实体/短语合并规则集，并将其应用于文本，为后续的主题建模生成一个语义单元被“固化”的、标准化的语料库。

**输入文件：** `data_processed/final_china_news.csv`
*   **格式：** CSV文件，约17万行。
*   **列名：** `DATE`, `CONTENT`
*   **数据特点：** 已初步筛选，但未经深度清洗。包含原始大小写、标点、数字、特殊字符等。

---

### **阶段一：候选短语的自动化发现 (The Discovery Engine)**

**此阶段总目标：** 从17万条新闻中，利用三种互补的算法，大规模地、无偏见地挖掘出所有潜在的、有意义的多词短语（包括命名实体和专业术语），形成一个待审核的“候选池”。

#### **1.1 前置任务：文本预分词**

*   **逻辑与目的：**
    为了让后续的`Gensim Phrases`模型能够高效运行，我们需要先将长文本切分成单词列表。此处的处理非常轻量，目的是保留尽可能多的原始短语结构，包括大小写和特殊字符组合。
*   **输入：** `final_china_news.csv` 中的 `CONTENT` 列。
*   **动作：**
    1.  **加载数据：** 使用Pandas读取CSV文件。
    2.  **创建分词器：** 加载spaCy模型 (`en_core_web_md`)。
    3.  **批量分词：** 使用`nlp.pipe`遍历所有文章。对于每个`token`，只进行最基础的过滤：只要它既不是纯标点符号(`is_punct`)，也不是纯空格(`is_space`)，就保留其**原始文本** (`token.text`)。
    4.  **产出内存对象：** 生成一个Python列表，其中每个元素是对应一篇文章的`token`列表。我们称之为 `raw_token_lists`。
*   **预估时间：** 15 - 40 分钟。

#### **1.2 候选发现策略一：统计共现 (Gensim `Phrases`)**

*   **逻辑与目的：**
    利用统计学原理，找出那些“粘合度”很高的、频繁在一起出现的单词组合。此方法对发现**专业术语**（如 `interest rate`, `stock market`）和高频实体特别有效。
*   **输入：** 上一步生成的 `raw_token_lists`。
*   **动作：**
    1.  **标准化输入：** 创建一个`raw_token_lists`的副本，并将其中所有`token`转为小写，作为`Phrases`模型的输入（因为它不区分大小写，这样做能聚合频率）。
    2.  **训练模型：** 初始化`gensim.models.Phrases`模型。设置合理的参数，例如`min_count=20`（至少出现20次），`threshold=10`（一个标准的粘合度阈值）。
    3.  **导出候选：** 使用模型的`export_phrases()`方法，提取出所有被发现的短语及其对应的“粘合度”得分。
*   **阶段产出：** 一个包含（短语，得分）的列表，例如 `gensim_candidates`。

#### **1.3 候选发现策略二：命名实体识别 (spaCy NER)**

*   **逻辑与目的：**
    利用预训练的深度学习模型，直接、精准地识别出文本中的人名、地名、组织机构名等**命名实体**。此方法是发现高质量专有名词的核心。
*   **输入：** 原始的 `CONTENT` 列（必须保留原始大小写）。
*   **动作：**
    1.  **批量识别：** 使用`nlp.pipe`遍历所有原始文章内容。
    2.  **提取目标实体：** 对于每篇文章的`doc.ents`，只提取我们感兴趣的实体类型（如`ORG`, `PERSON`, `GPE`, `NORP`, `PRODUCT`等）。
    3.  **计数与收集：** 将提取出的实体文本（转为小写以用于后续聚合）存入一个`collections.Counter`中，以统计它们的出现频率。
*   **阶段产出：** 一个包含（小写实体短语，频率）的字典，例如 `ner_candidates`。

#### **1.4 候选发现策略三：语法结构分析 (spaCy `noun_chunks`)**

*   **逻辑与目的：**
    作为前两种方法的补充，利用语法分析器找出所有符合“名词短语”结构的部分。这能捕获到一些非标准实体、描述性短语（如`"a sharp decline in prices"`）以及NER模型可能遗漏的组合。
*   **输入：** 原始的 `CONTENT` 列。
*   **动作：**
    1.  **批量分析：** 使用`nlp.pipe`遍历所有原始文章内容。
    2.  **提取名词短语：** 对于每篇文章的`doc.noun_chunks`，提取其文本。
    3.  **清洗与计数：** 对提取出的名词短语进行初步清洗（如移除首尾的冠词`a/an/the`），然后转为小写，存入一个`collections.Counter`中进行计数。
*   **阶段产出：** 一个包含（小写名词短语，频率）的字典，例如 `noun_chunk_candidates`。
*   **预估时间（1.2 - 1.4）：** 三种策略并行或串行处理17万条新闻，总耗时预计在 **1.5 - 4 小时**之间，其中NER和名词短语提取是主要耗时部分。

---

### **阶段二：候选池的整合与排序 (The Funnel)**

**此阶段总目标：** 将三个来源的候选短语汇集到一个地方，进行清洗、去重，并根据其重要性进行排序，最终生成一份高质量、待审核的CSV文件。

#### **2.1 数据框的构建与合并**

*   **逻辑与目的：**
    将三种不同格式的候选列表，统一到一个Pandas DataFrame中，方便进行后续的筛选和排序。
*   **输入：** `gensim_candidates`, `ner_candidates`, `noun_chunk_candidates`。
*   **动作：**
    1.  **分别创建DataFrame：** 为每个候选列表创建一个DataFrame，包含`candidate_phrase`, `frequency`, `score` (如果有), `source` (来源, e.g., 'Gensim', 'NER', 'NounChunk')等列。
    2.  **计算缺失频率：** 为Gensim发现的短语计算其在整个语料库中的出现频率。
    3.  **合并：** 使用`pd.concat()`将三个DataFrame合并成一个大的`df_candidates`。

#### **2.2 清洗、过滤与排序**

*   **逻辑与目的：**
    对合并后的候选池进行“提纯”，移除明显无用的条目，并按重要性排序，以极大减轻后续手动审核的负担。
*   **输入：** `df_candidates`。
*   **动作：**
    1.  **清洗短语文本：** 对`candidate_phrase`列进行标准化，例如移除多余的空格。
    2.  **过滤低频项：** 移除`frequency`低于某个阈值（例如 < 10）的行。
    3. **去重与优先级处理：**
        *   按`candidate_phrase`列去重。
        *   在去重前，先按`source`列排序（例如，优先级：NER > Gensim > NounChunk），确保对于同一个短语，我们保留信息质量最高的那条记录。
    4. **最终排序：** 按`frequency`进行降序排序，作为主要排序依据。
    5. **添加审核列：** 新增两列`action`和`standard_form`，并置为空，等待人工填写。
*   **阶段产出：** 一个名为 `candidate_phrases_for_review.csv` 的文件。
*   **预估时间：** 数据处理速度很快，预计在 **5 - 15 分钟**内。

---

### **阶段三：人工审核与决策 (The Human-in-the-Loop)**

**此阶段总目标：** 由你作为领域专家，对经过排序的候选列表进行审核，做出“合并”、“设为停用词”或“忽略”的决策。

#### **3.1 手动填写决策文件**

*   **逻辑与目的：**
    将你的领域知识注入到流程中。你的决策将直接决定最终的实体合并规则和自定义停用词列表。
*   **输入：** `candidate_phrases_for_review.csv`。
*   **动作：**
    1.  **打开CSV文件：** 使用Excel, Google Sheets或任何你习惯的表格工具。
    2.  **逐行决策：** 至少审核排名前2000-3000的短语。在`action_code`列填写数字代码：
        *   **填 `1` 代表 `Merge` (合并)**
        *   **填 `2` 代表 `Stopword` (设为停用词)**
        *   **填 `3` (或留空) 代表 `Ignore` (忽略)**

    3.  **决策指南：如何区分 `Stopword` 和 `Ignore`？**

        *   **什么样的词组应该标记为 `2` (Stopword)？**
            *   **定义：** 这些词组本身是**语法结构的一部分**，高频出现但**不携带任何特定主题信息**，它们的存在会干扰主题模型。将它们标记为`Stopword`后，程序会自动将其拆分，并把每个组成部分都加入到自定义停用词列表中。
            *   **核心原则：** “这个词组对区分任何一个主题都没有帮助，反而会成为所有主题的背景噪音。”
            *   **典型例子：**
                *   `"according to"`
                *   `"in addition to"`
                *   `"at the same time"`
                *   `"as well as"`
                *   `"a lot of"`
                *   `"for example"`
                *   `"on the other hand"`

        *   **什么样的词组应该标记为 `3` (Ignore)？**
            *   **定义：** 这些词组是**有具体意义的普通短语**，但你认为它们**不需要被“固化”成一个超级词**，也不需要被当作停用词移除。它们应该被正常地拆分成单个词，并进入后续的分析流程。
            *   **核心原则：** “这个短语有意义，但不够‘特殊’或‘固定’到需要合并成一个整体。让它自然地被处理就好。”
            *   **典型例子：**
                *   `"last year"` (有时间意义，但不应固化)
                *   `"many people"` (有数量意义，但太泛)
                *   `"good news"` (有情感意义，但不应固化)
                *   `"next step"`
                *   `"a long time"`
                *   `"high level"`

    4.  **填写`standard_form`列**：
        *   只有当你在`action_code`列填写了`1` (Merge)时，才需要在`standard_form`列填写你希望它合并成的标准形式（例如 `peoples_bank_of_china`）。
        *   对于`action_code`为`2`或`3`的行，`standard_form`列可以留空。

*   **阶段产出：** 一个被你填写、包含决策信息的CSV文件，我们称之为 `candidate_phrases_reviewed.csv`。
*   **预估时间：** 完全取决于你的投入。高效地审核2000个短语，可能需要 **2 - 5 小时**的专注工作。这是一个一次性的、高价值的投资。

#### **3.2 手动创建专家规则集**

*   **逻辑与目的：**
    弥补算法的不足。有些合并规则极其重要，但算法可能发现不了（例如，缩写与全称的映射，不同实体的概念性聚类）。这个环节让你拥有完全的控制权，可以强制添加任何你认为必要的规则。
*   **输入：** 你的大脑（领域知识）。
*   **动作：**
    1.  **创建专家规则文件：** 在你的项目文件夹中，创建一个新的、独立的CSV文件，命名为 `expert_rules.csv`。
    2.  **定义文件结构：** 这个CSV文件包含三列：`phrase_to_merge`, `standard_form`, `rule_type`。
    3.  **手动填写规则：** 在这个文件中，主动添加那些你认为必须存在的合并规则。这通常包括以下几类：
        *   **缩写与全称 (Acronyms)：**
            *   `phrase_to_merge`: `pboc`, `standard_form`: `peoples_bank_of_china`, `rule_type`: `Acronym`
            *   `phrase_to_merge`: `csrc`, `standard_form`: `china_securities_regulatory_commission`, `rule_type`: `Acronym`
        *   **不同名称的统一 (Synonyms)：**
            *   `phrase_to_merge`: `ministry of commerce`, `standard_form`: `mofcom`, `rule_type`: `Synonym` (假设你选择缩写为标准形式)
        *   **概念性聚类 (Conceptual Clustering)：**
            *   `phrase_to_merge`: `alibaba`, `standard_form`: `china_tech_giant`, `rule_type`: `Concept`
            *   `phrase_to_merge`: `tencent`, `standard_form`: `china_tech_giant`, `rule_type`: `Concept`
        *   **常见的拼写变体/错误 (Typos/Variants)：**
            *   `phrase_to_merge`: `shenzen`, `standard_form`: `shenzhen`, `rule_type`: `Typo`
*   **阶段产出：** 一个完全由你手动控制的、高优先级的规则文件：`expert_rules.csv`。
*   **预估时间：** 这取决于你的知识储备和研究深度。初次创建可能需要 **1 - 3 小时**，后续可以随时回来补充。

---

### **阶段四：规则生成与最终应用 (The Final Application)**

**此阶段总目标：** 将来自**两个来源**（算法发现+人工审核，以及专家手动创建）的规则合并，并应用到文本上，完成最终的“实体固化”。

#### **4.1 从决策文件生成规则词典 (修订版)**

*   **逻辑与目的：**
    整合所有规则，并确保专家规则拥有更高的优先级。
*   **输入：**
    1. `candidate_phrases_reviewed.csv` (来自被动审核)
    2. `expert_rules.csv` (来自主动添加)
*   **动作：**
    1.  **初始化规则容器：** 创建一个空的合并词典 `merge_dict` 和一个空的停用词集合 `new_stopwords`。
    2.  **首先，加载专家规则：**
        *   读取 `expert_rules.csv`。
        *   遍历每一行，将 `{'phrase_to_merge': 'standard_form'}` 添加到 `merge_dict`。
        *   **这是第一步，确保专家规则被优先载入。**
    3.  **其次，加载审核后的候选规则：**
        *   读取 `candidate_phrases_reviewed.csv`。
        *   遍历每一行：
            *   如果 `action_code` 是 `1` (Merge)，并且待合并的短语 `candidate_phrase` **尚未存在**于 `merge_dict` 的键中（这保证了专家规则的优先权），则将 `{'candidate_phrase': 'standard_form'}` 添加到 `merge_dict`。
            *   如果 `action_code` 是 `2` (Stopword)，则将 `candidate_phrase` 拆分成单个词，并添加到 `new_stopwords` 集合中。
*   **阶段产出：** 一个最终的、整合了所有智慧的 `merge_dict` 和 `new_stopwords`。

#### **4.2 应用规则固化文本**

*   **逻辑与目的：**
    这是实体固化的执行步骤。使用上一步生成的合并词典，对原始文本进行大规模替换。
*   **输入：** `final_china_news.csv`中的`CONTENT`列，以及`merge_dict`。
*   **动作：**
    1.  **构建高效替换引擎：** 由于`merge_dict`可能很大，简单的循环替换会很慢。这里再次使用`pyahocorasick`库。将`merge_dict`的键（待查找的短语）添加到自动机中，并将值（标准形式）作为其关联数据。
    2.  **批量替换：** 遍历所有文章。使用自动机找到所有匹配的短语，并用其对应的标准形式进行替换。这是一个非重叠的最长匹配替换过程，需要编写稍复杂的替换函数。
*   **阶段产出：** 一个新的DataFrame列，例如`content_solidified`，其中所有实体和短语已被固化。
*   **预估时间：** 使用`pyahocorasick`进行替换，速度很快，预计在 **15 - 30 分钟**内完成。

---
完成这四个阶段后，你就拥有了一个经过实体固化的文本语料库。接下来的深度净化步骤（如词形还原、移除停用词等）就可以在这个高质量的基底上安全、高效地进行了。