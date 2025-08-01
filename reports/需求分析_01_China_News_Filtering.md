### **需求文档：中国相关新闻高效筛选与过滤流程 (最终版)**

**项目目标：** 从一个包含海量全球新闻的巨大CSV文件 (`final_merged_all_news.csv`) 中，通过一个高度优化的、分为“快速初筛”和“上下文精筛”两阶段的自动化流程，精准地筛选出与中国经济、政治、科技等领域高度相关的英文新闻，并生成一个干净、高相关性的最终数据集以供后续分析。

**输入与输出文件：**
*   **主要输入文件：**
    *   `../data_raw/final_merged_all_news.csv`: 包含了所有新闻的原始、未经过滤的庞大语料库。
    *   `../data_raw/china_keywords_collection.json`: 一个结构化的JSON文件，定义了所有与中国相关的关键词、它们的别名、类别以及相关性等级 (tier 1-5)。
*   **过程与输出文件：**
    *   `../data_processed/china_news_candidates.csv`: 经过第一阶段快速初筛后产生的“候选文章”集合。
    *   `../data_processed/final_china_news.csv`: 经过第二阶段精筛后得到的最终高相关性新闻集合（**核心产出**）。
    *   `../data_processed/china_news_rejected_articles.csv`: 在精筛阶段因未达到相关性分数阈值而被拒绝的文章，附带其最终得分和评估详情，用于分析和优化规则。

---

### **阶段一：准备工作与环境设置**

**此阶段总目标：** 导入所有必需的Python库，定义文件路径，并根据运行环境的硬件资源（CPU核心数）智能配置并行处理参数，为后续高效处理奠定基础。

*   **逻辑与目的：**
    这是整个流程的初始化步骤。通过集中管理所有配置（包括精筛阶段的积分权重），可以方便地调整路径和参数。
*   **动作：**
    1.  **导入核心库与优化库：** 加载 `pandas`, `spacy`, `os`, `re`, `json`, `flashtext`, `tqdm`, `psutil` 等。
    2.  **定义全局路径与参数：** 设置所有输入输出文件的路径，以及 `CHUNKSIZE`, `BATCH_SIZE` 等。
    3.  **配置并行参数：** 自动检测CPU核心数，设定 `N_PROCESSES` 以供第二阶段精筛使用。
    4.  **配置5级积分系统权重：** 集中定义所有加分项（如 `LEAD_BONUS_TIER_5`, `TIER_4_SCORE`）和扣分项（`NEGATION_PENALTY`）的具体分值，以及最终的接受阈值 `ACCEPTANCE_THRESHOLD`。

*   **阶段产出：**
    *   所有必需的库被载入内存。
    *   所有全局参数和积分权重被设定。

---

### **阶段二：构建高效关键词初筛引擎**

**此阶段总目标：** 读取关键词JSON文件，并构建一个基于`Flashtext`库的高速关键词查找处理器。

*   **逻辑与目的：**
    初筛阶段旨在从海量数据中快速、无遗漏地捕获任何可能相关的文章，`Flashtext`的极高效率使其成为此阶段的最佳工具。
*   **输入：** `KEYWORD_JSON_PATH` 指向的 `china_keywords_collection.json` 文件。
*   **动作：**
    1.  **加载关键词文件：** 读取并解析 `china_keywords_collection.json`。
    2.  **构建处理器：** 初始化一个不区分大小写的 `Flashtext` `KeywordProcessor`，并将JSON中所有关键词及其别名添加进去。
*   **阶段产出：**
    *   一个名为 `keyword_processor` 的 `Flashtext` 对象，用于下一阶段的快速匹配。
    *   一个名为 `keywords_data` 的列表，供第四阶段构建精筛工具使用。

---

### **阶段三：执行第一阶段 - 大规模流式初筛**

**此阶段总目标：** 以**单进程流式处理**（streaming）的方式，对巨大的原始新闻文件进行扫描，使用 `Flashtext` 处理器进行快速匹配，并将包含任何关键词的文章筛选出来，存为候选集。

*   **逻辑与目的：**
    采用流式处理以避免内存耗尽，采用单进程以保证稳定性和逻辑简洁性。此阶段的目标是“求全”，宁可错选，不可放过。
*   **动作：**
    1.  **创建流式迭代器：** 使用 `pandas.read_csv` 的 `chunksize` 参数逐块读取大文件。
    2.  **顺序处理：** 遍历每个数据块，对其内容进行轻量级文本清洗，然后使用 `keyword_processor` 筛选出至少包含一个关键词的文章。
    3.  **写入结果：** 将筛选出的文章以追加模式实时写入 `CANDIDATES_FILE`。
*   **阶段产出：**
    *   一个名为 `china_news_candidates.csv` 的CSV文件，包含所有潜在相关的文章。

---

### **阶段四：准备第二阶段 - 构建5级动态积分评估引擎**

**此阶段总目标：** 加载 `spaCy` 模型，并根据配置的5级权重，构建一套**强力奖励高相关性信号**的动态积分评估规则。

*   **逻辑与目的：**
    此阶段的核心逻辑是：一篇真正相关的文章，不仅会包含多个关键词，而且最重要的关键词往往会出现在文章的开头部分。因此，本系统通过**高额的开篇奖励**和**权重显著的频率分**来确保高价值文章能够获得远超阈值的分数，使其足以抵御少量负面信号（如假设句）的扣分。
*   **动作：**
    1.  **加载优化后的 spaCy 模型**。
    2.  **构建关键词信息查找表 (`keyword_lookup`)**，将每个关键词映射到其 `relevance_tier`。
    3.  **构建 `PhraseMatcher`**。
    4.  **定义积分/扣分规则函数：**
        *   **正向加分规则 (核心):**
            *   **`score_lead_paragraphs_presence` (动态前导奖励):**
                *   检查文章的**前10个句子**。
                *   找出其中出现的**最高等级**的关键词。
                *   根据该关键词的等级，给予一个**动态的、高额的奖励**（例如，T5词+20分，T4词+15分，T3词+10分，T2词+5分）。
            *   **`score_keyword_frequency` (全文档加权频率分):**
                *   扫描**整篇文章**中的所有关键词。
                *   根据每个词的等级，给予**显著的加权分数**（例如，T5词+5分，T4词+4分...T1词+1分）。
        *   **负向扣分规则 (风险控制):**
            *   `penalize_hypothetical`: 每发现一个包含关键词的**假设句**，进行适度扣分（-2分）。
            *   `penalize_negation`: 每发现一个被直接**否定的关键词**，进行适度扣分（-3分）。
*   **阶段产出：**
    *   `nlp` 对象、`matcher` 对象和 `keyword_lookup` 字典。
    *   一套强大的、可调用的Python评估函数。

---

### **阶段五：执行第二阶段 - 上下文精筛与产出**

**此阶段总目标：** 读取候选集，对每篇文章应用5级动态积分评估体系，计算其最终相关性分数，并根据分数是否达到预设门槛来决定接受或拒绝。

*   **逻辑与目的：**
    这是质量控制的最后一步，但其主要目标是确认并放行绝大多数在初筛阶段被选中的文章，仅剔除那些得分极低、几乎没有积极信号或有多个风险信号的明显无关项。
*   **动作：**
    1.  **加载候选集**。
    2.  **并行NLP处理：** 使用 `nlp.pipe` 对所有候选文章的 `CONTENT` 列进行高效并行的 `spaCy` 处理。
    3.  **应用积分规则进行精筛：**
        *   遍历每一个 `Doc` 对象，为每篇文章初始化 `relevance_score = 0`。
        *   **累加正向分数：** 调用 `score_lead_paragraphs_presence` 和 `score_keyword_frequency` 函数，将得分累加。
        *   **减去负向分数：** 调用 `penalize_hypothetical` 和 `penalize_negation` 函数，从总分中扣除。
    4.  **决策与分离：**
        *   将每篇文章的最终 `relevance_score` 与 `ACCEPTANCE_THRESHOLD` (例如 5.0 分) 进行比较。
        *   如果分数达标，则接受；否则拒绝。
    5.  **保存结果：**
        *   将“接受”的DataFrame保存到 `FINAL_RESULT_FILE`。
        *   将被拒绝的文章及其详细的得分和评估明细保存到 `REJECTED_FILE`。
*   **阶段产出：**
    *   **`final_china_news.csv`**: 最终的高质量、高相关性新闻数据集。
    *   **`china_news_rejected_articles.csv`**: 少数被过滤掉的文章，附带其最终得分和评估明细，用于极端案例分析。