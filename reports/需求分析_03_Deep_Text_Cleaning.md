### **需求文档：深度文本清洗流程**

**项目目标：** 将经过实体固化的文本（`content_solidified`）通过一系列标准化的NLP清洗步骤，转换为最终可用于主题建模（如LDA）的高质量、纯净化的Token列表。

**核心输入文件：**
*   `data_processed/china_news_solidified.pkl`: 包含 `content_solidified` 列的DataFrame。
*   `data_processed/new_stopwords.pkl`: 在阶段`02_C`中，从人工审核生成的自定义停用词集合。
*   `configs/project_specific_stopwords.txt`: （**需新建**）一个手动维护的、项目专属的停用词文本文件，每行一个停用词。

**核心输出文件：**
*   `data_processed/china_news_cleaned_tokens.pkl`: 最终产物，一个包含新增 `tokens_for_lda` 列的DataFrame。该列的每一行都是一个经过深度清洗的Token列表。

---

### **阶段一：环境设置与数据加载**

**此阶段总目标：** 导入所有必需的库，加载核心数据和规则文件，并**配置`spaCy`以正确处理我们的“超级词”**。

*   **动作：**
    1.  **导入库：** 加载 `pandas`, `spacy`, `os`, `pickle`, `tqdm` 等。
    2.  **配置与路径：** 沿用我们的标准配置块，设置所有输入输出路径。并新建`configs`目录（如果不存在）。
    3.  **加载数据：** 读取 `china_news_solidified.pkl` 到DataFrame。
    4.  **加载NLP模型与特殊规则配置：**
        *   加载 `spacy.load("en_core_web_lg")`。
        *   加载 `merge_dict.pkl`。
        *   **关键操作：** 遍历`merge_dict.pkl`中的所有“超级词”（如 `peoples_bank_of_china`），为每一个词创建一个`spacy.tokens.Token.set_extension`属性，例如`is_super_word`，并默认为`False`。然后，将这些词作为**特殊规则**添加到`spaCy`的分词器（Tokenizer）中，确保`spaCy`在处理文本时，**绝不会**将它们错误地拆分开，并能识别它们。

---

### **阶段二：并行化的NLP处理与特征提取**

**此阶段总目标：** 利用`spaCy`的`nlp.pipe`功能，对所有固化后的文本进行一次性的、高效并行的处理，提取出后续清洗步骤所需的所有语言学特征（Token、Lemma、POS）。

*   **逻辑与目的：**
    我们不在循环中反复调用NLP功能，而是一次性处理所有文本，将结果（`Doc`对象）存储起来。这遵循了“计算一次，多次使用”的高效原则。
*   **输入：** `content_solidified` 列。
*   **动作：**
    1.  **禁用非必需组件：** 在`nlp.pipe`中，禁用`ner`和`parser`，因为我们在这个阶段主要需要分词、词性标注和词形还原。
    2.  **执行并行处理：** `docs = list(nlp.pipe(df['content_solidified'], n_process=N_PROCESSES))`
        *   **【稳健性说明】**：此并行处理在Linux/macOS环境下通常稳定高效。如果在Windows或特定的Jupyter环境中遇到卡死问题，应立即切换到单线程模式（通过移除`n_process`参数或将其设为-1）来保证流程的稳定执行。
*   **阶段产出：** 一个包含所有文章的`spaCy` `Doc`对象的Python列表 `docs`。

---

### **阶段三：集成的、多步骤清洗流程**

**此阶段总目标：** 遍历上一步生成的`docs`列表，对每个`Doc`对象应用一系列过滤和转换规则，生成最终的干净Token列表。

*   **逻辑与目的：**
    这是一个集成的处理管道，确保所有清洗步骤都以正确的顺序执行，以达到最佳的清洗效果和效率。
*   **动作：**
    1.  **构建最终停用词集：**
        *   获取 `spacy.Defaults.stop_words`。
        *   加载 `new_stopwords.pkl`。
        *   尝试读取 `configs/project_specific_stopwords.txt`。如果文件不存在，则打印警告并继续。
        *   将所有来源的停用词合并成一个大的`set`，名为 `FINAL_STOP_WORDS`。
    2.  **定义词性白名单：** `ALLOWED_POS = {'NOUN', 'PROPN', 'ADJ', 'VERB'}`。
    3.  **循环处理每个`Doc`对象：**
        *   初始化一个空的 `cleaned_tokens` 列表。
        *   遍历`Doc`中的每一个`token`：
            a. **初步过滤：** 如果`token`是标点符号、空格或纯数字，则跳过。
            b. **【顺序优化】词性过滤：** 如果该`token`不是我们定义的“超级词”，并且其`token.pos_`不在`ALLOWED_POS`白名单中，则跳过。（超级词即使词性识别不准也应保留）。
            c. **转为小写与词形还原：** 获取`token.lemma_.lower()`。对于“超级词”，其`lemma_`就是它本身，这一步不会对其造成影响。
            d. **停用词过滤：** 如果还原后的`lemma`存在于`FINAL_STOP_WORDS`中，则跳过。
            e. **短词过滤：** 如果`lemma`的长度小于3，则跳过。
            f. **通过所有关卡：** 如果一个`token`通过了以上所有过滤，则将其`lemma`添加到`cleaned_tokens`列表中。
    4.  **将结果添加回DataFrame：** 将所有文章的`cleaned_tokens`列表集合，作为一个新列 `tokens_for_lda` 添加回原始的DataFrame中。

---

### **阶段四：最终检查与保存**

**此阶段总目标：** 抽样检查清洗效果，并保存最终产物。

*   **动作：**
    1.  **抽样对比：** 随机抽取几行，同时打印原始的`CONTENT`、固化后的`content_solidified`和最终的`tokens_for_lda`，进行人工对比，确保清洗效果符合预期。
    2.  **保存结果：** 将包含`tokens_for_lda`列的最终DataFrame保存为 `china_news_cleaned_tokens.pkl` 和 `...for_review.csv`。

