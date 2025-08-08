### **需求文档：监督式主题建模与分析 (sLDA)**

**项目目标：** 将深度清洗后的文本数据与时间序列的汇率数据对齐，构建并训练一个sLDA模型，以发现能够直接解释和预测汇率波动的潜在经济叙事主题，并对结果进行深入的经济学叙事解读。

**核心输入文件：**
*   `data_processed/china_news_cleaned_tokens.pkl`: 包含 `tokens_for_lda` 列的DataFrame。
*   `data_raw/exchange_rate_daily.csv`: (**需新建/准备**) 包含至少两列 `DATE` 和 `PRICE` (例如USD/CNY收盘价) 的日度汇率数据文件。

**核心输出文件：**
*   `data_processed/slda_aligned_data.pkl`: 包含对齐后的每日新闻Tokens和对应汇率变动的DataFrame。
*   `data_processed/slda_dictionary.gensim`: 供sLDA模型使用的`gensim`词典。
*   `data_processed/slda_corpus.pkl`: 供sLDA模型使用的`gensim`语料库。
*   `models/slda_model_K{k}.model`: 训练好的、不同主题数K的sLDA模型文件。
*   `reports/slda_topic_analysis_K{k}.csv`: 包含最优模型的主题关键词、回归系数和显著性的分析报告。
*   `reports/slda_daily_topic_intensity.csv`: 包含每日新闻的主题强度时间序列，用于叙事归因。

---

### **阶段一：数据对齐与准备**

**此阶段总目标：** 为sLDA模型准备好两大核心输入：文档（每日合并的Token列表）和与之严格对应的响应变量（每日汇率变动）。

#### **1.1 准备响应变量 (Response Variable)**
*   **动作：**
    1.  **加载汇率数据**：使用`pandas`读取`exchange_rate_daily.csv`，并将`DATE`列解析为日期时间对象。
    2.  **计算对数收益率**：`df_exchange['log_return'] = np.log(df_exchange['PRICE'] / df_exchange['PRICE'].shift(1))`。这将作为我们的响应变量 `y`。
    3.  **处理缺失值**：移除因计算`shift(1)`产生的第一行`NaN`值。

#### **1.2 对齐文本与响应变量**
*   **输入**：`china_news_cleaned_tokens.pkl` 和处理好的汇率DataFrame。
*   **动作：**
    1.  **加载清洗后的新闻数据**。
    2.  **确保日期格式一致**：将新闻数据的`DATE`列也转换为日期时间对象（仅保留日期，不含时间）。
    3.  **按日合并新闻**：`daily_texts = df_news.groupby('DATE')['tokens_for_lda'].sum()`。这将把同一天的所有Token列表合并成一个大的列表。
    4.  **合并数据**：使用`pandas.merge`，以内连接（`inner`）的方式，将`daily_texts`和`df_exchange`按`DATE`列合并。这确保了我们只保留那些既有新闻又有对应汇率变动的数据点。
*   **阶段产出**：一个名为 `df_aligned` 的DataFrame，包含 `DATE`, `tokens`, `log_return` 三列。将其保存为 `slda_aligned_data.pkl`。

---

### **阶段二：Gensim语料库生成**

**此阶段总目标：** 将对齐好的每日Token列表转换为`gensim`库（或sLDA模型库可能需要的）标准格式。

#### **2.1 创建并过滤词典**
*   **输入**：`df_aligned`中的`tokens`列。
*   **动作：**
    1.  `dictionary = gensim.corpora.Dictionary(df_aligned['tokens'])`
    2.  **执行过滤**：`dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=100000)`。
    3.  **保存词典**：`dictionary.save('slda_dictionary.gensim')`。

#### **2.2 创建语料库 (DTM)**
*   **输入**：`df_aligned`中的`tokens`列和过滤后的`dictionary`。
*   **动作：**
    1.  `corpus = [dictionary.doc2bow(text) for text in df_aligned['tokens']]`
    2.  **保存语料库**：使用 `pickle` 将 `corpus` 列表保存到 `slda_corpus.pkl`。

---

### **阶段三：sLDA模型训练与主题发现**

**此阶段总目标：** 使用准备好的数据，训练sLDA模型，并通过超参数搜索找到最优模型。

#### **3.1 模型实现的技术选型**
*   **动作：**
    1.  **调研**：调研目前社区中稳定且有良好文档的sLDA Python库。
        *   **首选方案：`tomotopy`**。它是一个用C++编写的高性能库，支持sLDA（在其中称为`SLDAModel`），速度快，且有Python接口。
        *   备选方案：`pyslda` 或其他GitHub上的研究实现，但可能需要更多调试工作。
    2.  **决策**：我们初步决定采用 `tomotopy` 作为实现工具。

#### **3.2 超参数搜索与模型训练**
*   **逻辑**：我们将编写一个循环，来训练不同主题数 `K` 的sLDA模型，并评估它们的性能。
*   **动作：**
    1.  **定义K值范围**：`k_values = [20, 30, 40, 50, 60, 80, 100, 120, 150]`。
    2.  **准备数据**：将我们的`corpus`和`log_return` (`y`) 转换为`tomotopy`需要的数据格式。
    3.  **循环训练**：
        *   对于每个 `k` in `k_values`:
            *   初始化 `tp.SLDAModel(k=k, vars='r')` (`vars='r'`代表响应变量是实数/回归任务)。
            *   将数据添加到模型中。
            *   设置训练参数（如迭代次数、worker数量）并调用 `model.train()`。
            *   **评估**：记录模型的**对数似然（log-likelihood）**，并评估其对 `y` 的**均方误差（MSE）**。
            *   **保存模型**：将训练好的模型保存到 `models/slda_model_K{k}.model`。
    4.  **选择最优模型**：根据评估结果（例如，寻找MSE的“拐点”或最低点，同时参考对数似然），选择一个最优的主题数 `K_best`。

---

### **阶段四：结果分析与叙事解读**

**此阶段总目标：** 深入解读最优sLDA模型的输出，将量化结果转化为有意义的经济学洞见。

#### **4.1 主题解读**
*   **输入**：最优模型 `slda_model_K{K_best}.model`。
*   **动作：**
    1.  **提取主题-词汇分布**：使用 `model.get_topic_words(topic_id)` 方法，为每个主题提取前15个最相关的关键词及其权重。
    2.  **创建分析报告**：将所有主题的关键词整理到一个CSV文件 `slda_topic_analysis_K{K_best}.csv` 中，并增加一列 `topic_name` 待人工填写。

#### **4.2 分析主题与汇率的关系**
*   **动作：**
    1.  **提取回归系数**：`tomotopy`的`SLDAModel`有一个 `get_regression_coef()` 方法，可以直接返回回归系数向量 `η`。
    2.  将这个向量 `η` 添加到 `slda_topic_analysis_K{K_best}.csv` 报告中，与每个主题一一对应。
    3.  **识别关键主题**：按回归系数的绝对值对主题进行排序，找出对汇率波动影响最大的正面和负面主题。

#### **4.3 叙事归因与案例研究**
*   **动作：**
    1.  **计算每日主题强度**：遍历我们对齐好的数据集中的每一天，使用 `model.infer()` 方法来推断当天新闻的**主题分布**。将所有天的主题分布结果整理成一个时间序列DataFrame。
    2.  **可视化**：绘制关键主题（如影响最大）的强度时间序列图，并与汇率波动图进行叠加对比。
    3.  **寻找关键事件日**：找出汇率波动最剧烈（`log_return`绝对值最大）的几天。
    4.  **归因分析**：在这些关键日期，查看是哪些关键主题的强度（主题分布的概率值）出现了异常峰值。
    5.  **追溯原文**：根据日期，回到我们最原始的新闻数据库 (`final_china_news.csv`)，找到当天的所有新闻。通过阅读这些新闻的标题和内容，验证它们是否真的与sLDA发现的那个飙升的主题（例如“贸易紧张局势”）高度相关，从而完成从**量化信号**到**真实世界叙事**的闭环。

