# PlanB_01_parallel_filter.py

import sys
import time
import json
import re
import html
import unicodedata
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from flashtext import KeywordProcessor
from tqdm import tqdm

# --- 全局变量定义 ---
# 这些将在子进程中被初始化
global_keyword_processor = None
NEWS_COLUMN = 'CONTENT'
CHUNKSIZE = 10000


# --- 辅助函数 ---

def lightweight_clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub('<[^>]*>', '', text)
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def init_worker(processor):
    """每个子进程启动时调用，初始化关键词处理器"""
    global global_keyword_processor
    global_keyword_processor = processor


def process_chunk_safe(chunk_df):
    """处理单个数据块的安全版本"""
    if NEWS_COLUMN not in chunk_df.columns:
        return pd.DataFrame()
    cleaned_series = chunk_df[NEWS_COLUMN].apply(lightweight_clean)
    mask = cleaned_series.apply(lambda x: len(global_keyword_processor.extract_keywords(x)) > 0)
    return chunk_df[mask]


def build_keyword_processor(json_path):
    """从JSON文件构建Flashtext处理器"""
    with open(json_path, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)

    all_aliases = set()
    for item in keywords_data:
        all_aliases.add(item['keyword'])
        for alias in item.get('aliases', []):
            all_aliases.add(alias)

    keyword_processor = KeywordProcessor(case_sensitive=False)
    for kw in all_aliases:
        keyword_processor.add_keyword(kw)

    print(f"✅ 关键词处理器构建完成，包含 {len(all_aliases)} 个关键词。")
    return keyword_processor


# --- 主执行函数 ---
def main(source_file, keyword_file, output_file):
    """执行主要的并行筛选流程"""
    print("--- 开始执行并行初筛脚本 ---")
    start_time = time.time()

    # 构建关键词处理器
    keyword_processor = build_keyword_processor(keyword_file)

    # 创建CSV文件迭代器
    try:
        chunk_iterator = pd.read_csv(source_file, chunksize=CHUNKSIZE, on_bad_lines='skip', low_memory=False)
    except FileNotFoundError:
        print(f"❌ 错误: 源文件未找到 {source_file}")
        return

    is_first_chunk = True
    total_candidates = 0
    processed_chunks = 0

    num_processes = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1

    with ProcessPoolExecutor(max_workers=num_processes, initializer=init_worker,
                             initargs=(keyword_processor,)) as executor:
        results_iterator = executor.map(process_chunk_safe, chunk_iterator)

        # 预计算总块数以显示进度
        print("正在计算文件总块数...")
        try:
            total_chunks = sum(1 for row in open(source_file, 'r', encoding='utf-8', errors='ignore')) // CHUNKSIZE + 1
            print(f"文件约包含 {total_chunks} 个数据块。开始处理...")
        except Exception as e:
            print(f"无法计算总块数，进度条可能不准确。错误: {e}")
            total_chunks = None  # 如果计算失败，则不显示总数

        for candidates in tqdm(results_iterator, total=total_chunks, desc="并行初筛"):
            processed_chunks += 1
            if not candidates.empty:
                total_candidates += len(candidates)
                if is_first_chunk:
                    candidates.to_csv(output_file, index=False, mode='w', encoding='utf-8')
                    is_first_chunk = False
                else:
                    candidates.to_csv(output_file, index=False, mode='a', header=False, encoding='utf-8')

    end_time = time.time()
    print("\n--- 初筛流程执行完毕 ---")
    print(f"总共处理了 {processed_chunks} 个数据块。")
    print(f"总共找到 {total_candidates} 篇候选文章。")
    print(f"结果已保存到: {output_file}")
    print(f"耗时: {(end_time - start_time) / 60:.2f} 分钟。")


# --- 脚本入口 ---
if __name__ == "__main__":
    # 从命令行接收参数
    # sys.argv[0] 是脚本名
    # sys.argv[1] 是 source_file
    # sys.argv[2] 是 keyword_file
    # sys.argv[3] 是 output_file
    if len(sys.argv) != 4:
        print("用法: python PlanB_01_parallel_filter.py <source_csv_path> <keywords_json_path> <output_csv_path>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])