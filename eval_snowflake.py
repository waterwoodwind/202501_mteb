import mteb
from sentence_transformers import SentenceTransformer
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 显示当前设备信息
logger.info(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
logger.info(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")

# 清理显存
torch.cuda.empty_cache()

# 1. 加载模型
model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"  
model = SentenceTransformer(model_name)

# 2. 选择中文相关任务
tasks = [
    # 文本分类任务
    #"YueOpenriceReviewClassification",  # 粤语餐厅评论分类
    #"T2Reranking",  # 中文文本重排序
    "DuRetrieval",  # 中文语义相似度
]

try:
    # 3. 创建评测实例
    logger.info("开始创建评测实例...")
    evaluation = mteb.MTEB(
        tasks=tasks,
        task_langs=["cmn"]
    )

    # 4. 运行评测
    logger.info("开始运行评测...")
    results = evaluation.run(
        model,
        output_folder="results",
        encode_kwargs={
            "batch_size": 16,
            "show_progress_bar": True,
            "max_length": 256,
            "normalize_embeddings": True,
            "chunk_size": 10000
        }
    )

    # 5. 输出结果
    logger.info("评测完成，结果如下:")
    print(results)

except Exception as e:
    logger.error(f"评测过程中发生错误: {e}")
    raise

