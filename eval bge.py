import mteb
from sentence_transformers import SentenceTransformer

# 1. 加载模型
model_name = "BAAI/bge-m3"  
model = SentenceTransformer(model_name)

# 2. 选择中文相关任务
tasks = [
    # 文本分类任务
    #"YueOpenriceReviewClassification",  # 粤语餐厅评论分类
    #"T2Reranking",  # 中文文本重排序
    "STSB",  # 中文语义相似度
]

# 3. 创建评测实例
evaluation = mteb.MTEB(
    tasks=tasks,
    task_langs=["cmn"]  # cmn为中文普通话,yue为粤语
)

# 4. 运行评测
results = evaluation.run(
    model,
    output_folder="results",  # 结果保存路径
    encode_kwargs={
        "batch_size": 32,
        "show_progress_bar": True
    }
)

# 5. 查看结果
print("\n评测结果:")

