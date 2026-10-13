import os
import sys

# 1. 动态获取项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR

# 2. 指定相对本地保存路径 (支持通过环境变量自定义或默认存放在 models 目录下)
LOCAL_DIR = os.environ.get(
    "LOCAL_MODEL_DIR", 
    os.path.join(PROJECT_ROOT, "models", "DeepSeek-R1-Distill-Qwen-7B")
)

os.makedirs(LOCAL_DIR, exist_ok=True)

# 3. ModelScope 官方模型 ID
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print("==================================================")
print(f"🚀 开始从 ModelScope 下载模型: {MODEL_ID}")
print(f"📁 本地保存路径: {LOCAL_DIR}")
print("==================================================")

try:
    from modelscope import snapshot_download

    # 执行下载（如果本地已存在部分权重文件，ModelScope 会自动断点续传）
    model_dir = snapshot_download(
        MODEL_ID,
        local_dir=LOCAL_DIR,
        revision="master"
    )

    print("\n==================================================")
    print(f"🎉 下载完成！模型文件已存至: {model_dir}")
    print("==================================================")

except ImportError:
    print("\n❌ 错误: 未安装 modelscope 依赖包！")
    print("请先在终端运行: pip install modelscope")
except Exception as e:
    print(f"\n❌ 模型下载过程中发生异常: {e}")