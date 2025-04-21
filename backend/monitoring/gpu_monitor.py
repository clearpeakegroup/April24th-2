import pynvml
from loguru import logger
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
logger.info(f"GPU memory: {info.used / 1024**2:.2f} MB used / {info.total / 1024**2:.2f} MB total")
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
logger.info(f"GPU Utilization: {util.gpu}% | Memory Utilization: {util.memory}%") 