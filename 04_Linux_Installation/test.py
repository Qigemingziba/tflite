import tensorflow as tf
import numpy as np

# 定义输入数据
input_shape = (1, 3, 512, 512)  # 输入数据的形状
input_data = np.random.rand(*input_shape).astype(np.float32)  # 生成随机输入数据

# 载入 TFLite 模型
tflite_model_path = 'vae_encoder.tflite'  # 替换为你的 TFLite 模型路径

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 获取输入和输出张量的索引
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 将输入数据设置到输入张量
interpreter.set_tensor(input_details[0]['index'], input_data)

# 执行推理
interpreter.invoke()

# 获取输出数据
output_data = interpreter.get_tensor(output_details[0]['index'])

# 打印输出数据的形状（假设输出是一个张量）
print("Output shape:", output_data.shape)
