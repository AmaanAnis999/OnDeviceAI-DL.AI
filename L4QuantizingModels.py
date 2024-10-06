# L4: Quantizing Models
# ⏳ Note (Kernel Starting): This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.

# from datasets import load_dataset
# ​
# # Use input resolution of the network
# input_shape = (1, 3, 1024, 2048)
# ​
# # Load 100 RGB images of urban scenes 
# dataset = load_dataset("UrbanSyn/UrbanSyn", 
#                 split="train", 
#                 data_files="rgb/*_00*.png")
# dataset = dataset.train_test_split(1)
# ​
# # Hold out for testing
# calibration_dataset = dataset["train"]
# test_dataset = dataset["test"]
# calibration_dataset["image"][0]
# Setup calibration/inference pipleline
# import torch
# from torchvision import transforms
# ​
# # Convert the PIL image above to Torch Tensor
# preprocess = transforms.ToTensor()
# ​
# # Get a sample image in the test dataset
# test_sample_pil = test_dataset[0]["image"]
# test_sample = preprocess(test_sample_pil).unsqueeze(0) 
# print(test_sample)
# ​
# import torch.nn.functional as F
# import numpy as np
# from PIL import Image
# ​
# def postprocess(output_tensor, input_image_pil):
# ​
#     # Upsample the output to the original size
#     output_tensor_upsampled = F.interpolate(
#         output_tensor, input_shape[2:], mode="bilinear",
#     )
# ​
#     # Get top predicted class and convert to numpy
#     output_predictions = output_tensor_upsampled[0].argmax(0).byte().detach().numpy().astype(np.uint8)
# ​
#     # Overlay over original image
#     color_mask = Image.fromarray(output_predictions).convert("P")
# ​
#     # Create an appropriate palette for the Cityscapes classes
#     palette = [
#         128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156,
#         190, 153, 153, 153, 153, 153, 250, 170, 30, 220, 220, 0,
#         107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
#         255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100,
#         0, 0, 230, 119, 11, 32]
#     palette = palette + (256 * 3 - len(palette)) * [0]
#     color_mask.putpalette(palette)
#     out = Image.blend(input_image_pil, color_mask.convert("RGB"), 0.5)
#     return out
# Setup model in floating point
# from qai_hub_models.models.ffnet_40s.model import FFNet40S
# model = FFNet40S.from_pretrained().model.eval()
# ​
# # Run sample output through the model
# test_output_fp32 = model(test_sample)
# test_output_fp32
# ​
# postprocess(test_output_fp32, test_sample_pil)
# Prepare Quantized Model
# from qai_hub_models.models._shared.ffnet_quantized.model import FFNET_AIMET_CONFIG
# from aimet_torch.batch_norm_fold import fold_all_batch_norms
# from aimet_torch.model_preparer import prepare_model
# from aimet_torch.quantsim import QuantizationSimModel
# ​
# # Prepare model for 8-bit quantization
# fold_all_batch_norms(model, [input_shape])
# model = prepare_model(model)
# ​
# # Setup quantization simulator
# quant_sim = QuantizationSimModel(
#     model,
#     quant_scheme="tf_enhanced",
#     default_param_bw=8,              # Use bitwidth 8-bit
#     default_output_bw=8,
#     config_file=FFNET_AIMET_CONFIG,
#     dummy_input=torch.rand(input_shape),
# )
# Perform post training quantization
# size = 5  # Must be < 100
# ​
# def pass_calibration_data(sim_model: torch.nn.Module, args):
#     (dataset,) = args
#     with torch.no_grad():
#         for sample in dataset.select(range(size)):
#             pil_image = sample["image"]
#             input_batch = preprocess(pil_image).unsqueeze(0)
# ​
#             # Feed sample through for calibration
#             sim_model(input_batch)
# ​
# # Run Post-Training Quantization (PTQ)
# quant_sim.compute_encodings(pass_calibration_data, [calibration_dataset])
# test_output_int8 = quant_sim.model(test_sample)
# postprocess(test_output_int8, test_sample_pil)
# Run Quantized model on-device
# 💻   Access Utils File and Helper Functions: To access the files for this notebook, 1) click on the "File" option on the top menu of the notebook and then 2) click on "Open". For more help, please see the "Appendix - Tips and Help" Lesson.

# import qai_hub
# import qai_hub_models
# ​
# from utils import get_ai_hub_api_token
# ai_hub_api_token = get_ai_hub_api_token()
# ​
# !qai-hub configure --api_token $ai_hub_api_token
# ⏳ Note: To spread the load across various devices, we are selecting a random device. Feel free to change it to any other device you prefer.

# devices = [
#     "Samsung Galaxy S22 Ultra 5G",
#     "Samsung Galaxy S22 5G",
#     "Samsung Galaxy S22+ 5G",
#     "Samsung Galaxy Tab S8",
#     "Xiaomi 12",
#     "Xiaomi 12 Pro",
#     "Samsung Galaxy S22 5G",
#     "Samsung Galaxy S23",
#     "Samsung Galaxy S23+",
#     "Samsung Galaxy S23 Ultra",
#     "Samsung Galaxy S24",
#     "Samsung Galaxy S24 Ultra",
#     "Samsung Galaxy S24+",
# ]
# ​
# import random
# selected_device = random.choice(devices)
# print(selected_device)
# %run -m qai_hub_models.models.ffnet_40s_quantized.export -- --device "$selected_device"
# ​
# ​