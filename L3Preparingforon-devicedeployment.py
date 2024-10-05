# L3: Preparing for on-device deployment
# ⏳ Note (Kernel Starting): This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.

# Capture trained model
# from qai_hub_models.models.ffnet_40s import Model as FFNet_40s
# ​
# # Load from pre-trained weights
# ffnet_40s = FFNet_40s.from_pretrained()
# import torch
# input_shape = (1, 3, 1024, 2048)
# example_inputs = torch.rand(input_shape)
# traced_model = torch.jit.trace(ffnet_40s, example_inputs)
# traced_model
# Compile for device
# 💻   Access Utils File and Helper Functions: To access the files for this notebook, 1) click on the "File" option on the top menu of the notebook and then 2) click on "Open". For more help, please see the "Appendix - Tips and Help" Lesson.

# import qai_hub
# import qai_hub_models
# ​
# from utils import get_ai_hub_api_token
# ai_hub_api_token = get_ai_hub_api_token()
# ​
# !qai-hub configure --api_token $ai_hub_api_token
# for device in qai_hub.get_devices():
#     print(device.name)
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
# device = qai_hub.Device(selected_device)
# ​
# # Compile for target device
# compile_job = qai_hub.submit_compile_job(
#     model=traced_model,                        # Traced PyTorch model
#     input_specs={"image": input_shape},        # Input specification
#     device=device,                             # Device
# )
# # Download and save the target model for use on-device
# target_model = compile_job.get_target_model()
# Exercise: Try different runtimes
# compile_options="--target_runtime tflite"                  # Uses TensorFlow Lite
# compile_options="--target_runtime onnx"                    # Uses ONNX runtime
# compile_options="--target_runtime qnn_lib_aarch64_android" # Runs with Qualcomm AI Engine
# ​
# compile_job_expt = qai_hub.submit_compile_job(
#     model=traced_model,                        # Traced PyTorch model
#     input_specs={"image": input_shape},        # Input specification
#     device=device,                             # Device
#     options=compile_options,
# )
# Expore more compiler options here.

# On-Device Performance Profiling
# from qai_hub_models.utils.printing import print_profile_metrics_from_job
# ​
# # Choose device
# device = qai_hub.Device(selected_device)
# ​
# # Runs a performance profile on-device
# profile_job = qai_hub.submit_profile_job(
#     model=target_model,                       # Compiled model
#     device=device,                            # Device
# )
# ​
# # Print summary
# profile_data = profile_job.download_profile()
# print_profile_metrics_from_job(profile_job, profile_data)
# Exercise: Try different compute units
# profile_options="--compute_unit cpu"     # Use cpu 
# profile_options="--compute_unit gpu"     # Use gpu (with cpu fallback) 
# profile_options="--compute_unit npu"     # Use npu (with cpu fallback) 
# ​
# # Runs a performance profile on-device
# profile_job_expt = qai_hub.submit_profile_job(
#     model=target_model,                     # Compiled model
#     device=device,                          # Device
#     options=profile_options,
# )
# On-Device Inference
# sample_inputs = ffnet_40s.sample_inputs()
# sample_inputs
# torch_inputs = torch.Tensor(sample_inputs['image'][0])
# torch_outputs = ffnet_40s(torch_inputs)
# torch_outputs
# inference_job = qai_hub.submit_inference_job(
#         model=target_model,          # Compiled model
#         inputs=sample_inputs,        # Sample input
#         device=device,               # Device
# )
# ondevice_outputs = inference_job.download_output_data()
# ondevice_outputs['output_0']
# from qai_hub_models.utils.printing import print_inference_metrics
# print_inference_metrics(inference_job, ondevice_outputs, torch_outputs)
# Get ready for deployment!
# target_model = compile_job.get_target_model()
# _ = target_model.download("FFNet_40s.tflite")