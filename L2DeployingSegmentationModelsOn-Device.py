# L2: Deploying Segmentation Models On-Device
# ⏳ Note (Kernel Starting): This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.

# FFNet Paper

from qai_hub_models.models.ffnet_40s import Model
from torchinfo import summary
# Load from pre-trained weights
model = Model.from_pretrained()
input_shape = (1, 3, 1024, 2048)
stats = summary(model, 
  input_size=input_shape, 
  col_names=["num_params", "mult_adds"]
)
print(stats)
# Exercise: Try another variant of FFNet
# High resolution variants
from qai_hub_models.models.ffnet_40s import Model
#from qai_hub_models.models.ffnet_54s import Model
#from qai_hub_models.models.ffnet_78s import Model​
# Low resolution variants
low_res_input_shape = (1, 3, 512, 1024)
#from qai_hub_models.models.ffnet_78s_lowres import Model
#from qai_hub_models.models.ffnet_122ns_lowres import Model

model = Model.from_pretrained()
stats = summary(model, 
  input_size=input_shape, # use low_res_input_shape for low_res models
  col_names=["num_params", "mult_adds"]
)
print(stats)
# Setup AI Hub for device-in-the-loop deployment
import qai_hub
# 💻   Access Utils File and Helper Functions: To access the files for this notebook, 1) click on the "File" option on the top menu of the notebook and then 2) click on "Open". For more help, please see the "Appendix - Tips and Help" Lesson.

from utils import get_ai_hub_api_token
ai_hub_api_token = get_ai_hub_api_token()
# ​
# !qai-hub configure --api_token $ai_hub_api_token
# %run -m qai_hub_models.models.ffnet_40s.demo
# Run on a real smart phone!
# ⏳ Note: To spread the load across various devices, we are selecting a random device. Feel free to change it to any other device you prefer.

devices = [
    "Samsung Galaxy S22 Ultra 5G",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S22+ 5G",
    "Samsung Galaxy Tab S8",
    "Xiaomi 12",
    "Xiaomi 12 Pro",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S23",
    "Samsung Galaxy S23+",
    "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S24",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24+",
]

import random
selected_device = random.choice(devices)
print(selected_device)
# %run -m qai_hub_models.models.ffnet_40s.export -- --device "$selected_device"
# Note: To view the URL for each job, you require login. You can experience sample results in the following urls.

# FFNet 40s
# FFNet 54s
# FFNet 78s
# FFNet 78s-low-res
# FFNet 122ns-low-res
# On Device Demo
# %run -m qai_hub_models.models.ffnet_40s.demo -- --device "$selected_device" --on-device
