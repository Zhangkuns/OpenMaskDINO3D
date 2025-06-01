import os
from datetime import datetime

lr = 5e-6
epoch = 3
tag = "sqa3d#multi3dref"
output_dir = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{lr}_ep{epoch}_{tag}"

os.environ['OUTPUT_DIR'] = output_dir