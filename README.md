# ScanObjectDiff3F

# Install


```bash
conda env create -f environment.yaml                          
conda activate diff3f                                         
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
ipython3
```

(Paste into ipython terminal):
<!-- ```python
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html --force-reinstall
``` -->
```python
import sys
import torch

# Ensure you are on torch 2.4.0
print(f"Current PyTorch version: {torch.__version__}")

pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
version_str = "".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])

# Install dependencies normally from PyPI
!pip install fvcore iopath

# Install PyTorch3D from the specific wheel link, ignoring PyPI, BUT skipping dependencies
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html --force-reinstall --no-deps
```

Back to normal shell terminal

```bash
pip uninstall xformers
pip install accelerate==0.24.0
pip install plyfile
pip install trimesh --no-deps --force-reinstall
```

Download IP adapter models, save them in ipmodels/

https://huggingface.co/h94/IP-Adapter/tree/main/models
especially
* models/ip-adapter_sd15.bin
* models/image_encoder/(model.safetensors + config.json)

Dataset:
Download from https://hkust-vgd.ust.hk/scanobjectnn/
