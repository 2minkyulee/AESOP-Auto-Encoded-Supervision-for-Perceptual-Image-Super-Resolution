pip uninstall pyiqa -y
pip install pyiqa

pip uninstall timm -y
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple

apt-get update
apt-get install git -y
pip install git+https://github.com/openai/CLIP.git

# for basicsr setup
pip uninstall basicsr -y
python setup.py develop
pip install thop
pip install ptflops
