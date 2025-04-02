# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create a conda environment
conda create -n semnav python=3.9 cmake=3.14.0 -y
conda activate semnav

# Setup habitat-sim
git clone --depth 1 --branch v0.2.2 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim || exit
pip install -r requirements.txt
python setup.py install --headless
cd ..

# Install latest PyTorch (this can cause conflicts in the future do to inconsistent versions)
pip install torch torchvision torchaudio

# Install additional Python packages
pip install gym==0.22.0 urllib3==1.25.11 numpy==1.25.0 pillow==9.2.0

# Install habitat-lab
git clone https://github.com/carlosgual/habitat-lab.git
cd habitat-lab || exit
python setup.py develop --all
cd ..

# Install additional libraries and semnav
pip install wandb
conda install protobuf -y
pip install -e .

# Set environment variables to silence habitat-sim logs
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"
