# Requirements
1. Install Matterport3D simulators: follow instructions here. We use the latest version instead of v0.1.
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

2. Install requirements
conda create --name dialnav python=3.10
conda activate dialnav
pip install -r requirements.txt

3. download dataset from here #TODO and put it under <directory>/dataset
it should look like
directory
- dataset
- holistic
- modules

4. run
change YOUR_CODE_DIRECTORY in run.sh to your directory
cd holistic
bash script/run.sh

We tested under RTX 3090x1