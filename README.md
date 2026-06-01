# Requirements
1. Install Matterport3D simulators: follow instructions here. We use the latest version instead of v0.1.
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

2. Install requirements
conda create --name dialnav python=3.10
conda activate dialnav
pip install -r requirements.txt

# for lana
apt-get update && apt-get install -y openjdk-17-jre-headless

# for GCN Localization
import nltk
nltk.download('punkt_tab')

3. download dataset from here #TODO and put it under <directory>/dataset
it should look like
directory
- dataset
- holistic
- modules

4. run holisitic task with provided weights
change YOUR_CODE_DIRECTORY in run.sh to your directory
cd holistic
bash script/run.sh

We tested under RTX 3090x1


5. Training each modules
현재 기준 SOTA 모델들의 학습 script 를 공유한다.
## Navigator Agent
### Navigation Module
cd /modules/nav/DST/map_nav_src
bash ../script/local_train.sh 

### Question Module
cd /modules/qa/LANA/finetune_src
bash scripts/local_q_train.sh 

## Guide Agent
### Localization Module
cd /modules/loc/GTL/gtl
bash script/local_train.sh 

### Answer Module
cd /modules/qa/LANA/finetune_src
bash scripts/local_a_train.sh 






This soure is from
LANA
ScaleVLN
DUET
GCN.. 