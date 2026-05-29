### setup
source ~/00_WorkDir/.rainbow/bin/activate
export PYTHONPATH=/home/master/00_WorkDir/setup_pkg/Py310/build


### initial
python -m pip install -U pip setuptools wheel
python -m pip install -r /home/master/00_WorkDir/06_RAINbow/requirements.txt


### run 

cd holistic
bash script/local_script_cvpr.sh
