conda create -n lxmert python=3.6 pip --yes
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lxmert

# clone git repo locally, then copy files to remote instance

cd $HOME/playground/hmm/lib/lxmert
pip install -r requirements.txt
pip install sklearn

pip install ipykernel
python -m ipykernel install --prefix=/home/ubuntu/anaconda3 --name lxmert --display-name "Python (lxmert)"

cd $HOME
echo "Done"
