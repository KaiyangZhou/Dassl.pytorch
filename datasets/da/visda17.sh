# ------------------------------------------------------------------------
# ROOT is the root directory where you put your domain datasets.
# 
# Suppose you wanna put the dataset under $DATA, which stores all the
# domain datasets, run the following command in your terminal to
# download VisDa17:
#
# $ sh visda17.sh $DATA
#------------------------------------------------------------------------

ROOT=$1
mkdir $ROOT/visda17
cd $ROOT/visda17

wget http://csr.bu.edu/ftp/visda17/clf/train.tar
tar xvf train.tar

wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
tar xvf validation.tar  

wget http://csr.bu.edu/ftp/visda17/clf/test.tar
tar xvf test.tar

wget https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt -O test/image_list.txt