FILE=$1

if [ $FILE == 'CelebA' ]
then
    URL=https://www.dropbox.com/s/3e5cmqgplchz85o/CelebA_nocrop.zip?dl=0
    ZIP_FILE=./dataset/CelebA.zip

elif [ $FILE == 'LSUN' ]
then
    URL=https://www.dropbox.com/s/zt7d2hchrw7cp9p/church_outdoor_train_lmdb.zip?dl=0
    ZIP_FILE=./dataset/church_outdoor_train_lmdb.zip
else
    echo "Available datasets are: CelebA and LSUN"
    exit 1
fi

mkdir -p ./dataset/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./dataset/

if [ $FILE == 'CelebA' ]
then
    mv ./dataset/CelebA_nocrop ./dataset/CelebA
fi

rm $ZIP_FILE