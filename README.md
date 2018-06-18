# p3d_tf
p3d network base on tensorflow

# usage

1.split video

python split_video.py --video videopath --output outputpath

2.split dataset

python createdatalist.py --imagedir imagedir --labelpath labelpath --trainpath trainpath --testpath testpath [--frac frac]

3.create tfrecords

python create_tfrecords.py --picklepath picklepath --savepath savepath

4.train

python train.py

