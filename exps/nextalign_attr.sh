#python -u train.py --dataset=cora --use_attr --epochs=50 --record --exp_name=attr_noise --runs=10
#for attr_noise in $(seq 0.1 0.1 0.9); do
#  python -u train.py --dataset=cora --use_attr --epochs=50 --record --exp_name=attr_noise --runs=10 --attr_noise=$attr_noise
#done

python -u train.py --dataset=Douban --use_attr --epochs=10 --record --exp_name=attr_noise --runs=10 --robust --strong_noise
for attr_noise in $(seq 0.1 0.1 0.9); do
  python -u train.py --dataset=Douban --use_attr --epochs=10 --record --exp_name=attr_noise --runs=10 --attr_noise=$attr_noise --robust --strong_noise
done
