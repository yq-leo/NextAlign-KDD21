python train.py --dataset=phone-email --epochs=5 --record --exp_name=edge_noise --runs=10
for edge_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=phone-email --epochs=5 --record --exp_name=edge_noise --runs=10 --edge_noise=$edge_noise
done

python train.py --dataset=foursquare-twitter --epochs=5 --record --exp_name=edge_noise --runs=10
for edge_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=foursquare-twitter --epochs=5 --record --exp_name=edge_noise --runs=10 --edge_noise=$edge_noise
done
