python train.py --dataset=phone-email --epochs=150 --record --exp_name=edge_noise --runs=10
for %%N in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) do (
  python train.py --dataset=phone-email --epochs=150 --record --exp_name=edge_noise --runs=10 --edge_noise=%%N
)

python train.py --dataset=foursquare-twitter --epochs=150 --record --exp_name=edge_noise --runs=10
for %%N in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) do (
  python train.py --dataset=foursquare-twitter --epochs=150 --record --exp_name=edge_noise --runs=10 --edge_noise=%%N
)
