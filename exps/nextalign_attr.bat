@REM python train.py --dataset=cora --use_attr --epochs=50 --record --exp_name=attr_noise --runs=10
@REM for %%N in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) do (
@REM   python train.py --dataset=cora --use_attr --epochs=50 --record --exp_name=attr_noise --runs=10 --attr_noise=%%N
@REM )

python -u train.py --dataset=Douban --use_attr --epochs=50 --record --exp_name=attr_noise --runs=10 --robust --strong_noise
for %%N in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) do (
  python -u train.py --dataset=Douban --use_attr --epochs=50 --record --exp_name=attr_noise --runs=10 --attr_noise=%%N --robust --strong_noise
)