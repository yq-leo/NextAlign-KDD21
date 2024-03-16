echo Running on Phone-Email dataset
python train.py --epochs=50 --dataset=phone-email
echo -----------------------------

echo Running on Foursquare-Twitter dataset
python train.py --epochs=50 --dataset=foursquare-twitter
echo -----------------------------

echo Running on ACM-DBLP-P dataset
python train.py --epochs=50 --dataset=ACM-DBLP
echo -----------------------------