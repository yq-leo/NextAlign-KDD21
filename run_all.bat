echo Running on Phone-Email dataset
python train.py --epochs=200 --dataset=phone-email
echo -----------------------------

echo Running on Foursquare-Twitter dataset
python train.py --epochs=200 --dataset=foursquare-twitter
echo -----------------------------

echo Running on ACM-DBLP dataset
python train.py --epochs=200 --dataset=ACM-DBLP
echo -----------------------------