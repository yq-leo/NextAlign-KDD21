echo Running on Phone-Email dataset
python train.py --epochs=5 --dataset=phone-email
echo -----------------------------

echo Running on Foursquare-Twitter dataset
python train.py --epochs=5 --dataset=foursquare-twitter
echo -----------------------------

echo Running on ACM-DBLP dataset
python train.py --epochs=1 --dataset=ACM-DBLP
echo -----------------------------