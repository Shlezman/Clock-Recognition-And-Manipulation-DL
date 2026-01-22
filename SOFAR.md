# digital watch



## yolo model (identify hour and minutes)

### created synthetic digital watch dataset (generator)
### tuned yolo model on the dataset to identify hours and minutes
### created a framework to manually label real digital watches 
### fine tuned the yolo model on the new dataset (could add more in th future for more accurecy)


## SVHN-model
### created a generator for implementing 7-seg digits type into the svhn dataset
### trained cnn model on the infused dataset

TODO:
- make the yolo model more accurate by hand label more resl digital clock pictures and fine-tune the yolo model
- create a function that can turn time in int format to a basic analog clock scatch
- create a generator for the generative model that contain pairs (input and 'label' and also the analog watch sketch) of different types of generated analog watches on 400 different hours
- train pix2pix model
- wrap it all up with an end to end pipeline

