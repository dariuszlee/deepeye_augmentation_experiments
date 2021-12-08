# deepEye_data_augmentation_research_module

## Research Question
The goal of the RM project is to find a transformation of the training data, to train a model that performs well on unseen test data. To do this you are allowed to transform/enhence the training data. You are allowed to extend the data set. You are not allowed to change the model or the test data. Here are some examples what can be done with the data:
* delete samples (perform outlier detection, ...)
* augment the data (add random/specific noise)
* use methods to create artificial examples (GAN, ...)
* ...

## ToDo
You need to implement your transformation(s) in DataAugmentation/data_augmentation.py and call this function(s) in the notebook deepEye_data_augmentation_research_module.ipynb.
