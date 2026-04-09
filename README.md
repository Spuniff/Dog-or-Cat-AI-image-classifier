# Dog-or-Cat-AI-image-classifier
Created an image classifier using TensorFlow and Keras to distinguish between cats and dogs. Trained on labeled datasets with preprocessing and validation splitting. Outputs predictions with confidence scores for new images.



How to install:
1. Clone the repository:
git clone https://github.com/yourusername/image-classifier.git

2. create data folder with subfolders Cat, Dog. Put training data here, Create models folder, most recent model will go here. Create Test folder, Images you want to use to test your model go here.

  
3. Navigate into:
cd image-classifier

4. Create your virtual environment:
python -m venv venv

5. Activate:
for windows machines ---> venv\Scripts\activate
for mac/Linix machines -> source venv/bin/activate

6 Install dependencies:
pip install -r requirements.txt

7. Run training script:
python main.py

keep in mind you have to set TRAIN_MODEL = True/False for if you want to retrain a new model or if you want to test your last one.

Learned: 
1. how to build and train a CNN using TensorFlow
2. Handling image datasets
3. Model evaluation and accuracy tuning

License
This project is licensed under the MIT License.
