import sdc.image_processing.feature_extraction as fe
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import _pickle as pkl


class SVMClassifier:
    """ SVM Classifier for object detection """
    def __init__(self, settings=None):
        self.settings = settings
        self.X_scaler = None
        self.classifier = None
        self.score = 0

    def train(self, file_list, labels, test_size=0.2, rand_range=(0, 100)):
        """ Train classifier """
        rand_state = np.random.randint(rand_range[0], rand_range[1])
        """ Train classifier from the given file list """
        X = fe.extract_features(file_list, self.settings)
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)
        # Split dataset into training and validation sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            scaled_X, labels, test_size=test_size, random_state=rand_state)
        self.classifier = LinearSVC()
        # Fit model
        self.classifier.fit(X_train, Y_train)
        # Validate results
        self.score = self.classifier.score(X_test, Y_test)
        # Return model accuracy
        return self.score

    def save(self, fname='svm_model.pkl'):
        """ Persist model """
        # Save model, scaler, settings
        obj = {'model': self.classifier, 'scaler': self.X_scaler, 'settings': self.settings}
        with open(fname, 'wb') as f:
            pkl.dump(obj, f)

    def load(self, fname='svm_model.pkl'):
        """ Load model from file """
        with open(fname, 'rb') as f:
            obj = pkl.load(f)
            # Model
            self.classifier = obj['model']
            # Scaler
            self.X_scaler = obj['scaler']
            # Settings on which model was trained
            self.settings = obj['settings']

    def predict(self, features):
        """ Predict if given image contains object of interest """
        # Predict using classifier
        scaled_X = self.X_scaler.transform(np.array(features).reshape(1, -1))
        return self.classifier.predict(scaled_X)
