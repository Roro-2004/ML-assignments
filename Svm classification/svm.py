import os
import glob
from unittest import result
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

#LOAD DATA
train_data = "D:\\uni\\Machine Learning\\Assignment\\ML-assignments\\Svm classification\\Assignment dataset\\train"
test_data = "D:\\uni\\Machine Learning\\Assignment\\ML-assignments\\Svm classification\\Assignment dataset\\test"

#ENCODE
My_classes = ['accordian', 'dollar_bill', 'motorbike', 'soccer_ball']
class_to_label = {cls: idx for idx, cls in enumerate(My_classes)}
label_to_class = {idx: cls for cls, idx in class_to_label.items()}

X_train, y_train, X_test, y_test = [], [], [], []
#FUNCTIONS
def load_data (classes, folder_train_test, train_or_test):
    for c in classes:
        folder = os.path.join(folder_train_test, c)
        for img_file in glob.glob(os.path.join(folder, '*.jpg')):
            
            image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(image, (128, 64))
            fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
            if(train_or_test == "train"):
                X_train.append(fd)
                y_train.append(class_to_label[c])
            else:
                 X_test.append(fd)
                 y_test.append(class_to_label[c])
def train_and_evaluate_all_kernels(X_train, y_train, X_test, y_test, param_grids, classes=None, verbose=True):
    kernels = list(param_grids.keys())
    results = {}

    if classes is None:
        classes = np.unique(y_test)

    for kernel in kernels:
        # Get the model and grid
        model = SVC(kernel=kernel, probability=True, random_state=42)
        params = param_grids[kernel]

        if verbose:
            print(f"\nSVM with {kernel} kernel:")
            print("=" * 100)

        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        train_acc = best_model.score(X_train, y_train)
        test_acc = best_model.score(X_test, y_test)
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        results[kernel] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'model': best_model,
            'confusion_matrix': cm 
        }

        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            print(f"Training Accuracy: {train_acc * 100:.2f}%")
            print(f"Testing Accuracy: {test_acc * 100:.2f}%")
            #print(f"Condusion matrix: {cm}")
            

    return results
def get_best_kernel(results):
    best_kernel = None
    best_accuracy = 0
    
    for kernel, metrics in results.items():
        test_acc = metrics['test_accuracy']
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_kernel = kernel
            
    return best_kernel, best_accuracy



load_data(My_classes,train_data,"train")
X_train = np.array(X_train)
y_train = np.array(y_train)
load_data(My_classes,test_data,"test")
X_test = np.array(X_test)
y_test = np.array(y_test)

param_grids = {
    'linear': {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10],
        'class_weight': ['balanced', None]
    },
    'rbf': {
        'C': [0.001, 0.01, 0.1, 1, 5, 10, 100],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
        'class_weight': ['balanced', None]
    },
    'poly': {
        'C': [0.001, 0.01, 0.1, 1, 5, 10],
        'degree': [1, 2, 3, 4, 5],
        'coef0': [-1, 0, 1, 2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'class_weight': ['balanced', None]
    },
    'sigmoid': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'coef0': [-1, 0, 1, 2],
        'class_weight': ['balanced', None]
    }
}
svm_results = train_and_evaluate_all_kernels(X_train,y_train,X_test,y_test,param_grids,My_classes,verbose=True)
best_kernel, best_acc = get_best_kernel(svm_results)

print(f"\n Best Kernel: '{best_kernel}' with Test Accuracy = {best_acc * 100:.2f}%")
