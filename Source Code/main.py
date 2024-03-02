
from mediapipe.tasks.python import vision
from feature_extraction import get_all_features
from sklearn.neighbors import KNeighborsClassifier
import scipy.spatial.distance as distance
import numpy as np
import cv2 
from matplotlib import pyplot as plt 

def transform_data(features):
    labels = []
    feat = np.ndarray(shape=(len(features), len(features[0].feature)))
    
    for i, refs in enumerate(features):
        labels.append(refs.name)
        feat[i][:] = refs.feature[:]
    
    return (labels, feat)


def most_frequent(List):
    return max(set(List), key = List.count)

model = vision.ImageEmbedder.create_from_model_path("models/FaceNet.tflite")
REFERENCE_FEATURES = get_all_features("dataset/references", model)
TEST_FEATURES = get_all_features("dataset/test", model)

train_labels, train_features = transform_data(REFERENCE_FEATURES)
test_labels, test_features = transform_data(TEST_FEATURES)

clf = KNeighborsClassifier()
clf.fit(train_features, train_labels)

distances, indices = clf.kneighbors(test_features,  n_neighbors=7)
print(distances)
fig, axs = plt.subplots(nrows=len(indices), ncols=1 + len(indices[0]), figsize=(20, 20))

cols = ["Test image"]

cols.extend(['Prediction {}'.format(col) for col in range(1, len(indices[0]) + 1)])


pad = 5




predicitons = []

for i, ind in enumerate(indices):

    test_image_index = i

    test_image = cv2.imread(TEST_FEATURES[test_image_index].image_path) 
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    axs[i, 0].imshow(test_image)
    axs[i, 0].set_xlabel("bla")
    axs[i,0].tick_params(left=False, labelleft=False)
    axs[i,0].xaxis.set_visible(False)
    
    current_predictions = []
    for j, index in enumerate(ind):
        train_image = cv2.imread(REFERENCE_FEATURES[index].image_path) 
        train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
        axs[i, j + 1].imshow(train_image)
        axs[i, j + 1].tick_params(left=False, labelleft=False)
        axs[i, j + 1].xaxis.set_visible(False)
        current_predictions.append(train_labels[index])
    
    predicitons.append(most_frequent(current_predictions))


for i, (ax, col) in enumerate(zip(axs[0], cols)):
    if(i == 0):
        ax.annotate("Prediction / Correct Label         " + col, xy=(0.5, 1), xytext=(30, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='right', va='baseline')
    else:
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

for i, (ax, prediction, label) in enumerate(zip(axs[:,0], predicitons, test_labels)):
        
    ax.annotate(prediction + ' / ' + label, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

hits = 0
for i in range(len(predicitons)):
  if predicitons[i] == test_labels[i]:
    hits += 1

acc_test = hits/len(predicitons)
print(f'Accuracy:  {acc_test:.2f}')


plt.show()

