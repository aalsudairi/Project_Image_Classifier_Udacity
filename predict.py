import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

def process_numpy_image(numpy_image):
  processed_image = tf.convert_to_tensor(numpy_image, dtype=tf.float32)
  processed_image = tf.image.resize(processed_image, (image_size, image_size))
  processed_image /= 255
  return processed_image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_numpy_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(expanded_image)
    probs, labels = tf.nn.top_k(predictions, k=top_k)
    probs = list(probs.numpy()[0])
    labels = list(labels.numpy()[0])
    
    return probs, labels, processed_image


def load_class_names(category_names_path):
    with open(category_names_path, 'r') as f:
        return json.load(f)
    
    
def print_result(image_probs, image_classes):
    for i, result in enumerate(image_probs):
        print('\n')
        print('Label: ', image_classes[i])
        print('Confidance: {:.2%}'.format(result))
        print('\n')
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Prediction of flower label')

    parser.add_argument('--input', action='store', dest='input', default='./test_images/cautleya_spicata.jpg')
    parser.add_argument('--model', action='store', dest='model', default='./1625998552.h5')
    parser.add_argument('--top_k', action='store', dest='top_k', default=5, type=int)
    parser.add_argument('--category_names', action='store', dest="category_names", default='./label_map.json')

    args = parser.parse_args()

    input_image_path = args.input
    model_path = args.model
    top_k = args.top_k
    category_names_path = args.category_names

    model = reloaded_keras_model = tf.keras.models.load_model('./1625998552.h5', custom_objects={'KerasLayer':hub.KerasLayer},compile=False)


    image_probs, image_classes = predict(input_image_path, model, top_k)

    class_names = load_class_names(category_names_path)

    processed_class_names = []
    for label in image_classes:
        processed_class_names.append(class_names[label])

    print_result(image_probs, processed_class_names)

