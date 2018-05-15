import templatematching
import os
import segmentation
from sklearn.externals import joblib

# load the model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

classification_result = []
for each_character in segmentation.characters:
    # converts it to a 1D array
    letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
    each_character = each_character.reshape(1, -1);
    result = model.predict(each_character)
    probabilities = model.predict_proba(each_character)
    result_index = letters.index(result[0])
    prediction_probability = probabilities[0, result_index]
    # template matching when necessary
    if result[0] in templatematching.confusing_chars and prediction_probability < 0.15:
        print 'here'
        result[0] = templatematching.template_match(result[0],
            each_character, os.path.join(os.path.dirname(os.path.realpath(
            __file__)), 'train'))
    classification_result.append(result)

print(classification_result)

plate_string = ''
for eachPredict in classification_result:
    plate_string += eachPredict[0]

print(plate_string)

# it's possible the characters are wrongly arranged
# since that's a possibility, the column_list will be
# used to sort the letters in the right order

column_list_copy = segmentation.column_list[:]
segmentation.column_list.sort()
rightplate_string = ''
for each in segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print(rightplate_string)
