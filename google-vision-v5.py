import io
import os
from google.cloud import vision_v1 as vision
import pandas as pd
import pathlib

local_image = 'F:/Python/Kuvat/logos.jpg'
filename = os.path.basename(local_image)
csv_file = "google-results.csv"
file = pathlib.Path(csv_file)

def save_csv(df_new):
    if file.exists ():
        try:
            df_from_csv = pd.read_csv(csv_file)
            frames = [df_from_csv, df_new]
            df_edited = pd.concat(frames)
            df_edited.to_csv(csv_file, index=False)

        except pd.errors.EmptyDataError:
            df_new.to_csv(csv_file, index=False)

    else:
        print('No file, creating new')
        df_new.to_csv(csv_file, index=False)

def labels(image, client):
    response = client.label_detection(image=image)
    labels = [label.description for label in response.label_annotations]
    return labels

def objects(image, client):
    response = client.object_localization(image=image)
    objects = [object.name for object in response.localized_object_annotations]
    return objects    

def landmarks(image, client):
    response = client.landmark_detection(image=image)
    landmarks = [landmark.description for landmark in response.landmark_annotations]
    return landmarks

def logos(image, client):
    response = client.logo_detection(image=image)
    logos = [logo.description for logo in response.logo_annotations]
    return logos

def safesearch(image, client):
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                    'LIKELY', 'VERY_LIKELY')
    response = client.safe_search_detection(image=image)            
    filter_violence = dict(violence='{}'.format(likelihood_name[response.safe_search_annotation.violence]))
    violence_value = filter_violence['violence']
    return violence_value

def faces(image, client):
    response = client.face_detection(image=image)           
    face_count = len(response.face_annotations)
    return face_count

def web(image, client):
    response = client.web_detection(image=image)
    best_guess = [label.label for label in response.web_detection.best_guess_labels]          
    return best_guess

def texts(image, client):
    response = client.text_detection(image=image)
    texts = [text.description for text in response.text_annotations]
    return texts    

def main():
    client = vision.ImageAnnotatorClient()
    with io.open(local_image, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    df = pd.DataFrame()
    df['Filename'] = [filename]
    df['Labels'] = [labels(image,client)]
    df['Objects'] = [objects(image,client)]
    df['Websearch'] = [web(image,client)]
    df['Logos'] = [logos(image,client)]
    df['Violence'] = [safesearch(image,client)]
    df['Faces'] = [faces(image,client)]
    df['Landmarks'] = [landmarks(image,client)]
    df['Texts'] = [texts(image,client)]
    save_csv(df)
    
if __name__ == "__main__":
    main()