# Ethnicity Recognition
Realtime ethnicity recognition system, using *Viola-Jones* and dlib shape predictors for the detection; and models like *LightGBM*, *custom CNN*, *AlexNet*, and *SVM* for the classification. The classes that have been considered are:
- White
- Black
- Asian
- Indian
- Others (Middle Eastern, Latinos, ...)

The system is able to guess the ethnicity analyzing your eyes, nose and mouth through a video. Not only! The dataset UTKFace that we used to train our models, had information also about age and gender. So we will be able to predict them either!
If you want to try the demo, follow these steps:
1. download the trained models: you can try two versions, one trained with the CNN and another one with SVM. In both cases we suggest to not modify the structure and organization of the folders and files
   - demo > CNN > trained models link 
   - demo > SVM > trained_model_files.ipynb : download the files that you can find in "trained_model"
2. downlaod the code:
   - demo > CNN > demo_cnn.ipynb
   - demo > SVM > demo_svm.ipynb
3. download the shape predictor: Dlib > shape_predictors_from_dlib.ipynb . We sugges to use the "shape_predictor_68_face_landmarks.dat", due to the better performances

Remark: remember to rename the "folder_path" and "shape_predictor_path" that you will find in the demo_cnn.ipynb and demo_svm.ipynb files, based on your pc path.

Have fun! :)

UTKFace dataset: https://susanqq.github.io/UTKFace/
