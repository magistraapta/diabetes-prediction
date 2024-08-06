# Domain Proyek
Projek ini mengusung tema kesehatan, yang berfokus pada mendeteksi penderita diabetes berdasarkan catatan medis. Tujuan utama projek ini yaitu membuat sebuah model machine learning yang dapat memprediksi apakah seseorang menderita diabetes berdasarkan catatan medis.

# Business Understanding

## Problem Statement
Diabetes adalah salah satu penyakit kronis yang menyerang hampir jutaan orang di seluruh dunia. Melakukan deteksi dini dan pencegahan terhadap diabetes secara signifikan dapat membantu pasien untuk mengurangi biaya pengobatan dan rumah sakit. Tantangan terbesar saat ini yaitu bagaimana merancang sebuah model machine learning yang dapat memprediksi seseorang yang memiliki risiko penyakit diabetes atau tidak berdasarkan catatan medis.
## Goals
- Membangun sebuah model machine learning dengan akurasi tinggi yang dapat memprediksi risiko penyakit diabetes.
- Mengidentifikasi faktor utama yang dapat menyebabkan diabetes
# Data Understanding
Pada projek ini saya menggunakan dataset yang saya peroleh dari Kaggle yang berjudul [**Diabetes Prediction Dataset** ](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) Dataset berisikan 100.000 baris catatan medis dengan 9 kolom dan file dataset ini memiliki ukuran sebesar 3.8MB.

Variabel-variable yang ada pada dataset tersebut adalah sebagai berikut:
- Gender: Berisikan data berupa jenis kelamin daripada seseorang
- Age: Berisikan data berupa umur
- Hypertension: Berisikan catatan medis yang menggambarkan bahwa orang tersebut pernah mengalami hipertensi atau tidak, jika tidak pernah maka akan bernilai 0 dan jika pernah maka akan bernilai 1
- Heart Disease: Berisikan catatan medis yang menggambarkan bahwa orang tersebut pernah mengalami serangan jantung atau tidak, jika tidak pernah maka akan bernilai 0 dan jika pernah maka akan bernilai 1
- Smoking_history: Berisikan data berupa kondisi seseorang apakah ia seorang perokok atau tidak
- BMI: Berisikan data berupa jumlah lemak pada tubuh yang berdasarkan pada tinggi dan berat badan.
- HbA1c_level: Berisikan data berupa rata-rata ukuran gula darah seseorang pada 2-3 bulan terakhir
- Blood Glucose Level: Berisikan data yang menggambarkan jumlah gula darah pada tubuh.
- Diabetes: Merupakan target prediksi dari dataset ini yang berisikan tentang apakah orang tersebut menderita diabetes atau tidak,  jika tidak pernah maka akan bernilai 0 dan jika pernah maka akan bernilai 1
# Data Preperation

## Encoding Features
Agar fitur pada dataset dapat digunakan untuk memprediksi diabetes, saya perlu merubah beberapa data pada kolom "age" dan "smoking_history" dari categorical features menjadi numerical features menggunakan Label Encoder dari library Scikit-learn

| gender | age | hypertension | heart_disease | smoking_history | bmi | HbA1c_level | blood_glucose_level | diabetes |     |
| ------ | --- | ------------ | ------------- | --------------- | --- | ----------- | ------------------- | -------- | --- |
| 0      | 0   | 80.0         | 0             | 1               | 4   | 25.19       | 6.6                 | 140      | 0   |
| 1      | 0   | 54.0         | 0             | 0               | 0   | 27.32       | 6.6                 | 80       | 0   |
| 2      | 1   | 28.0         | 0             | 0               | 4   | 27.32       | 5.7                 | 158      | 0   |
| 3      | 0   | 36.0         | 0             | 0               | 1   | 23.45       | 5.0                 | 155      | 0   |
| 4      | 1   | 76.0         | 1             | 1               | 1   | 20.14       | 4.8                 | 155      | 0   |
## Split Dataset
Dikarenakan tidak adanya test dataset yang diberikan maka saya perlu membagi dataset yang saya gunakan menjadi dua yaitu train_set dan test_set menggunakan train_test_split dari library Scikit-Learn.

```Python
from sklearn.model_selection import train_test_split

X = df.drop(columns='diabetes')
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
# Modelling
Pada tahap modelling dikarenakan pada projek ini bertujuan untuk memprediksi apakah seseorang memiliki risiko diabetes atau tidak berarti untuk kasus ini termasuk kasus klasifikasi sebab kita harus membedakan antara orang yang memiliki risiko penyakit diabetes dengan yang tidak. 

Untuk melakukan prediksi tersebut saya menggunakan linear model dari Scikit-learn yaitu Random Forest Classifier. Random Forest adalah salah satu model yang cukup sering digunakan untuk melakukan klasifikasi. Random Forest bisa memberikan hasil yang lebih akurat karena dapat menggabungkan hasil dari banyak Decision Tree yang berbeda. Selain itu, model Random Forest juga cukup tahan terhadap overfitting jika dibandingkan dengan model ensemble learning lainnya
# Evaluasi
Setelah melakukan training dengan train_set menggunakan model Random Forest, saya melakulan evaluasi dengan metrics classification_report dari library Scikit-learn untuk mengecek hasil dari model yang sudah di train dengan memberikan dataset test. 

Dalam fungsi classification_report dapat menampilkan beberapa metrcis seperti berikut:
- Presisi digunakan untuk mengukur seberapa dapat diandalkan sebuah model ketika memberikan prediksi terhadap suatu kelas/_target_.
- _Recall_ digunakan untuk mengukur kemampuan model untuk memprediksi kelas _True Positive
- _F1-Score_ digunakan untuk mencari titik seimbang antara Presisi dan _Recall

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 1.00      | 0.97   | 0.98     | 18751   |
| 1            | 0.68      | 0.94   | 0.80     | 1249    |
| accuracy     |           |        | 0.97     | 20000   |
| macro avg    | 0.84      | 0.96   | 0.89     | 20000   |
| weighted avg | 0.98      | 0.97   | 0.97     | 20000   |

![confusion_matrix](https://github.com/user-attachments/assets/7f48b0b5-5cf7-484e-a041-599e2df4567e)
