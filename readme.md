# Domain Proyek
Diabetes adalah penyakit kronis yang secara langsung memberikan dampak pada pankreas [1].
Penyakit diabetes dapat menyebabkan penyakit lainnya pada kulit, syaraf, dan mata. Apabila tidak ditangani dengan baik diabetes dapat menyebabkan gagal ginjal. Berdasarkan IDF (International Diabetes Federation) statistik, sebanyak 500 juta orang telah mengidap diabetes pada tahun 2021 [2].

Melakukan deteksi dini dan pencegahan terhadap diabetes secara signifikan dapat membantu pasien untuk mengurangi biaya pengobatan dan rumah sakit. Tantangan terbesar saat ini yaitu bagaimana merancang sebuah model machine learning yang dapat memprediksi seseorang yang memiliki risiko penyakit diabetes atau tidak berdasarkan catatan medis.
# Business Understanding

## Problem Statement
Berdasarkan latar belakang di atas, maka saya mendapatkan problem statement sebagai berikut:
- Bagaimana cara membuat sebuah model yang dapat memprediksi seseorang yang memiliki risiko penyakit diabetes menggunakan catatan medis dengan akurat?
- Apa saja faktor utama yang dapat menyebabkan seseorang memiliki risiko penyakit diabetes?
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
## EDA
Pada proses Exploratory Data Analysis (EDA) bertujuan untuk menganalisis karakteristik dan menemukan pola pada data yang digunakan. Pada proses EDA ini saya membagi menjadi dua bagian yaitu Univariate dan Multivariate Analysis.
### Univariate Data Analysis

![age_dist](https://github.com/user-attachments/assets/026c4c45-1c20-45b9-bd07-7082506f6541)
Terlihat pada grafik di atas menampilkan distribusi umur dari dataset yang digunakan. Dapat terlihat bahwa orang yang memiliki risiko penyakit diabetes memiliki umur lebih dari 30 tahun.
### Multivariate Data Analysis
![blood_glu](https://github.com/user-attachments/assets/729eca3e-c2d2-4884-9d51-c6fe16f8805f)
Pada grafik tersebut dapat terlihat bahwa orang yang memiliki risiko penyakit diabetes cenderung memiliki Blood Glucose Level (level gula darah) yang lebih tinggi.
![corr_matrix](https://github.com/user-attachments/assets/d97a9f6c-b1b8-4e34-aa67-49de579bae19)
Pada grafik di atas dapat terlihat bahwa HbA1c Level dan Blood Glucose Level memiliki korelasi tertinggi dibandingkan fitur lainnya, ini menggambarkan bahwa orang yang memiliki HbA1c Level dan Blood Glucose Level cenderung memiliki risiko penyakit diabetes.
# Data Preperation

## Encoding Features
Agar fitur pada dataset dapat digunakan untuk memprediksi diabetes, saya perlu merubah beberapa data pada kolom "age" dan "smoking_history" dari categorical features menjadi numerical features menggunakan Label Encoder dari library Scikit-learn.

Pada dasarnya Label Encoder bertujuan untuk merubah data dari 0 sampai dengan n-class dari dataset. Sebagai contoh jika kita memiliki data gender "Male" dan "Female" dapat diubah menjadi 0 dan 1 sehingga model dapat membaca data tersebut.

```Python
data['gender'] = ['Male', 'Female']
data['gender'] = encoder.fit_transform(data['gender'])
data['gender']

output: [0, 1]
```

| gender | age  | hypertension | heart_disease | smoking_history | bmi   | HbA1c_level | blood_glucose_level | diabetes |
| ------ | ---- | ------------ | ------------- | --------------- | ----- | ----------- | ------------------- | -------- |
| 0      | 80.0 | 0            | 1             | 4               | 25.19 | 6.6         | 140                 | 0        |
| 0      | 54.0 | 0            | 0             | 0               | 27.32 | 6.6         | 80                  | 0        |
| 1      | 28.0 | 0            | 0             | 4               | 27.32 | 5.7         | 158                 | 0        |
| 0      | 36.0 | 0            | 0             | 1               | 23.45 | 5.0         | 155                 | 0        |
| 1      | 76.0 | 1            | 1             | 1               | 20.14 | 4.8         | 155                 | 0        |
## Under Sampling
![imba](https://github.com/user-attachments/assets/d72b598c-6975-4041-a424-73803afd4b75)
Pada grafik di atas dapat terlihat bahwa dataset yang digunakan ternyata memiliki ketidakseimbangan antara orang yang tidak memiliki risiko penyakit diabetes dengan orang yang memiliki risiko penyakit diabetes. Hal tersebut dapat mempengaruhi hasil prediksi dari model yang akan dibuat dan akan menciptakan bias.

Salah satu metode untuk mengatasi data yang tidak seimbang adalah metode Under Samping, yaitu mengurangi jumlah mayoritas data agar sama dengan data minoritas. Untuk melakukan metode under sampling dapat menggunakan library Imblearn.

```Python
#import library
from imblearn.under_sampling import RandomUnderSampler
# inisialisasi 
rus = RandomUnderSampler()

# setup data yang akan di undersample
X = df.drop(columns='diabetes')
y = df['diabetes']

# menerapkan under sampling dan menyimpad hasil pada variabel yang baru
X_resampled, y_resampled = rus.fit_resample(X,y)
```
## Split Dataset
Dikarenakan tidak adanya test dataset yang diberikan maka saya perlu membagi dataset yang saya gunakan menjadi dua yaitu train_set dan test_set menggunakan train_test_split dari library Scikit-Learn. Pada kasus ini saya menggunakan train_test_split untuk membadi dataset menjadi train_set dan test_set dengan ukuran test_set sebesar 20% dari jumlah dataset. Sehingga rasio train_set dengan test_set menjadi 80:20.

```Python
from sklearn.model_selection import train_test_split

X = df.drop(columns='diabetes')
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
# Modelling
Pada tahap modelling dikarenakan pada projek ini bertujuan untuk memprediksi apakah seseorang memiliki risiko diabetes atau tidak berarti untuk kasus ini termasuk kasus klasifikasi sebab kita harus membedakan antara orang yang memiliki risiko penyakit diabetes dengan yang tidak. 

Untuk melakukan prediksi tersebut saya menggunakan linear model dari Scikit-learn yaitu Random Forest Classifier dengan basic parameter. Random Forest adalah salah satu model yang cukup sering digunakan untuk melakukan klasifikasi. Random Forest bisa memberikan hasil yang lebih akurat karena dapat menggabungkan hasil dari banyak Decision Tree yang berbeda. Selain itu, model Random Forest juga cukup tahan terhadap overfitting jika dibandingkan dengan model ensemble learning lainnya.

# Evaluasi
Setelah melakukan training dengan train_set menggunakan model Random Forest, saya melakulan evaluasi dengan metrics classification_report dari library Scikit-learn untuk mengecek hasil dari model yang sudah di train dengan memberikan dataset test. 

Dalam fungsi classification_report dapat menampilkan beberapa metrcis seperti berikut:
- Presisi digunakan untuk mengukur seberapa dapat diandalkan sebuah model ketika memberikan prediksi terhadap suatu kelas/_target_. 
	Rumus Precison: $\dfrac{True Positive}{True Positive + False Positive}$
- _Recall_ digunakan untuk mengukur kemampuan model untuk memprediksi kelas _True Positive
	Rumus Recall: $\dfrac{True Positive}{True Positive + False Negative}$
- _F1-Score_ digunakan untuk mencari titik seimbang antara Presisi dan _Recall
	Rumus F1-Score: $\dfrac{2 \ast Precision \ast Recall}{Precision+Recall}$

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 1.00      | 0.97   | 0.98     | 18751   |
| 1            | 0.68      | 0.94   | 0.80     | 1249    |
| accuracy     |           |        | 0.97     | 20000   |
| macro avg    | 0.84      | 0.96   | 0.89     | 20000   |
| weighted avg | 0.98      | 0.97   | 0.97     | 20000   |

Dari tabel classification_report tersebut dapat kita ambil bahwa:
- Precision: Model yang saya gunakan dapat dengan baik untuk memprediksi dataset terhadap orang yang tidak memiliki diabetes sebesar 100%, sedangkan untuk data  terhadap orang yang memiliki diabetes sebesar 68%.
- Recall: Model yang saya gunakan mendapatkan hasil untuk data tidak memiliki diabetes sebesar 97%, sedangkan untuk orang yang memiliki diabetes sebesar 94%.
- F1-Score: Dari model yang saya gunakan telah berhasil memprediksi data untuk tidak memiliki diabetes sebesar 98% sedangkan untuk data orang yang memiliki diabetes sebesar 80%
## Kesimpulan
Dari prediksi yang telah saya lakukan menggunakan model Random Forest dapat diambil kesimpulan sebagai berikut:
- Telah berhasil membuat sebuah model machine learning yang dapat melakukan prediksi apakah seseorang mengalami risiko penyakit diabetes dengan mendapatkan akurasi sebesarr 97%.
- Berdasarkan Exploratory Data Analysis yang telah dilakukan saya menemukan beberapa faktor utama yang dapat menyebabkan seseorang memiliki risiko penyakit diabetes. Faktor utama tersebut adalah tingginya HbA1c Level dan Blood Glucose Level, hal ini dapat terjadi dikarenakan tingginya level gula darah merupakan indikasi seseorang memiliki risiko penyakit diabetes.
# Refrensi
[1] Kharroubi, A.T., Darwish, H.M.: Diabetes mellitus: The epidemic of the
century. World J. Diabetes 6, 850–867 (2015).
[2] Atlas, G.: Diabetes. International Diabetes Federation. 10th ed., IDF
Diabetes Atlas.