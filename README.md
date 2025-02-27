# Analisis Sentimen pada Tweet

![Lencana status](https://img.shields.io/badge/Status-Archived-important)

**Pembaruan** (21 September 2018): Saya tidak secara aktif mengelola repositori ini. Pekerjaan ini dilakukan untuk proyek mata kuliah dan dataset tidak dapat dirilis karena saya tidak memiliki hak cipta. Namun, semua yang ada di repositori ini dapat dengan mudah dimodifikasi untuk digunakan dengan dataset lain. Saya sarankan untuk membaca [laporan proyek yang ditulis secara asal-asalan] (https://github.com/abdulfatir/twitter-sentiment-analysis/tree/master/docs/report.pdf) untuk proyek ini yang dapat ditemukan di `docs/`.

## Informasi Dataset

Kami menggunakan dan membandingkan berbagai metode yang berbeda untuk analisis sentimen pada tweet (masalah klasifikasi biner). Dataset pelatihan diharapkan berupa file csv bertipe `tweet_id,sentimen,tweet` di mana `tweet_id` adalah sebuah bilangan bulat unik yang mengidentifikasi tweet, `sentimen` adalah `1` (positif) atau `0` (negatif), dan `tweet` adalah tweet yang diapit oleh `“” `. Demikian pula, dataset pengujian adalah file csv bertipe `tweet_id,tweet`. Harap diperhatikan bahwa header csv tidak diharapkan dan harus dihapus dari dataset pelatihan dan pengujian.  

Persyaratan ##

Ada beberapa persyaratan pustaka umum untuk proyek ini dan beberapa persyaratan khusus untuk masing-masing metode. Persyaratan umum adalah sebagai berikut.  
* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`

Persyaratan pustaka khusus untuk beberapa metode adalah sebagai berikut:
* `keras` dengan backend `TensorFlow` untuk Regresi Logistik, MLP, RNN (LSTM), dan CNN.
* `xgboost` untuk XGBoost.

**Catatan**: Direkomendasikan untuk menggunakan distribusi Python Anaconda.

## Penggunaan

## Preprocessing 

1. Jalankan `preprocess.py <raw-csv-path>` pada data latih dan data uji. Ini akan menghasilkan versi praproses dari dataset.
2. Jalankan `stats.py <preprocessed-csv-path>` di mana `<preprocessed-csv-path>` adalah jalur csv yang dihasilkan dari `preprocess.py`. Ini memberikan informasi statistik umum tentang dataset dan akan menghasilkan dua file acar yang merupakan distribusi frekuensi dari unigram dan bigram dalam dataset pelatihan. 

Setelah langkah-langkah di atas, Anda akan memiliki empat file secara total: `<preprocessed-train-csv>`, `<preprocessed-test-csv>`, `<freqdist>`, dan `<freqdist-bi>` yang merupakan preprocessed train dataset, preprocessed test dataset, distribusi frekuensi unigram, dan distribusi frekuensi bigram secara berurutan.

Untuk semua metode berikut, ubah nilai `TRAIN_PROCESSED_FILE`, `TEST_PROCESSED_FILE`, `FREQ_DIST_FILE`, dan `BI_FREQ_DIST_FILE` ke jalur Anda sendiri di file masing-masing. Di mana pun berlaku, nilai `USE_BIGRAMS` dan `FEAT_TYPE` dapat diubah untuk mendapatkan hasil menggunakan berbagai jenis fitur seperti yang dijelaskan dalam laporan.

### Baseline
3. Jalankan `baseline.py`. Dengan `TRAIN = True` maka akan menampilkan hasil akurasi pada dataset training.

### Naive Bayes
4. Jalankan `naivebayes.py`. Dengan `TRAIN = True` maka akan menampilkan hasil akurasi pada dataset validasi 10%.

### Entropi Maksimum
5. Jalankan `logistic.py` untuk menjalankan model regresi logistik ATAU jalankan `maxent-nltk.py <>` untuk menjalankan model MaxEnt dari NLTK. Dengan `TRAIN = True` maka akan menampilkan hasil akurasi pada 10% dataset validasi.

### Pohon Keputusan
6. Jalankan `decisiontree.py`. Dengan `TRAIN = True` maka akan muncul hasil akurasi pada 10% dataset validasi.

### Random Forest
7. Jalankan `randomforest.py`. Dengan `TRAIN = True` maka akan menampilkan hasil akurasi pada 10% dataset validasi.

### XGBoost
8. Jalankan `xgboost.py`. Dengan `TRAIN = True` maka akan menampilkan hasil akurasi pada 10% dataset validasi.

### SVM
9. Jalankan `svm.py`. Dengan `TRAIN = True` maka akan menampilkan hasil akurasi pada 10% dataset validasi.

### Multi-Layer Perceptron
10. Jalankan `neuralnet.py`. Akan melakukan validasi menggunakan 10% data dan menyimpan model terbaik ke `best_mlp_model.h5`.

### Jaringan Syaraf Tiruan Berulang
11. Jalankan `lstm.py`. Akan memvalidasi menggunakan 10% data dan menyimpan model untuk setiap epock di `./models/`. (Pastikan direktori ini sudah ada sebelum menjalankan `lstm.py`).

### Jaringan Syaraf Tiruan Konvolusional
12. Jalankan `cnn.py`. Ini akan menjalankan model 4-Conv-NN (4 conv layers neural network) seperti yang dijelaskan dalam laporan. Untuk menjalankan versi lain dari CNN, cukup beri komentar atau hapus baris di mana lapisan Conv ditambahkan. Akan memvalidasi menggunakan 10% data dan menyimpan model untuk setiap epoch di `./models/`. (Pastikan direktori ini sudah ada sebelum menjalankan `cnn.py`). 

### Ensemble Suara Mayoritas
13. Untuk mengekstrak fitur lapisan kedua dari belakang untuk set data pelatihan, jalankan `extract-cnn-feats.py <saved-model>`. Ini akan menghasilkan 3 file, `train-feats.npy`, `train-labels.txt` dan `test-feats.npy`.
14. Jalankan `cnn-feats-svm.py` yang menggunakan file-file dari langkah sebelumnya untuk melakukan klasifikasi SVM pada fitur-fitur yang diekstrak dari model CNN.
15. Letakkan semua file CSV prediksi yang ingin Anda ambil suara mayoritasnya di `./results/` dan jalankan `mayoritas-voting.py`. Ini akan menghasilkan `mayoritas-pemungutan suara.csv`.

## Informasi tentang berkas-berkas lain

* `dataset/positive-words.txt`: List of positive words.
* `dataset/negative-words.txt`: List of negative words.
* `dataset/glove-seeds.txt`: GloVe words vectors from StanfordNLP which match our dataset for seeding word embeddings.
* `Plots.ipynb`: IPython notebook used to generate plots present in report.
