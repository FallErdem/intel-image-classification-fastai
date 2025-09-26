Intel Image Classification (6 sınıf) · fastai + ResNet18

Amaç: Doğal/yapay ortam görsellerini (buildings, forest, glacier, mountain, sea, street) CNN tabanlı bir modelle sınıflandırmak.
Teknolojiler: fastai v2 · PyTorch · Kaggle GPU

Bu repo; Bootcamp isterlerini kapsar: veri önişleme + augmentation, CNN (ResNet18, transfer learning), eğitim grafikleri, Confusion Matrix & Classification Report, Grad-CAM, küçük HPO ve model kaydetme.

🔗 Bağlantılar

Kaggle Notebook: https://www.kaggle.com/code/fallerdem/globalai/edit

Model Ağırlıkları (Release): https://github.com/FallErdem/intel-image-classification-fastai/releases/tag/v0.1.0

Veri seti (Kaggle): puneet6060/intel-image-classification

🗂️ Proje Yapısı
notebooks/
  Intel_Image_Classification.ipynb     # tüm akış (Kaggle’dan indirildi)
  export.pkl                           # (opsiyonel) fastai export - inference için
reports/
  confusion_matrix.PNG                 # karışıklık matrisi
  gradcam_example.png                  # Grad-CAM örneği


Not: GitHub’ın 100 MB sınırı nedeniyle büyük .pth dosyası releases altında paylaşılmıştır (link yukarıda). export.pkl ile tek dosyada inference yapmak da mümkündür.

🧭 Yöntem (Özet)

Veri önişleme: fastai DataBlock, train/valid = 80/20, 224×224;
Augmentations: flip, rotate(±8°), zoom(1.1), lighting(0.2)

Model: ResNet18 (ImageNet ön-eğitimli) → Transfer Learning

Eğitim:

LR Finder ile aralık seçimi

12 epoch (frozen) → 5 epoch (unfreeze, küçük LR)

Değerlendirme: Accuracy/Loss grafikleri, Confusion Matrix, Classification Report

Açıklanabilirlik: Grad-CAM (son konv katmanından ısı haritası)

HPO (mini-grid): farklı lr_slice ve weight_decay denemeleri

Kayıt: export.pkl (fastai learner), en iyi ağırlık .pth (Release’te)

📊 Sonuçlar

Validation Accuracy (en iyi): ≈ 0.948

Test Accuracy: ≈ 0.87 (3000 görüntü)

En çok karışan sınıflar: glacier ↔ mountain/sea (benzer doku & ufuk çizgisi)

Confusion Matrix


Grad-CAM örneği


🚀 Çalıştırma
Kaggle (önerilen)

New Notebook → Add Data: puneet6060/intel-image-classification

GPU’yu aç (Settings → Accelerator → GPU).

Bu repodaki notebook’u yükle veya Kaggle’a import et.

Hücreleri sırayla çalıştır; çıktılar /kaggle/working/ altında oluşur.

Lokal inference (yalnızca tahmin)

Seçenek A — export.pkl ile (tek dosya, en pratik):

from fastai.vision.all import *
learn = load_learner('notebooks/export.pkl')
pred, _, probs = learn.predict(PILImage.create('any_test.jpg'))
print(pred, float(probs.max()))


Seçenek B — Release’ten .pth ile:

from fastai.vision.all import *

# dls'i (224px) oluşturduğun varsayılıyor (aynı augment/normalize ile)
learn = vision_learner(dls, resnet18, metrics=[accuracy],
                       path=Path('.'), model_dir=Path('models')).to_fp32()

# models/resnet18-intel-best.pth dosyasını indirip buraya koy
learn.load('resnet18-intel-best', with_opt=False)

pred, _, probs = learn.predict(PILImage.create('any_test.jpg'))
print(pred, float(probs.max()))


learn.load("resnet18-intel-best") dosyayı models/resnet18-intel-best.pth yolunda arar.

✅ Bootcamp İsterleri Eşlemesi

Kaggle notebook + GitHub repo + README ✅

Veri önişleme + Data Augmentation ✅

CNN tabanlı model (ResNet18, TL) ✅

Metrikler: Accuracy/Loss grafikleri, Confusion Matrix, Classification Report ✅

Grad-CAM görselleştirme ✅

Hiperparametre denemeleri (mini-grid) ✅

Model kaydetme (export.pkl / .pth) ✅

(Bonus) TensorBoard (trace_model=False) ✅

🔮 Geliştirme Fikirleri

Daha güçlü mimariler: ResNet34/50, EfficientNet, ConvNeXt

Mixup/CutMix, RandomErasing, TTA

Sınıf dengeleme: class-weight, focal loss, oversampling

W&B / MLflow ile deney takibi ve model izleme
