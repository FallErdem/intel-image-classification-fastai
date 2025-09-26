Intel Image Classification (6 sınıf) · fastai + ResNet18

Amaç: (buildings, forest, glacier, mountain, sea, street) sınıflarını CNN ile sınıflandırmak.
Teknolojiler: fastai v2 · PyTorch · Kaggle GPU

Bu repo, Bootcamp isterlerini kapsar: veri önişleme + augmentation, CNN (ResNet18, TL), eğitim grafikleri, Confusion Matrix & Classification Report, Grad-CAM, küçük HPO, model kaydetme.
Büyük model dosyası (.pth) GitHub Releases altında paylaşılmıştır.

🔗 Bağlantılar

Kaggle Notebook: <https://github.com/FallErdem/intel-image-classification-fastai/releases/tag/v0.1.0>

Veri Seti: puneet6060/intel-image-classification (Kaggle)

Model Ağırlığı (Releases): Releases → “Assets” altında resnet18-intel-best.pth

📁 Proje Yapısı (öneri)
notebooks/
  Intel_Image_Classification.ipynb
  export.pkl                      # (opsiyonel) fastai export - inference için
reports/
  confusion_matrix.PNG
  gradcam_example.png


Not: GitHub 100 MB sınırı nedeniyle .pth repoda değil, Releases altında paylaşıldı. export.pkl varsa tek dosya ile inference yapmak daha pratiktir.

🧭 Yöntem (Özet)

Veri önişleme: DataBlock, 80/20 train/valid; 224×224; augmentation (flip, rotate±8°, zoom 1.1, lighting 0.2)

Model: ResNet18 (ImageNet ön-eğitimli) → Transfer Learning

Eğitim: fit_one_cycle → 12 epoch (frozen) + 5 epoch (unfreeze)

Değerlendirme: Accuracy/Loss grafikleri, Confusion Matrix, Classification Report

Açıklanabilirlik: Grad-CAM (örnek overlay)

HPO (mini): farklı lr_slice ve weight_decay

📊 Sonuçlar

Validation Accuracy (en iyi): ≈ 0.948

Test Accuracy: ≈ 0.87 (3.000 görüntü)

Zorlanan sınıflar: glacier ↔ mountain/sea (benzer doku/ufuk çizgisi)

Confusion Matrix


Grad-CAM Örneği


🚀 Çalıştırma
Kaggle (önerilen)

New Notebook → Add Data: puneet6060/intel-image-classification

GPU’yu aç → hücreleri sırayla çalıştır.

Çıktılar /kaggle/working/ altında oluşur (grafikler & modeller).

Lokal inference (sadece tahmin, export.pkl ile)
from fastai.vision.all import *
learn = load_learner('notebooks/export.pkl')  # varsa
img = PILImage.create('any_test.jpg')
pred, _, probs = learn.predict(img)
print(pred, float(probs.max()))

Lokal/Colab eğitim + Releases’ten .pth yükleme

Ağırlığı Releases’tan indirip models/ klasörüne koyduktan sonra:

from fastai.vision.all import *

# dls'i (224px) Intel train klasörüyle oluşturduğunu varsayıyoruz
learn = vision_learner(
    dls, resnet18, metrics=[accuracy],
    path=Path('.'), model_dir=Path('models')
).to_fp32()

# models/resnet18-intel-best.pth dosyasını yükle
learn.load('resnet18-intel-best', with_opt=False)

# test örneği
pred, _, probs = learn.predict(PILImage.create('any_test.jpg'))
print(pred, float(probs.max()))


learn.load("ad") dosyayı learn.path/learn.model_dir/ad.pth içinde arar; bu yüzden dosyayı models/ içine koyuyoruz.

📦 Release nasıl hazırlandı?

Kaggle’da: learn.save("resnet18-intel-best") ile .pth üretildi.

GitHub → Releases → Draft a new release → tag & başlık verildi → resnet18-intel-best.pth “Assets” alanına yüklendi.

Büyük dosyalar için Releases uygundur (tek dosya limiti 2 GB).

🔮 Geliştirme Fikirleri

Daha güçlü mimariler (ResNet34/50, EfficientNet, ConvNeXt)

Mixup/CutMix, RandomErasing, TTA

Sınıf dengeleme: class-weight, focal loss, oversampling

W&B/MLflow ile deney takibi
