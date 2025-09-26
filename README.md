'''markdown

Intel Image Classification (6 sınıf) · fastai + ResNet18

Amaç: Doğal/yapay ortam görsellerini (buildings, forest, glacier, mountain, sea, street) bir CNN ile sınıflandırmak.
Teknolojiler: fastai v2 · PyTorch · Kaggle GPU

Bu repo; veri önişleme + augmentation, CNN (ResNet18, transfer learning), metrik grafikleri, Confusion Matrix & Classification Report, Grad-CAM, küçük HPO ve model kaydetme adımlarını içerir.

🔗 Bağlantılar

Kaggle Notebook: https://www.kaggle.com/code/fallerdem/globalai/edit

Model Ağırlıkları (GitHub Release): https://github.com/FallErdem/intel-image-classification-fastai/releases/tag/v0.1.0

Veri Seti (Kaggle): puneet6060/intel-image-classification

🗂️ Proje Yapısı

'''
notebooks/
Intel_Image_Classification.ipynb
export.pkl # (opsiyonel) tek dosya ile inference
reports/
confusion_matrix.PNG
gradcam_example.png
'''

Not: GitHub’ın 100 MB limiti nedeniyle büyük .pth dosyası repoda değil, Releases altında paylaşıldı (link yukarıda). Sadece tahmin için export.pkl en pratik yoldur.

🧭 Yöntem (Özet)

Veri önişleme: fastai DataBlock (80/20 train/valid), giriş boyutu 224×224

Augmentations: flip, rotate(±8°), zoom(1.1), lighting(0.2)

Model: ResNet18 (ImageNet ön-eğitimli) → Transfer Learning

Eğitim: LR Finder → 12 epoch (frozen) + 5 epoch (unfreeze, küçük LR)

Değerlendirme: Accuracy/Loss grafikleri, Confusion Matrix, Classification Report

Açıklanabilirlik: Grad-CAM (son konv katmanından ısı haritası)

HPO (mini-grid): farklı lr_slice & weight_decay kombinasyonları

Kayıt: export.pkl (fastai), .pth (Release’te)

📊 Sonuçlar

Validation Accuracy (en iyi): ~0.948

Test Accuracy: ~0.87 (3000 görüntü)

En çok karışan sınıflar: glacier ↔ mountain/sea (benzer doku & ufuk çizgisi)

Görseller:

Confusion Matrix → reports/confusion_matrix.PNG

Grad-CAM örneği → reports/gradcam_example.png

🚀 Çalıştırma
Kaggle (önerilen)

New Notebook → Add Data: puneet6060/intel-image-classification

Settings → Accelerator: GPU

Bu repodaki notebook’u yükleyin (veya Kaggle’a import edin).

Hücreleri sırayla çalıştırın; çıktılar /kaggle/working/ altında oluşur.

Lokal Inference (yalnızca tahmin)

Seçenek A — export.pkl ile (tek dosya, en pratik)
'''
from fastai.vision.all import *

learn = load_learner('notebooks/export.pkl')

img = PILImage.create('any_test.jpg')
pred, _, probs = learn.predict(img)
print(pred, float(probs.max()))
'''

Seçenek B — Release’ten indirilen .pth ile (eğitim/devam senaryosu)
'''
from fastai.vision.all import *
from pathlib import Path

dls'i (224px) eğitimdeki augment/normalize ile kurduğunuzu varsayar.

learn = vision_learner(
dls, resnet18, metrics=[accuracy],
path=Path('.'), model_dir=Path('models')
).to_fp32()

Release'ten indirdiğiniz dosyayı models/ klasörüne koyun:
models/resnet18-intel-best.pth

learn.load('resnet18-intel-best', with_opt=False)

img = PILImage.create('any_test.jpg')
pred, _, probs = learn.predict(img)
print(pred, float(probs.max()))
'''
Not: learn.load("resnet18-intel-best"), dosyayı models/resnet18-intel-best.pth yolunda arar.

✅ Bootcamp İsterleri Eşlemesi

Kaggle notebook + GitHub repo + README ✔️

Veri önişleme + Data Augmentation ✔️

CNN tabanlı model (ResNet18, TL) ✔️

Accuracy/Loss grafikleri + Confusion Matrix + Classification Report ✔️

Grad-CAM görselleştirme ✔️

Hiperparametre denemeleri (mini-grid) ✔️

Model kaydetme (export.pkl / .pth) ✔️

(Bonus) TensorBoard (trace_model=False) ✔️

🔮 Geliştirme Fikirleri

Daha güçlü mimariler: ResNet34/50, EfficientNet, ConvNeXt

Mixup/CutMix, RandomErasing, TTA

Sınıf dengeleme: class-weight, focal loss, oversampling

W&B / MLflow ile deney takibi ve izleme

🙏 Teşekkür

Dataset: Intel Image Classification — Kaggle (puneet6060)

fastai & PyTorch toplulukları
