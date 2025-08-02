#!/usr/bin/env python3
# train.py


# python train.py --data license-plate-data/data.yaml --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640 --device 0 --project runs/train --name license_plate


import os
import sys
import io
import json
import argparse
import logging
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 License Plate Training & Evaluation")
    p.add_argument('--data',      type=str, default='license-plate-data/data.yaml', help='YAML with train/val/test paths')
    p.add_argument('--model',     type=str, default='yolov8n.pt',                  help='Önceden eğitilmiş weight yolu')
    p.add_argument('--epochs',    type=int, default=80,                            help='Epoch sayısı')
    p.add_argument('--batch',     type=int, default=24,                            help='Batch boyutu')
    p.add_argument('--imgsz',     type=int, default=640,                           help='Input image size')
    p.add_argument('--device',    type=str, default='0',                           help='GPU id (örn: "0") veya "cpu"')
    p.add_argument('--project',   type=str, default='runs/train',                  help='Çıktı klasörü')
    p.add_argument('--name',      type=str, default='license_plate',               help='Deney adı')
    return p.parse_args()


def setup_logging(log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Konsol ve dosya için UTF-8 handle edebilen stream’ler yaratıyoruz
    utf8_console = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    handlers = [
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(utf8_console)
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=handlers
    )

def plot_and_save(df, out_dir):
    # Loss curves
    plt.figure()
    plt.plot(df['epoch'], df['box_loss'],  label='Box Loss')
    plt.plot(df['epoch'], df['cls_loss'],  label='Cls Loss')
    plt.plot(df['epoch'], df['dfl_loss'],  label='DFL Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

    # mAP curves
    plt.figure()
    plt.plot(df['epoch'], df['mAP_50'],     label='mAP@50')
    plt.plot(df['epoch'], df['mAP_50_95'],  label='mAP@50-95')
    plt.xlabel('Epoch'); plt.ylabel('mAP'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'map_curve.png'))
    plt.close()

    # Precision & Recall
    plt.figure()
    plt.plot(df['epoch'], df['precision'], label='Precision')
    plt.plot(df['epoch'], df['recall'],    label='Recall')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pr_curve.png'))
    plt.close()

def main():
    args = parse_args()
    out_dir = os.path.join(args.project, args.name)
    log_file = os.path.join(out_dir, 'train.log')
    setup_logging(log_file)

    logging.info("🚀 Training başlıyor")
    logging.info(f"Model: {args.model}, Data: {args.data}, Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}, Device: {args.device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        verbose=True
    )
    logging.info("✅ Eğitim tamamlandı")

    # Grafikleri çiz ve kaydet
    results_csv = os.path.join(out_dir, 'results.csv')
    if os.path.isfile(results_csv):
        df = pd.read_csv(results_csv)
        plot_and_save(df, out_dir)
        logging.info(f"📈 Eğitim grafiklerini '{out_dir}' altında kaydettim")
    else:
        logging.warning(f"⚠ results.csv bulunamadı: {results_csv}")

    # Validation üzerinde değerlendirme
     # Validation üzerinde değerlendirme
     logging.info("🔍 Validation değerlendirmesi yapılıyor")
-    val_metrics = model.val(data=args.data, split='val', plots=True)
+    val_metrics = model.val(data=args.data, split='val', plots=True)
     val_path = os.path.join(out_dir, 'val_metrics.json')
     with open(val_path, 'w', encoding='utf-8') as f:
-        json.dump(val_metrics, f, indent=2)
+        json.dump(val_metrics.__dict__, f, indent=2)
     logging.info(f"✅ Validation metrikleri kaydedildi: {val_path}")

     # Test üzerinde değerlendirme
     logging.info("🔍 Test değerlendirmesi yapılıyor")
-    test_metrics = model.val(data=args.data, split='test', plots=True)
+    test_metrics = model.val(data=args.data, split='test', plots=True)
     test_path = os.path.join(out_dir, 'test_metrics.json')
     with open(test_path, 'w', encoding='utf-8') as f:
-        json.dump(test_metrics, f, indent=2)
+        json.dump(test_metrics.__dict__, f, indent=2)
     logging.info(f"✅ Test metrikleri kaydedildi: {test_path}")


    # Test üzerinde değerlendirme
    logging.info("🔍 Test değerlendirmesi yapılıyor")
    test_metrics = model.val(data=args.data, split='test', plots=True)
    test_path = os.path.join(out_dir, 'test_metrics.json')
    with open(test_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logging.info(f"✅ Test metrikleri kaydedildi: {test_path}")

    logging.info("🎉 Tüm işlem tamamlandı, sonuç klasörünü kontrol edebilirsin.")

if __name__ == '__main__':
    main()
