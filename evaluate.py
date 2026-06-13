from ultralytics import YOLO

print("=" * 50)
print("YOLOv8 OBJECT DETECTION EVALUATION")
print("Dataset : COCO128")
print("=" * 50)

model = YOLO("yolov8n.pt")

metrics = model.val(
    data="coco128.yaml",
    workers=0
)

precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map5095 = metrics.box.map

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)

print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"mAP@50    : {map50*100:.2f}%")
print(f"mAP@50-95 : {map5095*100:.2f}%")

with open(
    "evaluation_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
f"""Dataset : COCO128

Precision : {precision*100:.2f} %

Recall : {recall*100:.2f} %

mAP@50 : {map50*100:.2f} %

mAP@50-95 : {map5095*100:.2f} %
"""
    )

print("\nResults saved to evaluation_results.txt")