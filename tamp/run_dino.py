from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

device = 'cuda:0'

model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
IMAGE_PATH = "RGB.jpg"
TEXT_PROMPT = "lamp, cd, strongbox, alarm"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

print(boxes)
print(phrases) # [x,y,w,h]

object = {}
for box, phrase in zip(boxes, phrases):
    object[phrase] = box

print(boxes.tolist())

for i in range(len(object)):
    for j in range(len(object)):
        if i != j:
            if boxes.tolist()[i][0] - boxes.tolist()[j][0] > 0.1:
                print(f"{list(object.keys())[i]} is right to {list(object.keys())[j]}")
            if boxes.tolist()[i][0] - boxes.tolist()[j][0] < -0.1:
                print(f"{list(object.keys())[i]} is left to {list(object.keys())[j]}")
            if boxes.tolist()[i][1] - boxes.tolist()[j][1] > 0.1:
                print(f"{list(object.keys())[i]} is down to {list(object.keys())[j]}")
            if boxes.tolist()[i][1] - boxes.tolist()[j][1] < -0.1:
                print(f"{list(object.keys())[i]} is up to {list(object.keys())[j]}")

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("RGB_annotated.jpg", annotated_frame)