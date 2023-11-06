import cv2
import numpy as np
from pathlib import Path

# load images
img_path = Path(r"related_work/paird_vs_unpaired_I2I.png")
img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
img = img_raw[..., :-1]
img_transparency = img_raw[..., -1]
is_thresh_only = True

# convert to graky
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold input image as mask
mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)[1]

# negate mask
mask = 255 - mask

if not is_thresh_only:
    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(
        mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT
    )

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)


# put mask into alpha channel
result = img_raw.copy()
result[:, :, 3] = mask & img_transparency

# save resulting masked image
cv2.imwrite(str(img_path.with_suffix(".png")), result)
