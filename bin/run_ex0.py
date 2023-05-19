import sys
sys.path.append(".")
import os
import cv2
import ex0

if __name__ == "__main__":
    ## TODO 3.1
    ## Load Image
    ## Show it on screen
    ## Note: implement show_images in ex0/functions.py
    file = 'img.png'  ## path to the image
    input_path = os.path.join("data", "ex0", "img.png")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    ex0.show_images([img], ["Display Image"])

    ## TODO 3.2
    ## Resize Image by a factor of 0.5
    ## Show it on screen
    ## Save as small.jpg
    ## Note: implement save_images, scale_down in ex0/functions.py
    small_img = ex0.scale_down(img)
    ex0.show_images([small_img], ["Small Image"])
    ex0.save_images([small_img], ["small.jpg"])

    ## TODO 3.3
    ## Create and save 3 single-channel images from small image
    ## one image each channel (r, g, b)
    ## Display the channel-images on screen
    ## Note: implement separate_channels in ex0/functions.py
    blue, green, red = ex0.separate_channels(img)
    ex0.show_images([img, blue, green, red], ["Original", "Blue", "Green", "Red"])
    ex0.save_images([blue, green, red], ["blue.png", "green.png", "red.png"])
