# 1. Imports: Bringing in the necessary tools
import torch
# 'torch' is the PyTorch library, the deep learning framework that YOLOv8 is built on.
# It handles tensors (multi-dimensional arrays) and GPU acceleration.

from ultralytics import YOLO
# 'ultralytics' is the official library for YOLOv8. 
# We import the 'YOLO' class, which is the main interface for loading, training, and using models.

import cv2
# 'cv2' is OpenCV, a powerful computer vision library. 
# We use it here to process images (specifically converting color formats) for display.

import matplotlib.pyplot as plt
# 'matplotlib.pyplot' is a plotting library. 
# We use it to display the final image with bounding boxes in a window.

def explain_my_model():
    # This defines a function to keep our code organized.
    
    # 2. Setting the Model Path
    # This variable stores the location of your trained model weights file.
    # "runs/detect/final_sota_150epochs/weights/best.pt" is the standard path where YOLO saves the best model.
    model_path = "runs/detect/final_sota_150epochs/weights/best.pt"
    
    try:
        # 3. Loading the Model
        # 'YOLO(model_path)' initializes the model architecture and loads your trained weights into it.
        model = YOLO(model_path)
        print(f"✅ Successfully loaded model from: {model_path}")
    except Exception as e:
        # This error handling block catches issues (like a wrong file path) and prints the error message.
        print(f"❌ Error loading model: {e}")
        return

    # 4. Printing the Architecture Summary
    # These print statements just format the output to look nice in the terminal.
    print("\n" + "="*50)
    print("     YOLOv8m MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    
    # 5. The Core Explanation Method: model.info()
    # This is a built-in method of the YOLO class.
    # 'detailed=True' tells it to print every single layer (Conv, C2f, SPPF, Detect), 
    # the number of parameters in each, and the gradients. 
    # This is the "code" representation of the model's structure.
    model.info(detailed=True) 

    # 6. Calculating Total Parameters
    # This line manually counts the total number of learnable parameters in the model.
    # 'model.parameters()' gives access to all weights and biases.
    # 'p.numel()' counts the number of elements in each parameter tensor.
    # We sum them up and divide by 1,000,000 (1e6) to display the count in Millions.
    print("\n" + "="*50)
    print(f" Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million")
    print("="*50 + "\n")

    # 7. Setting the Test Image Path
    # This points to an image you want to test. ///////XVR_ch9_main_20221004170008_20221004180008_004933-005454_mp4-26_jpg.rf.1600d7b4f7b9c734135e76f24d702442.jpg
    # Ideally, this should be a real image from your dataset's test folder. ///// Yangon_II_39_10_jpg.rf.740333086ea987584ac77f7a72b0a256.jpg 
    test_image_path = "data/Ultimate Helmet Detection/test/images/Yangon_II_39_10_jpg.rf.740333086ea987584ac77f7a72b0a256.jpg" 
    
    # 8. Running Inference (Prediction)
    # 'model(test_image_path)' passes the image through the network.
    # The network performs the forward pass and returns a list of 'Results' objects.
    results = model(test_image_path)

    # 9. Visualizing the Result
    # We loop through the results (usually just one for a single image).
    for result in results:
        # 'result.plot()' draws the bounding boxes and labels onto the image array.
        # It returns the image as a NumPy array in BGR (Blue-Green-Red) color format (standard for OpenCV).
        im_array = result.plot()
        
        # 'cv2.cvtColor(...)' converts the color from BGR to RGB (Red-Green-Blue).
        # This is necessary because Matplotlib expects RGB, but OpenCV uses BGR. 
        # If we skipped this, the colors would look wrong (e.g., blue faces).
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        # 10. Displaying with Matplotlib
        # 'plt.figure(figsize=(10, 10))' creates a figure window of size 10x10 inches.
        plt.figure(figsize=(10, 10))
        
        # 'plt.imshow(im_rgb)' displays the image data on the figure.
        plt.imshow(im_rgb)
        
        # 'plt.axis('off')' hides the X and Y axis numbers, making it look like a clean photo.
        plt.axis('off')
        
        # 'plt.title(...)' adds a title to the window.
        plt.title("YOLOv8m Prediction Result")
        
        # 'plt.show()' actually renders the window on your screen.
        plt.show()

if __name__ == "__main__":
    # This standard Python block ensures the script runs only when executed directly.
    explain_my_model()