import cv2
import numpy as np
import glob
import os

def is_document(image_path, debug=True):
    """Detects whether an image is a document using adaptive confidence scoring."""
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize for faster processing
    h, w = gray.shape[:2]
    scale = 500.0 / max(h, w)  # Resize longest dimension to 500px
    gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Adaptive scoring (0.0 - 1.0 for each feature)
    background_score = analyze_background(image)  # Gradient-based solid background
    contrast_score = analyze_contrast(image)  # High/low contrast detection
    shape_score = detect_document_shape(gray)  # Soft quadrilateral shape detection

    # Final classification score
    final_score = (background_score + shape_score - contrast_score/4) / 3

    # Debug display if enabled
    if debug:
        print(f"Image name: {image_path}, Scores - Background: {background_score:.2f}, Contrast: {contrast_score:.2f}, Shape: {shape_score:.2f}")
        # debug_display(image)

    return final_score >= 0.1  # Threshold-based classification

def analyze_background(image):
    """Returns a score (0-1) based on how uniform the background is."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    pixels = blurred.reshape(-1, 3)
    
    std_dev = np.std(pixels, axis=0)  # Measure color variation
    uniformity = 1 - np.mean(std_dev) / 50  # Normalize
    return max(0, min(1, uniformity))  # Clamp values between 0 and 1

def analyze_contrast(gray):
    """Returns a score (0-1) based on how strong the contrast is."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    peaks = np.argwhere(hist > np.max(hist) * 0.05).flatten()
    if len(peaks) < 2:
        return 0  # No contrast
    contrast_strength = abs(peaks[0] - peaks[-1]) / 255
    return max(0, min(1, contrast_strength))  # Normalize

def detect_document_shape(gray):
    """Returns a score (0-1) based on how closely the shape resembles a quadrilateral."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_score = 0
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        area = cv2.contourArea(cnt)
        
        if len(approx) >= 4 and area > 5000:
            similarity = min(1, len(approx) / 4)  # Score based on closeness to 4 sides
            max_score = max(max_score, similarity)

    return max_score  # Best quadrilateral match

def debug_display(image):
    """Displays the processed image for debugging."""
    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate_document_detection(documents_path, images_path):
    """Evaluates precision by testing document detection on a dataset of images."""
    
    document_images = glob.glob(os.path.join(documents_path, "*"))
    normal_images = glob.glob(os.path.join(images_path, "*"))

    total_docs = len(document_images)
    total_non_docs = len(normal_images)
    
    correct = 0
    incorrect = []

    for img_path in document_images:
        result = is_document(img_path)
        if result:
            correct += 1
        else:
            incorrect.append((img_path, "False Negative"))

    for img_path in normal_images:
        result = is_document(img_path)
        if not result:
            correct += 1
        else:
            incorrect.append((img_path, "False Positive"))

    total_images = total_docs + total_non_docs
    precision = correct / total_images if total_images > 0 else 0

    if incorrect:
        print("\nMisclassified images:")
        for img, error in incorrect:
            print(f"{img} - {error}")

    print(f"\nTotal Images: {total_images}")
    print(f"Correctly Classified: {correct}")
    print(f"Precision: {precision:.2%}")

# Paths to document and image folders
documents_folder = "documents"
images_folder = "images"

# Run evaluation
evaluate_document_detection(documents_folder, images_folder)
