import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import gradio as gr
from skimage.feature import hog

# Load models at the start
try:
    svm_model = joblib.load("hog_svm_traffic_signs_fast.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please make sure the model files are in the current working directory:")
    print("- hog_svm_traffic_signs_fast.pkl")
    print("- label_encoder.pkl")
    print("- scaler.pkl")
    svm_model, label_encoder, scaler = None, None, None

# Constants
IMG_SIZE = (64, 64)
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (3, 3)

def segment_red_signs(image):
    """Segment red traffic signs using HSV color space."""
    # Resize large images to a reasonable size while maintaining aspect ratio
    height, width = image.shape[:2]
    max_dimension = 800
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color (requires two ranges due to HSV cylindrical nature)
    lower_red1 = np.array([0, 40, 40])  # Even more lenient
    upper_red1 = np.array([15, 255, 255])  # Wider range
    lower_red2 = np.array([160, 40, 40])  # Even more lenient
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    
    return red_mask, image

def segment_blue_signs(image):
    """Segment blue traffic signs (information signs) using HSV color space."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue range
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    return blue_mask

def segment_yellow_signs(image):
    """Segment yellow traffic signs (warning) using HSV color space."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for yellow color
    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    
    # Create mask for yellow range
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    return yellow_mask

def detect_circles(mask, original_image):
    """Detect circular traffic signs using Hough Circle Transform."""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Use Hough Circle Transform with less restrictive parameters
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=30,  
        param1=30,   
        param2=15,   # Even less restrictive
        minRadius=5, 
        maxRadius=150
    )
    
    detected_signs = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            
            # Define bounding box coordinates (square around circle)
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(original_image.shape[1], x + r)
            y2 = min(original_image.shape[0], y + r)
            
            # Extract the region
            sign_region = original_image[y1:y2, x1:x2]
            
            if sign_region.size > 0 and sign_region.shape[0] > 0 and sign_region.shape[1] > 0:
                detected_signs.append({
                    'region': sign_region,
                    'bbox': (x1, y1, x2, y2)
                })
    
    return detected_signs

def detect_triangles(mask, original_image):
    """Detect triangular traffic signs using contour detection."""
    detected_signs = []
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < 100:
            continue
            
        # Approximate contour to polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If polygon has 3 vertices, it's a triangle
        if len(approx) == 3:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract the region
            sign_region = original_image[y:y+h, x:x+w]
            
            if sign_region.size > 0:  # Check if region is valid
                detected_signs.append({
                    'region': sign_region,
                    'bbox': (x, y, x+w, y+h),
                    'shape': 'triangle',
                    'vertices': approx
                })
    
    return detected_signs

def detect_rectangles(mask, original_image):
    """Detect rectangular traffic signs using contour detection."""
    detected_signs = []
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < 100:
            continue
            
        # Approximate contour to polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If polygon has 4 vertices, it's a rectangle
        if len(approx) == 4:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio to filter out very elongated rectangles
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.3:  # Approximately square signs
                # Extract the region
                sign_region = original_image[y:y+h, x:x+w]
                
                if sign_region.size > 0:  # Check if region is valid
                    detected_signs.append({
                        'region': sign_region,
                        'bbox': (x, y, x+w, y+h),
                        'shape': 'rectangle',
                        'vertices': approx
                    })
    
    return detected_signs

def extract_orb_features(image):
    """Extract ORB features from an image."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to standard size
    gray = cv2.resize(gray, (64, 64))
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def match_features(query_descriptors, template_descriptors):
    """Match ORB features between query and template."""
    if query_descriptors is None or template_descriptors is None:
        return 0
    
    # Create BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(query_descriptors, template_descriptors)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Only consider good matches (lower distance)
    good_matches = [m for m in matches if m.distance < 50]
    
    # Return number of good matches
    return len(good_matches)

def detect_signs(image, reference_templates=None):
    """Complete pipeline for traffic sign detection and recognition."""
    
    # Clone the image to avoid modifying the original
    result_image = image.copy()
    
    # Color segmentation
    red_mask, resized_image = segment_red_signs(image)
    blue_mask = segment_blue_signs(image)
    yellow_mask = segment_yellow_signs(image)
    
    # Shape detection
    detected_signs = []
    
    # Red signs - Circles (prohibitory) and triangles (warning)
    red_circles = detect_circles(red_mask, resized_image)
    for sign in red_circles:
        sign['color'] = 'red'
        sign['type'] = 'prohibitory'
        detected_signs.append(sign)
    
    red_triangles = detect_triangles(red_mask, resized_image)
    for sign in red_triangles:
        sign['color'] = 'red'
        sign['type'] = 'warning'
        detected_signs.append(sign)
    
    # Blue signs - Rectangles (information)
    blue_rectangles = detect_rectangles(blue_mask, resized_image)
    for sign in blue_rectangles:
        sign['color'] = 'blue'
        sign['type'] = 'information'
        detected_signs.append(sign)
    
    # Yellow signs - Triangles (warning)
    yellow_triangles = detect_triangles(yellow_mask, resized_image)
    for sign in yellow_triangles:
        sign['color'] = 'yellow'
        sign['type'] = 'warning'
        detected_signs.append(sign)
    
    # Feature matching and classification
    if reference_templates:
        for sign in detected_signs:
            region = sign['region']
            _, query_descriptors = extract_orb_features(region)
            
            best_match = {'class': 'unknown', 'score': 0}
            
            for sign_class, templates in reference_templates.items():
                for template in templates:
                    template_descriptors = template['descriptors']
                    match_score = match_features(query_descriptors, template_descriptors)
                    
                    if match_score > best_match['score']:
                        best_match['class'] = sign_class
                        best_match['score'] = match_score
            
            sign['class'] = best_match['class']
            sign['match_score'] = best_match['score']
    
    # Draw bounding boxes on the result image
    for sign in detected_signs:
        x1, y1, x2, y2 = sign['bbox']
        
        # Set color based on traffic sign type
        if sign['color'] == 'red':
            color = (0, 0, 255)  # BGR: Red
        elif sign['color'] == 'blue':
            color = (255, 0, 0)  # BGR: Blue
        elif sign['color'] == 'yellow':
            color = (0, 255, 255)  # BGR: Yellow
        else:
            color = (0, 255, 0)  # BGR: Green
        
        # Draw rectangle
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label if class is available
        if 'class' in sign and sign['class'] != 'unknown':
            label = sign['class']
            cv2.putText(result_image, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image, detected_signs

def test_on_sample(image_path, reference_templates=None):
    """Test the detection pipeline on a sample image."""
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Start time
    start_time = time.time()
    
    # Run detection
    result_image, detected_signs = detect_signs(image, reference_templates)
    
    # End time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Convert BGR to RGB for displaying with matplotlib
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display results
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_rgb)
    plt.title(f'Detected Signs (Processing Time: {processing_time:.2f}s)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detection details
    print(f"Number of detected signs: {len(detected_signs)}")
    for i, sign in enumerate(detected_signs):
        shape_type = sign.get('shape', 'unknown')
        color_type = sign.get('color', 'unknown')
        sign_type = sign.get('type', 'unknown')
        sign_class = sign.get('class', 'unknown')
        
        print(f"Sign {i+1}: {color_type} {shape_type} - Type: {sign_type}, Class: {sign_class}")
    
    return detected_signs

def evaluate_on_dataset(test_dir, reference_templates=None):
    """Evaluate detection performance on a test dataset."""
    total_images = 0
    total_detected = 0
    processing_times = []
    
    # Get list of test images
    test_images = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
    
    print(f"Found {len(test_images)} test images. Starting evaluation...")
    
    # Process each image
    for image_path in tqdm(test_images):
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        total_images += 1
        
        # Run detection with timing
        start_time = time.time()
        _, detected_signs = detect_signs(image, reference_templates)
        end_time = time.time()
        
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        total_detected += len(detected_signs)
    
    # Calculate metrics
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_signs_per_image = total_detected / total_images if total_images > 0 else 0
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total images processed: {total_images}")
    print(f"Total signs detected: {total_detected}")
    print(f"Average signs per image: {avg_signs_per_image:.2f}")
    print(f"Average processing time: {avg_processing_time:.3f} seconds")
    print(f"Frames per second: {1/avg_processing_time:.2f}")
    
    return {
        'total_images': total_images,
        'total_detected': total_detected,
        'avg_signs_per_image': avg_signs_per_image,
        'avg_processing_time': avg_processing_time,
        'fps': 1/avg_processing_time if avg_processing_time > 0 else 0
    }

def classify_with_hog_svm(detected_signs):
    """Classify detected signs using HOG+SVM model directly."""
    try:
        for sign in detected_signs:
            region = sign['region']
            
            # Preprocess for HOG+SVM
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Resize image
            gray = cv2.resize(gray, IMG_SIZE)
            
            # Extract HOG features
            features = hog(gray, 
                          pixels_per_cell=HOG_PIXELS_PER_CELL, 
                          cells_per_block=HOG_CELLS_PER_BLOCK, 
                          feature_vector=True)
            
            # Scale features
            scaled_features = scaler.transform([features])
            
            # Predict class
            predicted_label = svm_model.predict(scaled_features)[0]
            
            # Convert numeric label to class name
            class_name = label_encoder.inverse_transform([predicted_label])[0]
            
            sign['hog_svm_class'] = class_name
            
            # Get human-readable name for the class
            sign['name'] = get_sign_name(predicted_label)
        
        return detected_signs
    except Exception as e:
        print(f"Error classifying with HOG+SVM: {e}")
        return detected_signs

def get_sign_name(class_id):
    """Returns the full name of a traffic sign based on its class ID."""
    sign_names = {
        0: "Speed limit (20km/h)",
        1: "Speed limit (30km/h)",
        2: "Speed limit (50km/h)",
        3: "Speed limit (60km/h)",
        4: "Speed limit (70km/h)",
        5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)",
        7: "Speed limit (100km/h)",
        8: "Speed limit (120km/h)",
        9: "No passing",
        10: "No passing for vehicles over 3.5 tons",
        11: "Right-of-way at intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "Vehicles > 3.5 tons prohibited",
        17: "No entry",
        18: "General caution",
        19: "Dangerous curve left",
        20: "Dangerous curve right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows on the right",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children crossing",
        29: "Bicycles crossing",
        30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End speed + passing limits",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End no passing for vehicles > 3.5 tons"
    }
    return sign_names.get(int(class_id), f"Unknown Sign (Class {class_id})")

def create_gradio_interface():
    """Create a Gradio interface for traffic sign detection."""
    def process_image(image):
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run detection
        result_image, detected_signs = detect_signs(image)
        
        # Classify with HOG+SVM
        try:
            detected_signs = classify_with_hog_svm(detected_signs)
        except Exception as e:
            print(f"Error in HOG+SVM classification: {e}")
        
        # Format output
        detection_info = []
        for i, sign in enumerate(detected_signs):
            shape_type = sign.get('shape', 'unknown')
            color_type = sign.get('color', 'unknown')
            sign_type = sign.get('type', 'unknown')
            
            # Use the human-readable name if available
            if 'name' in sign:
                class_info = f"Class: {sign.get('name', 'unknown')}"
            elif 'hog_svm_class' in sign:
                class_info = f"Class: {sign.get('hog_svm_class', 'unknown')}"
            else:
                class_info = f"Class: {sign.get('class', 'unknown')}"
                
            detection_info.append(f"Sign {i+1}: {color_type} {shape_type} - Type: {sign_type}, {class_info}")
        
        # Convert back to RGB for display
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        return result_image, "\n".join(detection_info)
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="pil", label="Detected Signs"),
            gr.Textbox(label="Detection Results")
        ],
        title="Traffic Sign Detection & Classification",
        description="Upload an image to detect and classify traffic signs using computer vision and HOG+SVM."
    )
    
    return interface

def load_reference_templates(template_dir):
    """Load reference templates and extract features."""
    templates = {}
    
    # Assume directory structure: template_dir/sign_class/images.png
    for sign_class in os.listdir(template_dir):
        class_dir = os.path.join(template_dir, sign_class)
        
        # Skip non-directories
        if not os.path.isdir(class_dir):
            continue
            
        templates[sign_class] = []
        
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            
            # Skip non-image files
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Load image and extract features
            image = cv2.imread(image_path)
            keypoints, descriptors = extract_orb_features(image)
            
            if descriptors is not None:
                templates[sign_class].append({
                    'image': image,
                    'descriptors': descriptors
                })
    
    return templates

def create_confusion_matrix(test_dir, reference_templates):
    """Create a confusion matrix for sign classification accuracy."""
    # Get ground truth labels from directory structure
    # Assume test_dir structure: class_name/image.jpg
    ground_truth_dict = {}
    
    for root, _, files in os.walk(test_dir):
        class_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                ground_truth_dict[image_path] = class_name
    
    # Process images and store predictions
    predictions = []
    ground_truths = []
    
    for image_path, true_class in tqdm(ground_truth_dict.items()):
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Run detection
        _, detected_signs = detect_signs(image, reference_templates)
        
        if detected_signs:
            # Get the detected class with highest match score
            best_sign = max(detected_signs, key=lambda x: x.get('match_score', 0) if 'match_score' in x else 0)
            predicted_class = best_sign.get('class', 'unknown')
            
            predictions.append(predicted_class)
            ground_truths.append(true_class)
    
    # Calculate confusion matrix
    unique_classes = sorted(list(set(ground_truths + predictions)))
    confusion_mat = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    
    for gt, pred in zip(ground_truths, predictions):
        gt_idx = class_to_idx[gt]
        pred_idx = class_to_idx[pred]
        confusion_mat[gt_idx, pred_idx] += 1
    
    # Display confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(unique_classes))
    plt.xticks(tick_marks, unique_classes, rotation=90)
    plt.yticks(tick_marks, unique_classes)
    
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()
    
    # Calculate accuracy
    accuracy = sum(np.diag(confusion_mat)) / confusion_mat.sum()
    print(f"Overall Classification Accuracy: {accuracy:.4f}")
    
    return confusion_mat, accuracy

def process_image(image):
    """Process an image for traffic sign detection and classification."""
    # Check if we have the required models
    if svm_model is None or label_encoder is None or scaler is None:
        result_image = image.copy() if isinstance(image, np.ndarray) else np.array(image)
        if len(result_image.shape) == 3 and result_image.shape[2] == 3:
            # Convert to RGB for display if needed
            if isinstance(image, np.ndarray):
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        return result_image, "ERROR: Required model files not found. Please check console output."
    
    # Convert PIL image to OpenCV format if needed
    if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
        # Already in correct format (BGR)
        img = image.copy()
    else:
        # Convert from RGB (PIL) to BGR (OpenCV)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Segment red signs and get potentially resized image
    red_mask, img = segment_red_signs(img)
    
    # Create a copy for drawing results
    result_image = img.copy()
    
    # Get image dimensions for adaptive parameters
    height, width = img.shape[:2]
    avg_dimension = (height + width) / 2
    
    # Try to detect traffic signs using different methods
    detected_signs = []
    
    # 1. Detect circles for round signs
    blurred = cv2.GaussianBlur(red_mask, (9, 9), 2)  # More blur for large images
    
    # For the given images with larger signs, use adapted parameters
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=50,
        param1=50,
        param2=20,  # Less restrictive
        minRadius=20, 
        maxRadius=min(300, int(avg_dimension/3))  # Cap at 1/3 of avg dimension
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            
            # Define bounding box coordinates (square around circle)
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(img.shape[1], x + r)
            y2 = min(img.shape[0], y + r)
            
            # Extract the region
            sign_region = img[y1:y2, x1:x2]
            
            if sign_region.size > 0 and sign_region.shape[0] > 10 and sign_region.shape[1] > 10:
                detected_signs.append({
                    'region': sign_region,
                    'bbox': (x1, y1, x2, y2)
                })
    
    # 2. Detect triangles and other shapes using contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter small contours - adaptive threshold based on image size
        min_area = max(200, int(avg_dimension * 0.5))  # Larger threshold for bigger images
        if cv2.contourArea(contour) < min_area:
            continue
            
        # Approximate contour to polygon with more tolerant epsilon
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Less strict approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Accept triangles, rectangles, and similar shapes (3-5 vertices)
        if 3 <= len(approx) <= 5:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very thin rectangles
            if w > 20 and h > 20 and 0.3 < w/h < 3.0:
                # Extract the region
                sign_region = img[y:y+h, x:x+w]
                
                if sign_region.size > 0:  # Check if region is valid
                    detected_signs.append({
                        'region': sign_region,
                        'bbox': (x, y, x+w, y+h)
                    })
    
    # Classification and drawing results
    detection_info = []
    
    for i, sign in enumerate(detected_signs):
        region = sign['region']
        x1, y1, x2, y2 = sign['bbox']
        
        # Classify sign with HOG+SVM
        if svm_model is not None and scaler is not None and label_encoder is not None:
            try:
                # Preprocess for HOG+SVM
                if len(region.shape) == 3:
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                else:
                    gray = region
                
                # Enhance contrast
                gray = cv2.equalizeHist(gray)
                
                # Resize image
                gray = cv2.resize(gray, IMG_SIZE)
                
                # Extract HOG features
                features = hog(gray, 
                            pixels_per_cell=HOG_PIXELS_PER_CELL, 
                            cells_per_block=HOG_CELLS_PER_BLOCK, 
                            feature_vector=True)
                
                # Scale features
                scaled_features = scaler.transform([features])
                
                # Predict class
                predicted_label = svm_model.predict(scaled_features)[0]
                
                # Get human-readable name for the class
                sign_class = get_sign_name(predicted_label)
            except Exception as e:
                print(f"Error classifying sign: {e}")
                sign_class = "Classification error"
        else:
            sign_class = "Model not loaded"
        
        # Draw rectangle with thickness based on image size
        thickness = max(2, int(avg_dimension / 200))
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        
        # Add label with font size based on image size
        font_scale = max(0.5, avg_dimension / 1000)
        y_offset = max(25, int(avg_dimension / 40))  # More space for label on larger images
        cv2.putText(result_image, sign_class, (x1, max(y1-10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        
        # Add to detection info
        detection_info.append(f"Sign {i+1}: {sign_class}")
    
    # If no signs detected, add message
    if not detection_info:
        detection_info.append("No traffic signs detected. Try with a different image.")
    
    # Convert back to RGB for display
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_image, "\n".join(detection_info)

def main():
    """Main function to run the traffic sign detection GUI."""
    # Check if test images exist
    for test_img in ["Test/00012.png", "Test/00152.png"]:
        if not os.path.exists(test_img):
            print(f"Warning: Example image {test_img} not found.")
    
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="pil", label="Detected Signs"),
            gr.Textbox(label="Detection Results")
        ],
        title="Traffic Sign Detection & Classification",
        description="Upload an image to detect and classify traffic signs, including large signs in real-world photos.",
        examples=[
            ["Test/00012.png"],
            ["Test/00152.png"]
        ]
    )
    
    try:
        interface.launch()
        print("Interface launched successfully. Open the URL shown above in your browser.")
    except Exception as e:
        print(f"Error launching interface: {e}")
        print("Make sure all dependencies are installed by running:")
        print("pip install opencv-python numpy matplotlib scikit-image joblib gradio tqdm")

if __name__ == "__main__":
    main()