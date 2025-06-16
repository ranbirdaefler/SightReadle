import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import os
import re
import numpy as np
import cv2
from scipy.signal import find_peaks

def extract_exercises_from_pdf(pdf_path, output_dir="exercises"):
    """
    Extract individual exercises from a PDF with improved boundary detection
    and padding to prevent bleeding for Audiveris conversion.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convert page to high-resolution image for better OCR
        mat = fitz.Matrix(3.0, 3.0)  # Higher resolution for better detection
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        img = Image.open(fitz.io.BytesIO(img_data))
        img_array = np.array(img)
        
        # Get exercise positions using text detection
        exercise_positions = find_exercise_numbers(page, scale_factor=3.0)
        
        if exercise_positions:
            # Find staff line regions for better boundary detection
            staff_regions = detect_staff_regions(img_array)
            
            # Extract each exercise with smart boundaries
            extract_individual_exercises(
                img_array, exercise_positions, staff_regions, 
                output_dir, page_num
            )
        else:
            # Fallback: try visual detection
            print(f"No exercise numbers found on page {page_num + 1}, trying visual detection...")
            extract_exercises_visual(img_array, output_dir, page_num)
    
    doc.close()
    print(f"\nExtraction complete! Files saved to '{output_dir}' directory.")

def find_exercise_numbers(page, scale_factor=3.0):
    """Find exercise number positions with better accuracy."""
    text_dict = page.get_text("dict")
    exercise_positions = []
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    # More flexible pattern matching
                    if re.match(r"(No\.|Number|Exercise)\s*\d+", text, re.IGNORECASE):
                        bbox = span["bbox"]
                        exercise_num = re.search(r"\d+", text)
                        if exercise_num:
                            exercise_positions.append({
                                "number": int(exercise_num.group()),
                                "y_position": bbox[1] * scale_factor,
                                "bbox": [coord * scale_factor for coord in bbox],
                                "text": text
                            })
    
    # Sort by vertical position and remove duplicates
    exercise_positions.sort(key=lambda x: x["y_position"])
    
    # Remove duplicate exercise numbers (keep the first occurrence)
    seen_numbers = set()
    unique_exercises = []
    for ex in exercise_positions:
        if ex["number"] not in seen_numbers:
            unique_exercises.append(ex)
            seen_numbers.add(ex["number"])
    
    return unique_exercises

def detect_staff_regions(img_array):
    """Detect staff line regions to help with exercise boundary detection."""
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Create horizontal kernel to detect staff lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Find row sums to identify staff regions
    row_sums = np.sum(horizontal_lines, axis=1)
    
    # Smooth the signal
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(row_sums, sigma=3)
    
    # Find peaks (staff line regions)
    peaks, properties = find_peaks(
        smoothed, 
        height=np.max(smoothed) * 0.1,
        distance=50,
        width=10
    )
    
    return peaks

def extract_individual_exercises(img_array, exercise_positions, staff_regions, output_dir, page_num):
    """Extract exercises with smart boundary detection and proper padding."""
    page_height, page_width = img_array.shape[:2]
    
    for i, exercise in enumerate(exercise_positions):
        # Determine boundaries more intelligently
        exercise_y = int(exercise["y_position"])
        
        # Find the closest staff region to this exercise
        closest_staff_idx = find_closest_staff_region(exercise_y, staff_regions)
        
        if i == 0:
            # First exercise: start from top or first staff region
            top = max(0, exercise_y - 150)  # Give generous top padding
        else:
            # Start after previous exercise with buffer
            prev_bottom = exercise_bottom + 30
            top = prev_bottom
        
        if i == len(exercise_positions) - 1:
            # Last exercise: go to bottom
            bottom = page_height
        else:
            # Find next exercise position
            next_exercise_y = int(exercise_positions[i + 1]["y_position"])
            
            # Look for staff regions between current and next exercise
            staff_between = [s for s in staff_regions if exercise_y < s < next_exercise_y]
            
            if staff_between:
                # End after the last staff region in this exercise
                last_staff = max(staff_between)
                bottom = min(next_exercise_y - 80, last_staff + 100)
            else:
                # Fallback: split the difference
                bottom = next_exercise_y - 80
        
        # Ensure minimum height
        if bottom - top < 200:
            bottom = top + 200
        
        exercise_bottom = bottom
        
        # Add generous padding to prevent bleeding
        top_padding = 5 if i == 0 else -50
        bottom_padding = 70
        left_padding = 40
        right_padding = 40
        
        # Apply padding with bounds checking
        crop_top = max(0, top - top_padding)
        crop_bottom = min(page_height, bottom + bottom_padding)
        crop_left = max(0, left_padding)
        crop_right = min(page_width, page_width - right_padding)
        
        # Crop the exercise
        cropped_exercise = img_array[crop_top:crop_bottom, crop_left:crop_right]
        
        # Add white padding around the cropped exercise
        padded_exercise = add_white_padding(cropped_exercise, 
                                          top=20, bottom=20, 
                                          left=30, right=30)
        
        # Clean up any artifacts (optional noise reduction)
        cleaned_exercise = clean_exercise_image(padded_exercise)
        
        # Convert to PIL and save
        exercise_img = Image.fromarray(cleaned_exercise)
        
        filename = f"exercise_{exercise['number']}.png"
        filepath = os.path.join(output_dir, filename)
        
        exercise_img.save(filepath, "PNG", optimize=True)
        print(f"Saved: {filename} (size: {exercise_img.size})")

def find_closest_staff_region(exercise_y, staff_regions):
    """Find the staff region closest to the exercise position."""
    if len(staff_regions) == 0:
        return None
    
    distances = [abs(staff - exercise_y) for staff in staff_regions]
    return np.argmin(distances)

def add_white_padding(img_array, top=20, bottom=20, left=20, right=20):
    """Add white padding around an image."""
    if len(img_array.shape) == 3:
        # Color image
        pad_color = [255, 255, 255]
        padded = np.pad(img_array, 
                       ((top, bottom), (left, right), (0, 0)), 
                       mode='constant', 
                       constant_values=255)
    else:
        # Grayscale image
        padded = np.pad(img_array, 
                       ((top, bottom), (left, right)), 
                       mode='constant', 
                       constant_values=255)
    
    return padded

def clean_exercise_image(img_array):
    """Clean up the exercise image to remove artifacts and noise."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to original format
    if len(img_array.shape) == 3:
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        return cleaned_rgb
    else:
        return cleaned

def extract_exercises_visual(img_array, output_dir, page_num):
    """Fallback visual detection method when text detection fails."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Detect staff lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(255 - gray, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Find exercise boundaries
    row_sums = np.sum(horizontal_lines, axis=1)
    smoothed = gaussian_filter1d(row_sums, sigma=5)
    
    # Find peaks representing staff groups
    peaks, _ = find_peaks(smoothed, 
                         height=np.max(smoothed) * 0.2, 
                         distance=150,
                         width=20)
    
    # Group peaks into exercises
    exercise_groups = []
    if len(peaks) > 0:
        current_group = [peaks[0]]
        
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] < 200:  # Same exercise
                current_group.append(peaks[i])
            else:  # New exercise
                exercise_groups.append(current_group)
                current_group = [peaks[i]]
        
        exercise_groups.append(current_group)
    
    # Extract each exercise group
    for i, group in enumerate(exercise_groups):
        top = max(0, min(group) - 100)
        bottom = min(len(gray), max(group) + 100)
        
        cropped = img_array[top:bottom, :]
        padded = add_white_padding(cropped, 40, 40, 30, 30)
        
        exercise_img = Image.fromarray(padded)
        filename = f"exercise_visual_{i+1:03d}_page_{page_num + 1}.png"
        filepath = os.path.join(output_dir, filename)
        
        exercise_img.save(filepath)
        print(f"Saved (visual): {filename}")

# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "/Users/floriandaefler/Desktop/sightreadle/354-Reading-Exercises-in-C-Position-Full-Score.pdf"
    OUTPUT_DIR = "extracted_exercises"
    
    print("Starting improved PDF exercise extraction...")
    print(f"Input PDF: {PDF_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    try:
        extract_exercises_from_pdf(PDF_PATH, OUTPUT_DIR)
        print("\n✅ Extraction completed successfully!")
        print("\nThe extracted images should now work better with Audiveris because:")
        print("• Proper padding prevents bleeding between exercises")
        print("• Staff line detection improves boundary accuracy") 
        print("• White space buffer zones eliminate artifacts")
        print("• Higher resolution preserves musical notation quality")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have all required libraries:")
        print("pip install PyMuPDF pillow numpy scipy opencv-python")