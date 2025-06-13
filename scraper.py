import fitz  # PyMuPDF
from PIL import Image
import os
import re
import numpy as np

def extract_exercises_from_pdf(pdf_path, output_dir="exercises"):
    """
    Extract individual exercises from a PDF and save them as PNG files.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_dir (str): Directory to save the extracted exercise images
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convert page to image
        mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        img = Image.open(fitz.io.BytesIO(img_data))
        img_array = np.array(img)
        
        # Search for exercise numbers in the text
        text_dict = page.get_text("dict")
        exercise_positions = []
        
        # Find all "No. X" patterns and their positions
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if re.match(r"No\.\s*\d+", text):
                            # Get the bounding box of the text
                            bbox = span["bbox"]
                            exercise_num = re.search(r"\d+", text).group()
                            exercise_positions.append({
                                "number": int(exercise_num),
                                "y_position": bbox[1],  # Top y-coordinate
                                "bbox": bbox
                            })
        
        # Sort exercises by their vertical position
        exercise_positions.sort(key=lambda x: x["y_position"])
        
        # If we found exercises, crop and save them
        if exercise_positions:
            page_height = img_array.shape[0]
            
            for i, exercise in enumerate(exercise_positions):
                # Determine crop boundaries
                if i == 0:
                    # First exercise: start from top of page
                    top = 0
                else:
                    # Start from previous exercise's end
                    top = crop_bottom
                
                if i == len(exercise_positions) - 1:
                    # Last exercise: go to bottom of page
                    bottom = page_height
                else:
                    # End at next exercise's start (with some padding)
                    next_exercise_y = int(exercise_positions[i + 1]["y_position"] * 2)  # Scale factor
                    bottom = max(next_exercise_y - 20, top + 100)  # Minimum height
                
                crop_bottom = bottom
                
                # Add padding around the exercise (more padding at top to avoid bleeding)
                top_padding = 40  # Extra padding at top to avoid bleeding from previous exercise
                bottom_padding = 20
                top = max(0, top - top_padding)
                bottom = min(page_height, bottom + bottom_padding)
                
                # Crop the exercise
                cropped_exercise = img_array[top:bottom, :]
                
                # Convert back to PIL Image and save
                exercise_img = Image.fromarray(cropped_exercise)
                
                # Generate filename - simple sequential numbering
                filename = f"exercise_{exercise['number']}.png"
                filepath = os.path.join(output_dir, filename)
                
                exercise_img.save(filepath)
                print(f"Saved: {filename}")
        
        else:
            # If no exercises found, save the whole page
            filename = f"page_{page_num + 1}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"No exercises detected. Saved whole page: {filename}")
    
    doc.close()
    print(f"\nExtraction complete! Files saved to '{output_dir}' directory.")

def extract_exercises_alternative(pdf_path, output_dir="exercises"):
    """
    Alternative method using visual detection of staff lines to separate exercises.
    Use this if the text-based method doesn't work well.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convert to high-resolution image
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(fitz.io.BytesIO(img_data))
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Find horizontal lines (staff lines)
        horizontal_kernel = np.ones((1, 50), np.uint8)
        horizontal_lines = cv2.morphologyEx(255 - img_array, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find regions with multiple staff lines (indicating exercises)
        row_sums = np.sum(horizontal_lines, axis=1)
        
        # Find peaks in row sums (areas with many horizontal lines)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(row_sums, height=np.max(row_sums) * 0.3, distance=100)
        
        # Group peaks into exercises
        exercise_regions = []
        current_start = 0
        
        for i, peak in enumerate(peaks):
            if i > 0 and peak - peaks[i-1] > 200:  # Large gap indicates new exercise
                exercise_regions.append((current_start, peaks[i-1] + 100))
                current_start = peak - 100
        
        # Add the last region
        if peaks.size > 0:
            exercise_regions.append((current_start, len(img_array)))
        
        # Save each exercise region
        for i, (start, end) in enumerate(exercise_regions):
            cropped = img_array[start:end, :]
            exercise_img = Image.fromarray(cropped)
            
            filename = f"exercise_{i+1}_page_{page_num + 1}_alt.png"
            filepath = os.path.join(output_dir, filename)
            exercise_img.save(filepath)
            print(f"Saved: {filename}")
    
    doc.close()

# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_PATH = "/Users/floriandaefler/Desktop/sightreadle/354-Reading-Exercises-in-C-Position-Full-Score.pdf"  # Change this to your PDF file path
    OUTPUT_DIR = "extracted_exercises"
    
    print("Starting PDF exercise extraction...")
    print(f"Input PDF: {PDF_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    try:
        extract_exercises_from_pdf(PDF_PATH, OUTPUT_DIR)
    except Exception as e:
        print(f"Error with primary method: {e}")
        print("Trying alternative method...")
        try:
            extract_exercises_alternative(PDF_PATH, OUTPUT_DIR)
        except Exception as e2:
            print(f"Error with alternative method: {e2}")
            print("Please check that the required libraries are installed:")
            print("pip install PyMuPDF pillow numpy scipy opencv-python")

# Required libraries installation command:
# pip install PyMuPDF pillow numpy scipy opencv-python