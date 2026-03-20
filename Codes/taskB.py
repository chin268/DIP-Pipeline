import cv2
import numpy as np
import os

def extract_paragraphs_as_blocks(image_path, output_folder, column_count=3):
    # Read and preprocess image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Get vertical projection for column detection
    vertical_histogram = np.sum(binary, axis=0)
    
    # Find column boundaries
    if column_count > 1:
        # Smooth histogram for better valley detection
        kernel_size = 50
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_histogram = np.convolve(vertical_histogram, kernel, mode='same')
        
        # Identify valleys as column separators
        threshold = np.mean(smoothed_histogram) * 0.2  # Adjust threshold for column separation
        valleys = np.where(smoothed_histogram < threshold)[0]
        
        # Group valleys to determine column boundaries
        column_boundaries = []
        if len(valleys) > 0:
            current_group = [valleys[0]]
            for i in range(1, len(valleys)):
                if valleys[i] - valleys[i-1] < 50:  # Group nearby valleys
                    current_group.append(valleys[i])
                else:
                    column_boundaries.append(int(np.mean(current_group)))
                    current_group = [valleys[i]]
            column_boundaries.append(int(np.mean(current_group)))

            # Ensure columns cover the entire image width
            column_boundaries = [0] + column_boundaries + [binary.shape[1]]
        else:
            # Fallback: Divide into equal-width columns
            col_width = binary.shape[1] // column_count
            column_boundaries = [i * col_width for i in range(column_count)] + [binary.shape[1]]
    else:
        column_boundaries = [0, binary.shape[1]]
    
    # Process each column separately
    paragraph_count = 1
    for i in range(len(column_boundaries) - 1):
        col_start, col_end = column_boundaries[i], column_boundaries[i + 1]
        column_binary = binary[:, col_start:col_end]
        
        # Get horizontal projection for paragraph detection
        horizontal_histogram = np.sum(column_binary, axis=1)
        row_indices = np.where(horizontal_histogram > 50)[0]  # Adjust threshold as needed
        
        # Find paragraph boundaries
        if len(row_indices) > 0:
            current_paragraph = [row_indices[0]]
            for j in range(1, len(row_indices)):
                if row_indices[j] - row_indices[j-1] > 20:  # Gap threshold for new paragraph
                    if len(current_paragraph) > 10:  # Minimum paragraph height
                        row_start = min(current_paragraph)
                        row_end = max(current_paragraph)
                        
                        # *Extract paragraph image*
                        paragraph_image = binary[
                            max(0, row_start - 5):min(binary.shape[0], row_end + 5),
                            max(0, col_start - 5):min(binary.shape[1], col_end + 5)
                        ]
                        
                        # Save paragraph if it contains enough content
                        if np.sum(paragraph_image) > 500:
                            paragraph_image = cv2.bitwise_not(paragraph_image)  # Invert colors
                            cv2.imwrite(f"{output_folder}/paragraph_{paragraph_count}.png", paragraph_image)
                            paragraph_count += 1
                    
                    current_paragraph = [row_indices[j]]
                else:
                    current_paragraph.append(row_indices[j])
            
            # Save the last paragraph in the column
            if len(current_paragraph) > 10:
                row_start = min(current_paragraph)
                row_end = max(current_paragraph)
                paragraph_image = binary[
                    max(0, row_start - 5):min(binary.shape[0], row_end + 5),
                    max(0, col_start - 5):min(binary.shape[1], col_end + 5)
                ]
                if np.sum(paragraph_image) > 500:
                    paragraph_image = cv2.bitwise_not(paragraph_image)  # Invert colors
                    cv2.imwrite(f"{output_folder}/paragraph_{paragraph_count}.png", paragraph_image)
                    paragraph_count += 1
    
    return paragraph_count - 1


# Test the code
image_path = input("Enter the path of the image: ")
output_folder = r"C:\Users\jaron\.spyder-py3\output_folder"

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the number of columns from the user
column_count = int(input("Enter the number of columns: "))

# Extract paragraphs from the image
num_paragraphs = extract_paragraphs_as_blocks(image_path, output_folder, column_count)
print(f"Found {num_paragraphs} paragraphs")














