# Digital Image Processing Pipeline

## Overview
This project applies digital image and video processing techniques to solve real-world multimedia challenges. It was developed as part of the CSC2014 course. The repository features an automated video processing pipeline designed for content creators and a precise text extraction tool for multi-column academic papers.

## Features

### 1. YouTube Video Processing (Task A)
Automates key video processing workflows, including daytime/nighttime identification, face blurring, video overlay, watermarking, and end-screen addition.
Utilizes OpenCV's Haar Cascade Classifier to detect facial regions.
Applies Gaussian blur to identified faces frame-by-frame, effectively masking identities while maintaining the original frame rate and resolution.

### 2. Paragraph Extraction from Scientific Papers (Task B)
Recognizes, extracts, and stores text paragraphs from multi-column scientific paper images.
Converts images to grayscale and binarizes them using thresholding to analyze pixel intensities.
Constructs vertical and horizontal projection histograms to accurately detect column boundaries and separate individual text blocks.
Successfully handles single-column, double-column, and triple-column layouts.
Inverts extracted paragraph images for better readability before saving.

## Technologies Used
**Python** 
**OpenCV** 
**NumPy** 
