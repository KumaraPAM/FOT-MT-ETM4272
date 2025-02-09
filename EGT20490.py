import cv2
import matplotlib.pyplot as plt

def process_video_and_plot_liquid_level(video_path):
    """
    Processes a video to calculate the liquid level in each frame, 
    and plots the liquid level over time.

    Parameters:
    - video_path (str): Path to the input video file.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame to use as the background
    ret, background = cap.read()

    # List to store the liquid level percentages for plotting
    liquid_levels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Subtract the background from the current frame
        img_sub = cv2.subtract(background, frame)

        # Convert the subtracted image to grayscale
        img_sub_gray = cv2.cvtColor(img_sub, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain a binary image
        ret, img_sub_thresh = cv2.threshold(img_sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Print the used threshold value
        print(f'Threshold Value: {ret}')

        # Find contours in the binary image
        contours, _ = cv2.findContours(img_sub_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Calculate the total area of the image
        total_area = img_sub_thresh.size
        print(f'Total Area: {total_area}')

        # Calculate the percentage of the largest contour area in the image
        largest_contour_area = cv2.contourArea(contours[0]) if contours else 0
        percentage = round((largest_contour_area / total_area) * 100, 1)
        liquid_levels.append(percentage)

        # Print the percentage of liquid level
        print(f'Percentage: {percentage} %')

        # Display the original frame, subtracted image, and thresholded image
        cv2.imshow('Video', frame)
        cv2.imshow('Sub', img_sub)
        cv2.imshow('Sub Thresh', img_sub_thresh)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot the liquid level over time (video frames)
    plt.plot(liquid_levels, label='Liquid Level (%)')
    plt.xlabel('Frame Number')
    plt.ylabel('Liquid Level (%)')
    plt.title('Liquid Level Plot Over Video Duration')
    plt.legend()
    plt.show()

# Example usage:
video_path = 'demo2_images/liquidVideo.mp4'
process_video_and_plot_liquid_level(video_path)
