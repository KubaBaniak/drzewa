import matplotlib.pyplot as plt

def show_images_and_masks(data, n_images=10, figsize=(25, 5), model=None):
    
    # Configuration
    if model is None:
        n_cols = 3
    else:
        n_cols = 5
    
    # Collect the data
    images, masks = next(iter(data))
    
    # Iterate over the data
    for n in range(n_images):
        
        # Plotting configuration
        plt.figure(figsize=figsize)
        
        # Plot the image
        plt.subplot(1, n_cols, 1)
        plt.title("Original Image")
        plt.imshow(images[n].permute(1, 2, 0))  # Assuming images are in CHW format
        plt.axis('off')
        
        # Plot the Mask
        plt.subplot(1, n_cols, 2)
        plt.title("Original Mask")
        plt.imshow(masks[n][0], cmap='gray')  # Assuming masks are single-channel
        plt.axis('off')
        
        # Plot image and mask overlay
        plt.subplot(1, n_cols, 3)
        plt.title('Image and Mask overlay')
        plt.imshow(masks[n][0], alpha=0.8, cmap='binary_r')
        plt.imshow(images[n].permute(1, 2, 0), alpha=0.5)
        plt.axis('off')
        
        # Model predictions
        if model is not None:
            with torch.no_grad():
                pred_mask = model(images[n].unsqueeze(0))
            pred_mask = (pred_mask >= 0.5).squeeze(0)
            
            plt.subplot(1, n_cols, 4)
            plt.title('Predicted Mask')
            plt.imshow(pred_mask, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, n_cols, 5)
            plt.title('Predicted Mask Overlay')
            plt.imshow(pred_mask, alpha=0.8, cmap='binary_r')
            plt.imshow(images[n].permute(1, 2, 0), alpha=0.5)
            plt.axis('off')
    
        # Show final plot
        plt.show()