// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Function to remove watermark
    function removeWatermark() {
        const watermark = document.querySelector('.MuiStack-root.watermark.css-1705j0v');
        if (watermark) {
            watermark.remove();
        }
    }

    // Initial removal
    removeWatermark();

    // Create observer to handle dynamically added elements
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length) {
                removeWatermark();
            }
        });
    });

    // Start observing the document with the configured parameters
    observer.observe(document.body, { childList: true, subtree: true });
});
