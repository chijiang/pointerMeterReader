from app import create_gradio_interface


def main():
    """Main function to launch the application"""
    print("Initializing Meter Reading Extraction App...")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    print("Launching Gradio interface...")
    interface.launch(
        server_name="127.0.0.1",  # Localhost only
        server_port=7860,         # Use different port to avoid conflict
        share=False,              # Set to True for public sharing
        debug=True
    )


if __name__ == "__main__":
    main() 