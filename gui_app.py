import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
import threading
import logging
from typing import TYPE_CHECKING # For type hinting without circular import

if TYPE_CHECKING:
    from model_handler import ModelHandler # Only for type checking

logger = logging.getLogger(__name__)

class DigitRecognizerApp(tk.Tk):
    """
    A Tkinter application for handwritten digit recognition.
    Allows users to draw digits and get predictions from a Keras model.
    """
    CANVAS_WIDTH = 600  # Wider canvas for better drawing
    CANVAS_HEIGHT = 300  # Shorter height but still comfortable
    CONTROL_WIDTH = 120  # Slimmer control panel
    BRUSH_RADIUS = 8     # Smaller brush for more precise drawing
    PREDICTION_THRESHOLD = 0.6  # Threshold for prediction confidence

    def __init__(self, model_handler: 'ModelHandler'):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.geometry(f"{self.CANVAS_WIDTH + self.CONTROL_WIDTH + 20}x{self.CANVAS_HEIGHT + 40}")  # Wide window size
        self.resizable(False, False)

        self.model_handler = model_handler
        self.last_x, self.last_y = 0, 0
        self.eraser_mode = False
        self.undo_stack = []
        self.redo_stack = []

        # Image buffer for drawing and prediction
        self.image = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), 255)
        self.draw = ImageDraw.Draw(self.image)

        self._create_widgets()
        self._setup_layout()
        self._bind_events()

        logger.info("GUI application initialized.")

    def _create_widgets(self):
        """Creates and configures all GUI widgets."""
        # Header
        self.header_label = ttk.Label(self, text="Handwritten Digit Recognition", font=("Helvetica", 14, "bold"), background="#eaf6fb", anchor="center")

        # Canvas with wider size
        self.canvas = tk.Canvas(self, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT,
                                bg="white", cursor="cross", bd=2, relief="sunken", highlightthickness=2, highlightbackground="#4A90E2")

        # Drawing hint
        self.hint_label = ttk.Label(self, text="Draw a digit (0-9) clearly in the box.", font=("Helvetica", 12), background="#eaf6fb", anchor="center")

        # Prediction result
        self.result_digit_var = tk.StringVar(value="?")
        self.result_digit_label = ttk.Label(self, textvariable=self.result_digit_var, font=("Helvetica", 48, "bold"), foreground="#4A90E2", background="#f0f0f0", anchor="center")

        self.confidence_var = tk.StringVar(value="Confidence: --%")
        self.confidence_label = ttk.Label(self, textvariable=self.confidence_var, font=("Helvetica", 14), background="#f0f0f0", anchor="center")

        self.label_var = tk.StringVar(value="Draw a digit!")
        self.prediction_label = ttk.Label(self, textvariable=self.label_var, font=("Helvetica", 16), anchor="center", background="#f0f0f0")

        # Buttons
        self.classify_btn = ttk.Button(self, text="Recognise", command=self.start_prediction_thread, style="Accent.TButton")
        self.clear_btn = ttk.Button(self, text="Clear", command=self.clear_all)
        self.eraser_btn = ttk.Button(self, text="Eraser", command=self.toggle_eraser)
        self.undo_btn = ttk.Button(self, text="Undo", command=self.undo_action)
        self.redo_btn = ttk.Button(self, text="Redo", command=self.redo_action)
        self.save_btn = ttk.Button(self, text="Save Drawing", command=self.save_drawing)

        # Tooltips
        self._add_tooltip(self.classify_btn, "Recognise the digit you drew")
        self._add_tooltip(self.clear_btn, "Clear the canvas")
        self._add_tooltip(self.eraser_btn, "Toggle eraser mode")
        self._add_tooltip(self.undo_btn, "Undo last action")
        self._add_tooltip(self.redo_btn, "Redo last undone action")
        self._add_tooltip(self.save_btn, "Save your drawing as an image file")

        # Progress bar for visual feedback during prediction
        self.progress_bar = ttk.Progressbar(self, mode='indeterminate', length=150)

        # Style for buttons
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="#fff", background="#4A90E2", font=("Helvetica", 10, "bold"))
        style.configure("Compact.TButton", font=("Helvetica", 9), padding=1)  # Smaller buttons
        style.configure("TLabel", background="#eaf6fb")

    def _setup_layout(self):
        """Arranges widgets using the grid layout manager."""
        self.configure(background="#eaf6fb")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # Header
        self.header_label.grid(row=0, column=0, columnspan=2, pady=(5,0), sticky="ew")

        # Canvas and hint
        self.canvas.grid(row=1, column=0, padx=(10,2), pady=(5,5), sticky="nsew")
        self.hint_label.configure(font=("Helvetica", 10))  # Smaller hint text
        self.hint_label.grid(row=2, column=0, padx=(10,2), pady=(0,5), sticky="ew")

        # Right panel for results and controls - very compact
        right_frame = ttk.Frame(self, style="TLabel", width=self.CONTROL_WIDTH)
        right_frame.grid(row=1, column=1, rowspan=2, padx=(2,5), pady=(5,5), sticky="nsew")
        right_frame.grid_propagate(False)  # Prevent frame from shrinking
        
        # Configure right frame rows
        for i in range(10):  # More rows for very compact layout
            right_frame.grid_rowconfigure(i, weight=0)
        right_frame.grid_columnconfigure(0, weight=1)

        # Prediction result (very compact)
        self.result_digit_label.configure(font=("Helvetica", 36, "bold"))  # Slightly smaller result
        self.result_digit_label.grid(row=0, column=0, pady=(0,2), sticky="ew")
        self.confidence_label.configure(font=("Helvetica", 10))  # Smaller confidence text
        self.confidence_label.grid(row=1, column=0, pady=(0,2), sticky="ew")
        self.prediction_label.configure(font=("Helvetica", 10))  # Smaller prediction text
        self.prediction_label.grid(row=2, column=0, pady=(0,2), sticky="ew")
        self.progress_bar.configure(length=100)  # Shorter progress bar
        self.progress_bar.grid(row=3, column=0, pady=(0,5), sticky="ew")

        # Compact button panel
        button_frame = ttk.Frame(right_frame)
        button_frame.grid(row=4, column=0, sticky="ew", pady=(5,0))
        button_frame.grid_columnconfigure(0, weight=1)

        # Style for compact buttons
        style = ttk.Style()
        style.configure("Compact.TButton", padding=2)
        
        # All buttons in a vertical layout
        buttons = [
            (self.classify_btn, "Recognise"),    # Main action
            (self.eraser_btn, "Eraser"),        # Drawing tools
            (self.undo_btn, "Undo"),
            (self.redo_btn, "Redo"),
            (self.clear_btn, "Clear"),
            (self.save_btn, "Save")
        ]
        
        for i, (btn, text) in enumerate(buttons):
            btn.configure(style="Compact.TButton", text=text)
            btn.grid(row=i, column=0, padx=2, pady=2, sticky="ew")

    def _add_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        label = tk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1, font=("Helvetica", 10))
        label.pack()
        def enter(event):
            x = widget.winfo_rootx() + 50
            y = widget.winfo_rooty() + 20
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()
        def leave(event):
            tooltip.withdraw()
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def save_drawing(self):
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.image.save(file_path)

    def _bind_events(self):
        """Binds mouse events to drawing functions."""
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw_lines)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

    def _start_draw(self, event):
        """Initializes drawing coordinates when mouse button is pressed."""
        self.last_x, self.last_y = event.x, event.y
        self._save_undo_state()
        r = self.BRUSH_RADIUS
        color = 255 if self.eraser_mode else 0
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                                fill='white' if self.eraser_mode else 'black', outline='white' if self.eraser_mode else 'black', width=0, tags="drawing")
        self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=color)


    def _draw_lines(self, event):
        """Draws lines on canvas and PIL image buffer as mouse moves."""
        current_x, current_y = event.x, event.y
        r = self.BRUSH_RADIUS
        color = 255 if self.eraser_mode else 0
        canvas_color = 'white' if self.eraser_mode else 'black'

        self.canvas.create_line(self.last_x, self.last_y, current_x, current_y,
                                fill=canvas_color, width=r * 2, capstyle=tk.ROUND,
                                smooth=tk.TRUE, tags="drawing")
        self.draw.line([self.last_x, self.last_y, current_x, current_y],
                       fill=color, width=r * 2, joint="curve")
        self.last_x, self.last_y = current_x, current_y

    def _stop_draw(self, event):
        """Action on mouse button release (currently no specific action)."""
        logger.debug("Drawing stopped.")
        pass

    def clear_all(self):
        """Clears the canvas and resets the image buffer."""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_var.set("Draw a digit!")
        self.result_digit_var.set("?")
        self.confidence_var.set("Confidence: --%")
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.eraser_mode = False
        logger.info("Canvas and image buffer cleared.")

    def toggle_eraser(self):
        """Toggle eraser mode on/off."""
        self.eraser_mode = not self.eraser_mode
        self.eraser_btn.config(text="Draw" if self.eraser_mode else "Eraser")

    def _save_undo_state(self):
        """Save current image for undo stack."""
        self.undo_stack.append(self.image.copy())
        self.redo_stack.clear()

    def undo_action(self):
        """Undo last drawing action."""
        if self.undo_stack:
            self.redo_stack.append(self.image.copy())
            self.image = self.undo_stack.pop()
            self.draw = ImageDraw.Draw(self.image)
            self._update_canvas_from_image()

    def redo_action(self):
        """Redo last undone action."""
        if self.redo_stack:
            self.undo_stack.append(self.image.copy())
            self.image = self.redo_stack.pop()
            self.draw = ImageDraw.Draw(self.image)
            self._update_canvas_from_image()

    def _update_canvas_from_image(self):
        """Update the Tkinter canvas from the PIL image buffer."""
        self.canvas.delete("all")
        tk_img = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self.canvas.image = tk_img

    def start_prediction_thread(self):
        """Starts the prediction in a separate thread to keep the GUI responsive."""
        self.classify_btn.config(state=tk.DISABLED) # Disable button to prevent multiple clicks
        self.clear_btn.config(state=tk.DISABLED)
        self.label_var.set("Processing...")
        self.progress_bar.start() # Start indeterminate progress bar
        
        logger.info("Starting prediction thread.")
        prediction_thread = threading.Thread(target=self._run_prediction)
        prediction_thread.daemon = True # Allow thread to exit with main app
        prediction_thread.start()

    def _run_prediction(self):
        """Performs the actual digit prediction (runs in a separate thread)."""
        try:
            digit, acc = self.model_handler.predict_digit(self.image)
            
            # Update GUI elements back in the main thread
            self.after(0, self._update_gui_after_prediction, digit, acc)
        except Exception as e:
            logger.error(f"Error in prediction thread: {e}", exc_info=True)
            # Use a bound method instead of lambda to avoid variable capture issues
            self.after(0, self._handle_prediction_error, e)

    def _update_gui_after_prediction(self, digit: int, acc: float):
        """Updates the GUI with prediction results (called from main thread)."""
        self.progress_bar.stop()
        self.classify_btn.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.NORMAL)

        if digit == -1:
            self.result_digit_var.set("?")
            self.confidence_var.set("Confidence: --%")
            self.label_var.set("Error predicting!")
            self.result_digit_label.config(foreground="red")
            messagebox.showerror("Prediction Error", "An error occurred during digit prediction. Check logs.")
            logger.warning("Prediction resulted in an error state.")
            return

        self.result_digit_var.set(str(digit))
        self.confidence_var.set(f"Confidence: {int(acc * 100)}%")
        if acc < self.PREDICTION_THRESHOLD:
            self.label_var.set("Uncertain prediction. Try drawing more clearly.")
            self.result_digit_label.config(foreground="red")
            logger.info(f"Prediction: {digit} with low confidence ({acc*100:.2f}%).")
        else:
            self.label_var.set("Prediction successful!")
            self.result_digit_label.config(foreground="#4A90E2")
            logger.info(f"Prediction: {digit} with confidence {acc*100:.2f}%.")

    def _handle_prediction_error(self, error: Exception):
        """Handles errors from the prediction thread by updating GUI and logging."""
        self.progress_bar.stop()
        self.classify_btn.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.NORMAL)
        self.label_var.set("Error!")
        messagebox.showerror("Prediction Error", f"An unexpected error occurred during prediction:\n{error}\nCheck application logs.")
        logger.error(f"Handled error from prediction thread: {error}")