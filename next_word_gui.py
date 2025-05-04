import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from collections import defaultdict
import time
import os

class MarkovChainTextPredictor:
    def __init__(self):
        self.model = defaultdict(list)
        self.trained = False
        self.training_time = 0
        self.vocab_size = 0
        self.unique_transitions = 0
        self.total_transitions = 0
        self.f1_score = 0
        self.file_size = 0
        self.file_name = ""
        
    def train(self, corpus, n=2):
        """Train model with performance tracking"""
        start_time = time.time()
        self.model.clear()
        tokens = corpus.split()
        self.vocab_size = len(set(tokens))
        
        transition_counts = defaultdict(int)
        #main Markov model Created 
        for i in range(len(tokens) - n + 1):
            key = tuple(tokens[i:i + n - 1])
            next_word = tokens[i + n - 1]
            self.model[key].append(next_word)
            transition_counts[(key, next_word)] += 1
        
        self.unique_transitions = len(transition_counts)
        self.total_transitions = sum(transition_counts.values())
        
        # Calculate F1 score approximation
        correct = sum(c > 1 for c in transition_counts.values())
        precision = correct / self.unique_transitions if self.unique_transitions else 0
        recall = correct / self.total_transitions if self.total_transitions else 0
        self.f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        
        self.training_time = time.time() - start_time
        self.trained = True
        
    def predict_next(self, text, top_k=3):
        """Generate predictions with timing"""
        if not self.trained:
            return ["⚠️ Please load training data first"]
            
        words = text.strip().split()
        if not words:
            return []
            
        key = tuple(words[-1:])
        candidates = self.model.get(key, [])
        
        if not candidates:
            return ["🤖 No predictions available"]
            
        freq = defaultdict(int)
        for word in candidates:
            freq[word] += 1
            
        sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [word for word, _ in sorted_words[:top_k]]
        
    def get_stats(self):
        """Return model performance metrics"""
        return {
            "file_name": self.file_name,
            "file_size": f"{self.file_size/1024:.1f} KB",
            "training_time": f"{self.training_time:.2f} seconds",
            "vocab_size": self.vocab_size,
            "unique_transitions": self.unique_transitions,
            "total_transitions": self.total_transitions,
            "f1_score": f"{self.f1_score:.3f}",
            "compression_ratio": f"{self.vocab_size/self.unique_transitions:.1f}x" if self.unique_transitions else "N/A"
        }

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📚 Next Word Predictor with File Upload")
        self.root.geometry("900x650")
        
        # Configure styles
        self.bg_color = "#f5f5f5"
        self.text_bg = "#ffffff"
        self.button_color = "#4a6fa5"
        self.highlight_color = "#166088"
        
        self.root.configure(bg=self.bg_color)
        
        # Create main container
        self.main_frame = tk.Frame(root, bg=self.bg_color)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # File upload frame
        self.upload_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.upload_frame.pack(fill="x", pady=(0, 10))
        
        self.upload_label = tk.Label(
            self.upload_frame,
            text="Upload Text File for Training:",
            bg=self.bg_color,
            font=("Segoe UI", 10)
        )
        self.upload_label.pack(side="left", padx=(0, 10))
        
        self.upload_btn = tk.Button(
            self.upload_frame,
            text="📁 Choose File",
            command=self.upload_file,
            bg=self.button_color,
            fg="white",
            font=("Segoe UI", 10),
            relief=tk.FLAT
        )
        self.upload_btn.pack(side="left")
        
        self.file_label = tk.Label(
            self.upload_frame,
            text="No file selected",
            bg=self.bg_color,
            fg="#666666",
            font=("Segoe UI", 9)
        )
        self.file_label.pack(side="left", padx=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.main_frame,
            orient="horizontal",
            mode="determinate",
            length=400
        )
        self.progress.pack(pady=(0, 10))
        self.progress.pack_forget()  # Hide initially
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 12),
            bg=self.text_bg,
            padx=10,
            pady=10,
            state='disabled'
        )
        self.chat_display.pack(fill="both", expand=True, pady=(0, 10))
        
        # Input frame
        self.input_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.input_frame.pack(fill="x", pady=5)
        
        self.user_input = tk.Entry(
            self.input_frame,
            font=("Segoe UI", 14),
            bg=self.text_bg,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=self.highlight_color,
            highlightbackground="#cccccc"
        )
        self.user_input.pack(fill="x", expand=True, side="left")
        self.user_input.bind("<Return>", self.process_input)
        
        # Button frame
        self.button_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.button_frame.pack(fill="x", pady=5)
        
        self.predict_btn = tk.Button(
            self.button_frame,
            text="🔮 Predict Next Word",
            command=lambda: self.process_input(),
            bg=self.highlight_color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=15
        )
        self.predict_btn.pack(side="left", padx=5)
        
        self.stats_btn = tk.Button(
            self.button_frame,
            text="📊 Show Statistics",
            command=self.show_statistics,
            bg=self.button_color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=15
        )
        self.stats_btn.pack(side="left", padx=5)
        
        self.clear_btn = tk.Button(
            self.button_frame,
            text="🗑️ Clear Chat",
            command=self.clear_chat,
            bg="#d9534f",
            fg="white",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            padx=15
        )
        self.clear_btn.pack(side="right", padx=5)
        
        # Initialize predictor
        self.predictor = MarkovChainTextPredictor()
        
        # Add welcome message
        self.add_message("🤖 Welcome to Next Word Predictor!\nPlease upload a text file to train the model.")
    
    def upload_file(self):
        """Handle file upload for training"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Select Training Data"
        )
        
        if file_path:
            try:
                self.file_label.config(text=os.path.basename(file_path))
                self.add_message(f"⚙️ Loading file: {os.path.basename(file_path)}...")
                self.root.update()
                
                # Show progress bar
                self.progress.pack()
                self.progress["value"] = 0
                self.root.update()
                
                # Read file with simulated progress
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Get file size for progress calculation
                    file.seek(0, 2)
                    file_size = file.tell()
                    file.seek(0)
                    self.predictor.file_size = file_size
                    self.predictor.file_name = os.path.basename(file_path)
                    
                    # Read in chunks to update progress
                    chunk_size = 1024 * 1024  # 1MB chunks
                    content = ""
                    while True:
                        chunk = file.read(chunk_size)
                        if not chunk:
                            break
                        content += chunk
                        progress = (file.tell() / file_size) * 100
                        self.progress["value"] = progress
                        self.root.update()
                
                self.add_message("⚙️ Training model... Please wait")
                self.root.update()
                
                # Train model
                self.predictor.train(content)
                
                # Hide progress bar
                self.progress.pack_forget()
                
                stats = self.predictor.get_stats()
                training_msg = (
                    f"✅ Training completed!\n"
                    f"📄 File: {stats['file_name']} ({stats['file_size']})\n"
                    f"⏱️ Time: {stats['training_time']}\n"
                    f"📖 Vocabulary: {stats['vocab_size']} words\n"
                    f"🔄 Transitions: {stats['unique_transitions']} unique\n"
                    f"📈 F1 Score: {stats['f1_score']}\n"
                    f"💾 Compression: {stats['compression_ratio']}"
                )
                
                self.add_message(training_msg)
                messagebox.showinfo("Training Complete", "Model trained successfully!")
                
            except Exception as e:
                self.progress.pack_forget()
                messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")
                self.add_message(f"❌ Error: {str(e)}")
                self.file_label.config(text="Upload failed", fg="red")
    
    def process_input(self, event=None):
        """Handle user input and generate predictions"""
        text = self.user_input.get().strip()
        if not text:
            return
            
        self.user_input.delete(0, tk.END)
        self.add_message(f"👤 You: {text}")
        
        start_time = time.time()
        predictions = self.predictor.predict_next(text)
        response_time = time.time() - start_time
        
        if predictions:
            response = (
                f"🤖 Predictions ({response_time:.3f}s):\n"
                f"  • {'  • '.join(predictions)}"
            )
            self.add_message(response)
    
    def show_statistics(self):
        """Display model performance statistics"""
        if not self.predictor.trained:
            messagebox.showwarning("No Data", "Please train the model first")
            return
            
        stats = self.predictor.get_stats()
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Model Statistics")
        stats_window.geometry("450x350")
        stats_window.resizable(False, False)
        
        stats_frame = tk.Frame(stats_window, padx=20, pady=20)
        stats_frame.pack(fill="both", expand=True)
        
        tk.Label(
            stats_frame,
            text="📊 Model Performance Metrics",
            font=("Segoe UI", 12, "bold")
        ).pack(pady=(0, 15))
        
        metrics = [
            ("Training File:", stats['file_name']),
            ("File Size:", stats['file_size']),
            ("Training Time:", stats['training_time']),
            ("Vocabulary Size:", f"{stats['vocab_size']} words"),
            ("Unique Transitions:", stats['unique_transitions']),
            ("Total Transitions:", stats['total_transitions']),
            ("F1 Score:", stats['f1_score']),
            ("Compression Ratio:", stats['compression_ratio'])
        ]
        
        for label, value in metrics:
            row = tk.Frame(stats_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=20, anchor="w", font=("Segoe UI", 9)).pack(side="left")
            tk.Label(row, text=value, anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left")
    
    def add_message(self, text):
        """Add message to chat display"""
        self.chat_display['state'] = 'normal'
        self.chat_display.insert(tk.END, text + "\n\n")
        self.chat_display.yview(tk.END)
        self.chat_display['state'] = 'disabled'
    
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display['state'] = 'normal'
        self.chat_display.delete(1.0, tk.END)
        self.chat_display['state'] = 'disabled'
        self.add_message("Chat cleared. Model remains trained.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
