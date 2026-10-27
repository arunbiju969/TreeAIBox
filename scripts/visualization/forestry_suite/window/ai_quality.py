"""AI segmentation-quality analysis (screenshots + LLM). (TreeVisualizerGUI mixin)."""

import os
import sys
import json
import base64
import tempfile
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QMessageBox, QProgressBar, QDialogButtonBox, QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from forestry_suite.widgets import ModernButton

AI_ANALYSIS_PROMPT = """You are an AI agent evaluating tree segmentation quality in point cloud data from aerial leaf-off LiDAR imagery. Analyze these images from different views of a tree and provide your assessment in JSON format only.

Context: These images show point clouds colored by RGB values from aerial orthoimagery (leaf-off condition):
- DECIDUOUS TREES: Appear grey to white in color (branches visible without leaves)
- CONIFEROUS TREES: Appear green in color (foliage present year-round) with a cone/pyramidal shape. NOTE: Coniferous trees may have white/grey coloring at the BASE (trunk area) and green coloring in the CROWN/TOP - look at the upper portion to identify tree type
- NOISE/GROUND: Random scattered points, no organized structure, no distinct tree crowns
- MULTIPLE TREES: Several distinct tree structures/crowns visible in the same segment (segmentation error)

IMPORTANT: Do NOT mark multiple tree detection as noise. Multiple trees are valid tree structures but indicate a segmentation error where two or more trees were grouped together.

First, determine what this segment contains: noise/ground, a single tree, or multiple trees. Then provide appropriate analysis.

Please respond with a valid JSON object.

If the point cloud appears to be noise or random points (NO distinct tree structure), use this format:
{
  "is_tree": "no",
  "appears_to_be_noise": "yes",
  "recommended_action": "convert_to_noise_class",
  "noise_itc_value": 9999,
  "analysis_summary": "brief description of why this appears to be noise rather than a tree"
}

If the point cloud contains MULTIPLE DISTINCT TREES, use this format:
{
  "is_tree": "yes",
  "appears_to_be_noise": "no",
  "multiple_trees_detected": "yes",
  "total_trunks_detected": integer (count the number of distinct tree trunks/structures),
  "distinct_crowns": integer (count the number of separate crown structures visible),
  "trunk_disconnected": "yes",
  "tree_type": "mixed" or list individual types observed,
  "analysis_summary": "The point cloud contains multiple trees rather than a single tree, with several distinct tree structures visible across different views. The presence of multiple trees indicates that the segmentation is not accurate for a single tree. Recommend re-segmentation to separate individual trees."
}

If the point cloud appears to be a SINGLE VALID TREE, use this format:
{
  "is_tree": "yes",
  "appears_to_be_noise": "no",
  "multiple_trees_detected": "no",
  "tree_type": "coniferous" or "deciduous" (coniferous trees show GREEN coloring in the CROWN/TOP with cone/pyramidal shape, may have white/grey at base; deciduous trees show GREY/WHITE coloring overall with branching structure visible),
  "trunk_well_defined": "yes" or "no" (answer "yes" if the central trunk forms a single, continuous, well-connected vertical structure from base to crown - check for large disconnects or gaps),
  "trunk_disconnected": "yes" or "no" (answer "yes" if the central trunk appears broken into separate large segments with significant gaps between them),
  "total_trunks_detected": integer (count the number of separate large trunk segments - use 1 for a single connected trunk, higher numbers for disconnected segments),
  "crown_shape_quality": "good", "fair", or "poor" (evaluate if the crown shape is well-formed and organized - good for conical/pyramidal coniferous or rounded deciduous, poor for fragmented or irregular shapes),
  "analysis_summary": "brief description of your assessment focusing on tree type, trunk continuity, crown shape, and segmentation quality"
}

Important:
- Only output valid JSON, no additional text
- Color interpretation: Look at the CROWN/TOP of the point cloud for tree type identification
  * GREEN crown = coniferous (evergreen) - may have white/grey base/trunk
  * GREY/WHITE overall = deciduous (leaf-off)
- CRITICAL: Multiple trees are NOT noise - they are valid trees with poor segmentation
- For coniferous trees: Examine the upper crown portion for green coloring and pyramidal shape, even if base is discolored
- For noise detection: look for scattered/random colored points, no clear tree structures, no organized crown shapes
- For multiple trees: look for several distinct crown structures and separate trunks in one segment
- For single tree analysis: evaluate the central trunk for continuity and the crown for good shape
- A well-defined single tree should have a continuous trunk from base to crown and a well-organized crown shape
- Coniferous: look for cone/pyramidal shape (especially in upper crown) and green coloring in foliage area
- Deciduous: look for grey/white skeletal branching structure with organized crown"""

class AiQualityMixin:
    """AI segmentation-quality analysis (screenshots + LLM)."""

    def analyze_tree_with_ai(self):
        """Analyze the currently visualized tree using AI for structure evaluation."""
        if self.current_tree_points is None or len(self.current_tree_points) == 0:
            QMessageBox.warning(self, "Warning", "No tree currently visualized for analysis.")
            return

        try:
            self.log_to_console("Starting AI analysis of tree structure...")

            # Store the points being analyzed for potential noise conversion
            self.last_analyzed_points = self.current_tree_points.copy() if self.current_tree_points is not None else None

            # Generate unique analysis ID
            analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create analysis log directory
            analysis_log_dir = os.path.join(os.path.dirname(__file__), 'analysis_log')
            os.makedirs(analysis_log_dir, exist_ok=True)

            # Create subdirectory for this analysis
            analysis_dir = os.path.join(analysis_log_dir, f"analysis_{analysis_id}")
            os.makedirs(analysis_dir, exist_ok=True)

            self.log_to_console(f"Analysis log will be saved to: {analysis_dir}")

            # Take screenshots from 4 different views
            screenshot_paths = self.take_tree_screenshots()

            if not screenshot_paths:
                QMessageBox.warning(self, "Error", "Failed to capture tree screenshots.")
                return

            # Save screenshots to analysis directory
            saved_screenshot_paths = self.save_analysis_screenshots(screenshot_paths, analysis_dir, analysis_id)

            # Run AI analysis
            ai_result = self.run_ai_analysis(screenshot_paths)

            # Save analysis results
            self.save_analysis_results(analysis_dir, analysis_id, ai_result, saved_screenshot_paths)

            # Display results
            self.display_ai_results(ai_result)
            


            # Clean up temporary screenshots
            self.cleanup_screenshots(screenshot_paths)

        except Exception as e:
            self.log_to_console(f"Error in AI analysis: {str(e)}")
            QMessageBox.critical(self, "Error", f"AI analysis failed: {str(e)}")


    def take_tree_screenshots(self):
        """Take screenshots of the tree from 4 different views."""
        import tempfile
        import os

        if not self.plotter:
            return []

        screenshot_paths = []

        try:
            # Store original camera position
            original_camera = self.plotter.camera.copy()

            # Define camera positions for each view
            views = [
                ('front', [1, 0, 0], [0, 0, 1]),    # Front view
                ('back', [-1, 0, 0], [0, 0, 1]),    # Back view
                ('left', [0, 1, 0], [0, 0, 1]),     # Left view
                ('right', [0, -1, 0], [0, 0, 1])    # Right view
            ]

            for view_name, position, up_vector in views:
                # Set camera position
                self.plotter.camera.position = position
                self.plotter.camera.focal_point = [0, 0, 0]  # Center of tree
                self.plotter.camera.up = up_vector
                self.plotter.reset_camera()

                # Update plotter
                self.plotter.update()

                # Take screenshot
                temp_file = tempfile.NamedTemporaryFile(suffix=f'_{view_name}.png', delete=False)
                screenshot_path = temp_file.name
                temp_file.close()

                # Capture screenshot
                self.plotter.screenshot(screenshot_path)
                screenshot_paths.append(screenshot_path)

                self.log_to_console(f"Captured {view_name} view screenshot")            # Restore original camera position
            self.plotter.camera = original_camera
            self.plotter.reset_camera()
            self.plotter.update()

        except Exception as e:
            self.log_to_console(f"Error taking screenshots: {str(e)}")
            # Clean up any screenshots taken so far
            self.cleanup_screenshots(screenshot_paths)
            return []

        return screenshot_paths


    def run_ai_analysis(self, screenshot_paths):
        """Run AI analysis on the screenshots using local LM Studio or NVIDIA NIM API."""
        import base64
        from PIL import Image
        import io
        import litellm

        def encode_image(image_path, max_size=(896, 896), quality=85):
            """Encode image file to base64 string with optional resizing"""
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for PNG with transparency, etc.)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')

                # Resize if larger than max_size
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Save to bytes buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                buffer.seek(0)

                return base64.b64encode(buffer.getvalue()).decode('utf-8')

        try:
            # Check if using local LM Studio or NVIDIA NIM
            use_local_model = os.getenv("USE_LOCAL_LM_STUDIO", "false").lower() == "true"
            
            if use_local_model:
                # Use local LM Studio model
                lm_studio_api_url = os.getenv("LM_STUDIO_API_URL", "http://localhost:1234/v1")
                model = os.getenv("LM_STUDIO_MODEL", "local-model")  # Name of loaded model in LM Studio
                
                self.log_to_console(f"Using local LM Studio model from {lm_studio_api_url}")
                
                # Set up LiteLLM for local LM Studio (OpenAI-compatible API)
                litellm.api_base = lm_studio_api_url
                litellm.api_key = "not-needed"  # LM Studio typically doesn't require API key
            else:
                # Use NVIDIA NIM cloud API
                model = "nvidia_nim/meta/llama-4-maverick-17b-128e-instruct"
                
                # Set API key from environment or use the one from original script
                api_key = os.getenv("NVIDIA_API_KEY", "")
                if not api_key:
                    self.log_to_console("[WARN] NVIDIA_API_KEY is not set; NVIDIA NIM requests will fail until the key is provided in the environment.")
                os.environ["NVIDIA_NIM_API_KEY"] = api_key
                
                self.log_to_console("Using NVIDIA NIM cloud API")

            # Add analysis prompt
            prompt = AI_ANALYSIS_PROMPT

            # Prepare message content
            content = [{"type": "text", "text": prompt}]
            for image_path in screenshot_paths:
                base64_image = encode_image(image_path, max_size=(896, 896))
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

            messages = [{"role": "user", "content": content}]

            self.log_to_console("Running AI analysis...")

            # Call LiteLLM with appropriate settings
            if use_local_model:
                # For local models, use simpler API call
                response = litellm.completion(
                    model=f"openai/{model}",
                    messages=messages,
                    max_tokens=512,
                    temperature=0.20,
                    top_p=0.70,
                    stream=True
                )
            else:
                # For NVIDIA NIM
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.20,
                    top_p=0.70,
                    frequency_penalty=0.00,
                    presence_penalty=0.00,
                    stream=True  # Enable streaming for real-time output
                )

            # Handle streaming response
            full_response = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content

            return full_response.strip()

        except Exception as e:
            self.log_to_console(f"Error running AI analysis: {str(e)}")
            return f"Failed to run AI analysis: {str(e)}"


    def display_ai_results(self, ai_result):
        """Display the AI analysis results in a dialog."""
        if not ai_result or ai_result.startswith("Failed"):
            QMessageBox.warning(self, "AI Analysis Failed", ai_result)
            return

        # Create result dialog
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("AI Tree Structure Analysis")
        result_dialog.setGeometry(300, 300, 600, 400)
        
        # Set dialog background to match app theme
        result_dialog.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)

        layout = QVBoxLayout(result_dialog)

        # Title
        title_label = QLabel("AI Tree Structure Analysis")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        layout.addWidget(title_label)

        # Tree info
        tree_info = f"Analyzed Tree: {self.current_tree_id or 'Unknown'} ({len(self.current_tree_points) if self.current_tree_points is not None else 0} points)"
        info_label = QLabel(tree_info)
        info_label.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        layout.addWidget(info_label)

        # Try to parse JSON response
        try:
            import json
            json_result = json.loads(ai_result)
            
            # Create structured display
            results_layout = QVBoxLayout()
            
            # Check if this is noise detection or tree analysis
            is_tree = json_result.get('is_tree', 'yes')  # Default to tree analysis for backward compatibility
            
            if is_tree.lower() == 'no':
                # Noise detection result
                appears_noise = json_result.get('appears_to_be_noise', 'unknown')
                noise_color = "#FF4444" if appears_noise.lower() == "yes" else "#4CAF50"
                noise_label = QLabel(f"Appears to be Noise: {appears_noise.upper()}")
                noise_label.setStyleSheet(f"font-weight: bold; color: {noise_color}; font-size: 12px;")
                results_layout.addWidget(noise_label)
                
                # Recommended action
                recommended_action = json_result.get('recommended_action', 'unknown')
                action_label = QLabel(f"Recommended Action: {recommended_action.replace('_', ' ').upper()}")
                action_label.setStyleSheet("font-weight: bold; color: #FF8844; font-size: 12px;")
                results_layout.addWidget(action_label)
                
                # Noise ITC value
                noise_itc = json_result.get('noise_itc_value', 'unknown')
                itc_label = QLabel(f"ITC Value for Noise: {noise_itc}")
                itc_label.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 12px;")
                results_layout.addWidget(itc_label)
                
                # Add "Convert to Noise" button
                convert_button = ModernButton("Convert to Noise Class (ITC 9999)")
                convert_button.clicked.connect(lambda: self.convert_to_noise_class(9999, self.last_analyzed_points))
                results_layout.addWidget(convert_button)
                
            else:
                # Tree analysis result
                # Tree type
                tree_type = json_result.get('tree_type', 'unknown')
                tree_type_color = "#4CAF50" if tree_type.lower() in ['coniferous', 'deciduous'] else "#FF8844"
                tree_type_label = QLabel(f"Tree Type: {tree_type.upper()}")
                tree_type_label.setStyleSheet(f"font-weight: bold; color: {tree_type_color}; font-size: 12px;")
                results_layout.addWidget(tree_type_label)
                
                # Trunk well defined
                trunk_status = json_result.get('trunk_well_defined', 'unknown')
                trunk_color = "#4CAF50" if trunk_status.lower() == "yes" else "#FF4444"
                trunk_label = QLabel(f"Trunk Well Defined: {trunk_status.upper()}")
                trunk_label.setStyleSheet(f"font-weight: bold; color: {trunk_color}; font-size: 12px;")
                results_layout.addWidget(trunk_label)
                
                # Trunk disconnected
                trunk_disconnected = json_result.get('trunk_disconnected', 'unknown')
                disconnected_color = "#FF4444" if trunk_disconnected.lower() == "yes" else "#4CAF50"
                disconnected_label = QLabel(f"Trunk Disconnected: {trunk_disconnected.upper()}")
                disconnected_label.setStyleSheet(f"font-weight: bold; color: {disconnected_color}; font-size: 12px;")
                results_layout.addWidget(disconnected_label)
                
                # Total trunks detected
                trunk_count = json_result.get('total_trunks_detected', 'unknown')
                trunk_count_label = QLabel(f"Total Trunks Detected: {trunk_count}")
                trunk_count_label.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 12px;")
                results_layout.addWidget(trunk_count_label)
            
            # Analysis summary (common to both formats)
            summary_text = json_result.get('analysis_summary', 'No summary provided')
            summary_label = QLabel("Analysis Summary:")
            summary_label.setStyleSheet("font-weight: bold; color: #4CAF50; margin-top: 10px;")
            results_layout.addWidget(summary_label)
            
            summary_display = QTextEdit()
            summary_display.setPlainText(summary_text)
            summary_display.setReadOnly(True)
            summary_display.setMaximumHeight(100)
            summary_display.setStyleSheet("""
                QTextEdit {
                    background-color: #363636;
                    color: #e0e0e0;
                    border: 1px solid #555555;
                    border-radius: 0px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 11px;
                }
            """)
            results_layout.addWidget(summary_display)
            
            # Raw JSON response from LLM
            raw_json_label = QLabel("Raw LLM Response (JSON):")
            raw_json_label.setStyleSheet("font-weight: bold; color: #4CAF50; margin-top: 10px;")
            results_layout.addWidget(raw_json_label)
            
            raw_json_display = QTextEdit()
            raw_json_display.setPlainText(json.dumps(json_result, indent=2))
            raw_json_display.setReadOnly(True)
            raw_json_display.setMaximumHeight(150)
            raw_json_display.setStyleSheet("""
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #00ff00;
                    border: 1px solid #555555;
                    border-radius: 0px;
                    font-family: 'Courier New', monospace;
                    font-size: 10px;
                }
            """)
            results_layout.addWidget(raw_json_display)
            
            layout.addLayout(results_layout)
            
        except json.JSONDecodeError:
            # Fallback to raw text display if JSON parsing fails
            error_label = QLabel("[WARN] AI response was not in expected JSON format. Raw response:")
            error_label.setStyleSheet("color: #FF8844; font-size: 11px;")
            layout.addWidget(error_label)
            
            result_text = QTextEdit()
            result_text.setPlainText(ai_result)
            result_text.setReadOnly(True)
            result_text.setStyleSheet("""
                QTextEdit {
                    background-color: #363636;
                    color: #e0e0e0;
                    border: 1px solid #555555;
                    border-radius: 0px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 11px;
                    selection-background-color: #4CAF50;
                }
            """)
            layout.addWidget(result_text)

        # Close button
        close_button = ModernButton("Close")
        close_button.clicked.connect(result_dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)

        result_dialog.exec()






    def cleanup_screenshots(self, screenshot_paths):
        """Clean up temporary screenshot files."""
        for path in screenshot_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                self.log_to_console(f"Warning: Failed to clean up {path}: {str(e)}")


    def save_analysis_screenshots(self, temp_screenshot_paths, analysis_dir, analysis_id):
        """Save screenshots to the analysis directory with descriptive names."""
        import shutil

        saved_paths = []
        view_names = ['front', 'back', 'left', 'right']

        try:
            for i, temp_path in enumerate(temp_screenshot_paths):
                view_name = view_names[i] if i < len(view_names) else f'view_{i}'
                filename = f"{analysis_id}_{view_name}.png"
                dest_path = os.path.join(analysis_dir, filename)

                # Copy the temporary screenshot to the analysis directory
                shutil.copy2(temp_path, dest_path)
                saved_paths.append(dest_path)

                self.log_to_console(f"Saved {view_name} screenshot: {filename}")

            return saved_paths

        except Exception as e:
            self.log_to_console(f"Error saving analysis screenshots: {str(e)}")
            return []


    def save_analysis_results(self, analysis_dir, analysis_id, ai_result, screenshot_paths):
        """Save analysis results, prompt, and metadata to the analysis directory."""
        try:
            # Save AI prompt
            prompt = AI_ANALYSIS_PROMPT

            prompt_file = os.path.join(analysis_dir, f"{analysis_id}_prompt.txt")
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            self.log_to_console(f"Saved analysis prompt: {os.path.basename(prompt_file)}")

            # Save AI response
            response_file = os.path.join(analysis_dir, f"{analysis_id}_response.txt")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(ai_result)
            self.log_to_console(f"Saved AI response: {os.path.basename(response_file)}")

            # Save metadata
            metadata_file = os.path.join(analysis_dir, f"{analysis_id}_metadata.txt")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Analysis ID: {analysis_id}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Tree ID: {self.current_tree_id or 'Unknown'}\n")
                f.write(f"Number of Points: {len(self.current_tree_points) if self.current_tree_points is not None else 0}\n")
                f.write(f"Screenshots: {len(screenshot_paths)}\n")
                for i, path in enumerate(screenshot_paths):
                    f.write(f"  - {os.path.basename(path)}\n")
                f.write(f"Analysis Success: {'Yes' if not ai_result.startswith('Failed') else 'No'}\n")

            self.log_to_console(f"Saved analysis metadata: {os.path.basename(metadata_file)}")

        except Exception as e:
            self.log_to_console(f"Error saving analysis results: {str(e)}")


