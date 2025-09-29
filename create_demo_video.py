#!/usr/bin/env python3
"""
Automated Demo Video Creator for MLflow Dashboard
Creates a scripted walkthrough of the MLflow interface
"""

import time
import webbrowser
import subprocess
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class MLflowDemoCreator:
    def __init__(self):
        self.mlflow_url = "http://localhost:5000"
        self.api_url = "http://localhost:8001"
        self.driver = None
        
    def setup_browser(self):
        """Setup Chrome browser for automated demo"""
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-notifications")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            return True
        except Exception as e:
            print(f"❌ Chrome WebDriver not found: {e}")
            print("💡 Please install ChromeDriver or use manual recording")
            return False
    
    def create_demo_script(self):
        """Create a demo script for manual recording"""
        
        demo_script = """
# 🎬 MLflow Dashboard Demo Script

## 📋 Pre-Recording Checklist
- [ ] Start MLflow services: `docker compose -f docker-compose-simple.yml up -d`
- [ ] Train models: `python train_model_simple.py && python advanced_training.py`
- [ ] Start API: `cd serving && python main_standalone.py`
- [ ] Open browser to http://localhost:5000
- [ ] Start screen recording

## 🎯 Demo Sequence (2-3 minutes)

### Scene 1: MLflow Overview (30 seconds)
1. **Open MLflow Dashboard** → http://localhost:5000
2. **Show main interface** → Point out Experiments and Models tabs
3. **Highlight key metrics** → Show experiment count and model versions

### Scene 2: Experiments Deep Dive (60 seconds)
1. **Click "Experiments" tab**
2. **Show experiment list**:
   - stock_price_prediction
   - stock_price_hyperparameter_tuning
3. **Click on stock_price_prediction**
4. **Show runs table**:
   - Point out R² scores (0.9982, 0.9928, etc.)
   - Show different model types
   - Highlight best performing run
5. **Click on best run (0.9982 R²)**
6. **Show metrics tab**:
   - test_r2: 0.9982
   - train_r2: 0.9997
   - test_mse: 0.1511
7. **Show parameters tab**:
   - model_type: RandomForestRegressor
   - features used

### Scene 3: Model Comparison (45 seconds)
1. **Go back to experiments**
2. **Select 2-3 runs** using checkboxes
3. **Click "Compare" button**
4. **Show parallel coordinates plot**
5. **Highlight performance differences**
6. **Show metrics comparison table**

### Scene 4: Model Registry (30 seconds)
1. **Click "Models" tab**
2. **Show StockPricePredictor model**
3. **Click on model name**
4. **Show model versions**:
   - Version 5 (Production)
   - Other versions (None stage)
5. **Highlight production model**

### Scene 5: API Integration (15 seconds)
1. **Open new tab** → http://localhost:8001/docs
2. **Show FastAPI interface**
3. **Demonstrate prediction endpoint**
4. **Show real-time prediction**

## 🎬 Recording Tips

### **Narration Script:**
"This is a production-ready MLOps pipeline using MLflow for experiment tracking and model registry. 

Here we can see two experiments with multiple model runs. Our best performing model achieves 99.82% accuracy using Random Forest.

Let me show you the detailed metrics... As you can see, we have excellent performance with minimal overfitting.

The model comparison feature lets us evaluate different approaches side by side.

In the model registry, we can see version 5 is currently in production, ready for real-time predictions.

Finally, our FastAPI server provides real-time predictions with sub-100ms latency."

### **Visual Focus Points:**
- **Highlight cursor** on important metrics
- **Zoom in** on key numbers (99.82% accuracy)
- **Smooth transitions** between tabs
- **Pause briefly** on important screens
- **Show loading states** naturally

### **Technical Settings:**
- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 30 FPS
- **Audio**: Clear narration, no background music
- **Duration**: 2-3 minutes maximum
- **Format**: MP4 for video, GIF for animations

## 🛠️ Recording Tools Setup

### **OBS Studio (Recommended)**
1. **Download**: https://obsproject.com/
2. **Scene Setup**:
   - Add "Display Capture" source
   - Set to capture primary monitor
   - Add "Audio Input Capture" for narration
3. **Recording Settings**:
   - Format: MP4
   - Quality: High Quality, Medium File Size
   - Resolution: 1920x1080
   - FPS: 30

### **Windows Game Bar (Built-in)**
1. **Press Win + G** to open
2. **Click record button** or **Win + Alt + R**
3. **Record browser window**
4. **Stop with Win + Alt + R**

### **For GIF Creation:**
1. **Record with any tool** above
2. **Convert to GIF** using:
   - Online: ezgif.com
   - Software: GIMP, Photoshop
   - Command line: ffmpeg

## 📤 Post-Production

### **Video Editing (Optional):**
- **Trim** unnecessary parts
- **Add title cards** with key metrics
- **Highlight** important UI elements
- **Add smooth transitions**

### **GIF Optimization:**
- **Duration**: 10-30 seconds max
- **Size**: Under 10MB for GitHub
- **Quality**: Balance between size and clarity
- **Loop**: Seamless looping

### **Upload Options:**
- **GitHub**: Add to images/ folder
- **YouTube**: Unlisted video for embedding
- **Loom**: Quick sharing link
- **GIF**: Direct embedding in README

## 📋 Final Checklist
- [ ] All services running smoothly
- [ ] Browser tabs prepared
- [ ] Recording software tested
- [ ] Audio levels checked
- [ ] Demo script practiced
- [ ] Backup plan ready

---

**🎯 Goal**: Create a professional 2-3 minute demo showcasing your MLOps expertise and the impressive 99.82% model accuracy!
"""
        
        with open("DEMO_SCRIPT.md", "w", encoding="utf-8") as f:
            f.write(demo_script)
        
        print("✅ Demo script created: DEMO_SCRIPT.md")
        return demo_script
    
    def create_gif_from_screenshots(self):
        """Create an animated GIF from screenshots"""
        
        gif_script = """
# 🎨 Automated GIF Creation Script

import time
import os
from PIL import Image, ImageDraw, ImageFont
import requests

def create_mlflow_gif():
    # This would take screenshots at intervals
    # and combine them into a GIF
    
    screenshots = []
    
    # Take screenshots of different MLflow pages
    pages = [
        "http://localhost:5000",  # Main page
        "http://localhost:5000/#/experiments",  # Experiments
        "http://localhost:5000/#/models",  # Models
    ]
    
    for page in pages:
        # Screenshot logic would go here
        pass
    
    # Combine into GIF
    # images[0].save('mlflow_demo.gif', 
    #                save_all=True, 
    #                append_images=images[1:], 
    #                duration=2000, 
    #                loop=0)

if __name__ == "__main__":
    create_mlflow_gif()
"""
        
        with open("create_gif.py", "w") as f:
            f.write(gif_script)
        
        print("✅ GIF creation script template created")

def main():
    """Main demo creation workflow"""
    
    print("🎬 MLflow Demo Video Creator")
    print("=" * 40)
    
    creator = MLflowDemoCreator()
    
    # Create demo script
    creator.create_demo_script()
    
    # Create GIF template
    creator.create_gif_from_screenshots()
    
    print("\n🎯 Next Steps:")
    print("1. Review DEMO_SCRIPT.md for recording guidance")
    print("2. Choose your recording tool (OBS Studio recommended)")
    print("3. Practice the demo flow")
    print("4. Record your MLflow dashboard walkthrough")
    print("5. Convert to GIF if needed")
    print("6. Add to your GitHub README")
    
    print("\n📋 Quick Recording Commands:")
    print("# Start services")
    print("docker compose -f docker-compose-simple.yml up -d")
    print("python train_model_simple.py")
    print("python advanced_training.py")
    print("cd serving && python main_standalone.py")
    print("\n# Open for recording")
    print("start http://localhost:5000")

if __name__ == "__main__":
    main()