# 🎬 MLflow Dashboard Demo Recording Guide

## 🎯 Quick Start Options

### **Option 1: Simple Screen Recording (Recommended)**

#### **Windows Built-in (Game Bar)**
1. **Press `Win + G`** to open Game Bar
2. **Click the record button** (red circle)
3. **Record your MLflow demo**
4. **Press `Win + Alt + R`** to stop
5. **Find video** in `C:\Users\[Username]\Videos\Captures\`

#### **Free Tools:**
- **OBS Studio**: https://obsproject.com/ (Professional)
- **ShareX**: https://getsharex.com/ (Lightweight)
- **LICEcap**: https://www.cockos.com/licecap/ (GIF only)

### **Option 2: Online GIF Makers**
- **Loom**: https://loom.com (Record + auto-convert to GIF)
- **CloudApp**: https://cloudapp.com (Easy sharing)
- **Screencastify**: Chrome extension

## 📋 Demo Script (2-3 minutes)

### **🎬 Scene 1: MLflow Overview (30 seconds)**
1. **Open** http://localhost:5000
2. **Show main dashboard** with experiments
3. **Highlight** "2 experiments, 5 runs"
4. **Point out** Experiments and Models tabs

**Narration**: *"This is our MLflow dashboard showing 2 experiments with 5 model runs, achieving up to 99.82% accuracy."*

### **🎬 Scene 2: Experiments Deep Dive (60 seconds)**
1. **Click "Experiments" tab**
2. **Show experiments list**:
   - stock_price_prediction
   - stock_price_hyperparameter_tuning
3. **Click on stock_price_prediction**
4. **Show runs table** with R² scores
5. **Click on best run** (0.9982 R²)
6. **Show Metrics tab**:
   - test_r2: 0.9982
   - train_r2: 0.9997
   - test_mse: 0.1511
7. **Show Parameters tab**:
   - model_type: RandomForestRegressor

**Narration**: *"Here we can see our best model achieves 99.82% accuracy with minimal overfitting. The Random Forest algorithm performs exceptionally well."*

### **🎬 Scene 3: Model Comparison (45 seconds)**
1. **Go back to experiments**
2. **Select 2-3 runs** using checkboxes
3. **Click "Compare" button**
4. **Show parallel coordinates plot**
5. **Highlight performance differences**
6. **Show metrics table**

**Narration**: *"The comparison view lets us evaluate different models side-by-side. You can see the Random Forest clearly outperforms other approaches."*

### **🎬 Scene 4: Model Registry (30 seconds)**
1. **Click "Models" tab**
2. **Show StockPricePredictor model**
3. **Click on model name**
4. **Show versions**:
   - Version 5 (Production)
   - Other versions
5. **Highlight production status**

**Narration**: *"In the model registry, Version 5 is currently deployed to production, ready for real-time predictions."*

### **🎬 Scene 5: API Demo (15 seconds)**
1. **Open new tab**: http://localhost:8001/docs
2. **Show FastAPI interface**
3. **Make a quick prediction**
4. **Show response time**

**Narration**: *"Our FastAPI server provides real-time predictions with sub-100ms latency."*

## 🛠️ Recording Setup

### **Pre-Recording Checklist:**
```bash
# 1. Start all services
docker compose -f docker-compose-simple.yml up -d

# 2. Train models
python train_model_simple.py
python advanced_training.py

# 3. Start API
cd serving && python main_standalone.py

# 4. Verify everything works
curl http://localhost:5000
curl http://localhost:8001/health
```

### **Browser Setup:**
1. **Open Chrome/Edge** in full screen
2. **Prepare tabs**:
   - Tab 1: http://localhost:5000
   - Tab 2: http://localhost:8001/docs
3. **Clear browser** of distractions
4. **Set zoom to 100%**

### **Recording Settings:**
- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 30 FPS
- **Duration**: 2-3 minutes max
- **Audio**: Optional narration
- **Format**: MP4 (convert to GIF later)

## 🎨 Creating GIFs

### **Method 1: Online Conversion**
1. **Record MP4** using any tool above
2. **Go to**: https://ezgif.com/video-to-gif
3. **Upload your video**
4. **Set options**:
   - Duration: 10-30 seconds
   - Size: Under 10MB
   - Quality: Medium-High
5. **Download GIF**

### **Method 2: OBS + GIF Export**
1. **Record with OBS Studio**
2. **Export as MP4**
3. **Use online converter** or software

### **Method 3: Direct GIF Recording**
1. **Use LICEcap** (Windows/Mac)
2. **Record directly as GIF**
3. **Optimize size** for GitHub

## 📤 Adding to GitHub

### **For Video (MP4):**
```markdown
## 🎬 MLflow Dashboard Demo

[![MLflow Demo](images/mlflow-demo-thumbnail.png)](https://youtu.be/your-video-id)
*Click to watch the full MLflow dashboard walkthrough*
```

### **For GIF:**
```markdown
## 🎬 MLflow Dashboard Demo

![MLflow Demo](images/mlflow-demo.gif)
*Interactive MLflow dashboard showing 99.82% model accuracy*
```

### **For Multiple GIFs:**
```markdown
## 🎬 MLflow Dashboard Walkthrough

### Experiments Overview
![MLflow Experiments](images/mlflow-experiments.gif)

### Model Comparison
![Model Comparison](images/mlflow-comparison.gif)

### Model Registry
![Model Registry](images/mlflow-registry.gif)
```

## 🎯 Pro Tips

### **Recording Tips:**
- **Practice first** - do a dry run
- **Slow down** - move cursor deliberately
- **Highlight** important metrics with cursor
- **Pause briefly** on key screens
- **Keep it smooth** - avoid jerky movements

### **GIF Optimization:**
- **Keep under 10MB** for GitHub
- **Focus on key actions** only
- **Use smooth transitions**
- **Optimize colors** to reduce file size
- **Test loading speed**

### **Professional Touch:**
- **Add title cards** with key metrics
- **Highlight cursor** for visibility
- **Use consistent timing**
- **Show loading states** naturally
- **End with impressive results**

## 📋 Quick Commands

```bash
# Start everything for recording
docker compose -f docker-compose-simple.yml up -d
python train_model_simple.py && python advanced_training.py
cd serving && python main_standalone.py

# Open for recording
start http://localhost:5000
start http://localhost:8001/docs

# Verify services
curl http://localhost:5000
curl http://localhost:8001/health
```

## 🎉 Expected Results

After following this guide, you'll have:
- **Professional demo video/GIF** of your MLflow dashboard
- **Showcase of 99.82% model accuracy**
- **Evidence of MLOps expertise**
- **Impressive GitHub portfolio addition**

**🚀 This will make your project stand out to employers and demonstrate real-world MLOps skills!**