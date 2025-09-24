# 🎬 Video Demo Commands - Copy & Paste Ready

## 📋 **Commands to Run During Video Recording**

### **Section 1: System Overview**
```bash
# Show project structure
ls -la

# Show running containers
docker ps

# Show system health
curl http://localhost:8000/
```

### **Section 2: Model Training Demo**
```bash
# Train new model with hyperparameter tuning
python advanced_training.py

# While running, explain the MLflow tracking happening in background
```

### **Section 3: MLflow Dashboard**
```
# Open in browser: http://localhost:5000
# Navigate through:
# - Experiments tab
# - Models tab  
# - Compare runs
# - Show metrics and parameters
```

### **Section 4: API Testing**
```bash
# Test API health
curl http://localhost:8000/

# Get model information
curl http://localhost:8000/model/info

# Test different market scenarios
echo "Normal Market:"
curl -X POST "http://localhost:8000/predict?ma_3=150.0&pct_change_1d=0.01&volume=1000000"

echo "Bull Market:"
curl -X POST "http://localhost:8000/predict?ma_3=200.0&pct_change_1d=0.05&volume=2000000"

echo "Bear Market:"
curl -X POST "http://localhost:8000/predict?ma_3=100.0&pct_change_1d=-0.03&volume=500000"

echo "High Volatility:"
curl -X POST "http://localhost:8000/predict?ma_3=175.0&pct_change_1d=0.08&volume=5000000"
```

### **Section 5: Comprehensive Testing**
```bash
# Run full test suite
python test_pipeline.py

# Show monitoring
python model_monitoring.py
```

### **Section 6: Production Configuration**
```bash
# Show deployment config
cat deployment_config.json

# Show production Docker setup
head -20 docker-compose.prod.yml

# Show Nginx load balancer config
head -15 nginx.conf
```

---

## 🎯 **Browser Tabs to Have Open**

1. **MLflow Dashboard**: http://localhost:5000
2. **FastAPI Docs**: http://localhost:8000/docs
3. **API Health**: http://localhost:8000/
4. **VS Code**: With project open
5. **Terminal**: Split pane for commands

---

## 📊 **Key Metrics to Mention**

- **99.82% Model Accuracy** (R² score)
- **16ms Average Response Time** (from latest test)
- **100% Test Success Rate**
- **7 Model Versions** in registry
- **4 Microservices** running
- **Production Ready** with load balancing

---

## 🎬 **Recording Checklist**

### Before Recording:
- [ ] All Docker containers running
- [ ] Test suite passes 100%
- [ ] Browser tabs prepared
- [ ] Audio/video quality tested
- [ ] Script reviewed
- [ ] Environment clean (close unnecessary apps)

### During Recording:
- [ ] Speak clearly and confidently
- [ ] Explain before executing commands
- [ ] Highlight key metrics and results
- [ ] Show enthusiasm for the technology
- [ ] Keep good pacing (not too fast/slow)

### After Recording:
- [ ] Review for audio/video quality
- [ ] Check all key points covered
- [ ] Add captions if needed
- [ ] Upload with proper title/description
- [ ] Share link in applications/portfolio

---

## 💡 **Pro Tips for Great Demo**

1. **Start Strong**: "This is a production-ready MLOps system..."
2. **Show Results First**: Lead with metrics and achievements
3. **Explain the Why**: Connect technical features to business value
4. **Be Confident**: You built something impressive!
5. **End with Impact**: How this solves real problems

Your system is performing excellently - 16ms response time and 100% test success rate! This will make for a compelling demo. 🚀