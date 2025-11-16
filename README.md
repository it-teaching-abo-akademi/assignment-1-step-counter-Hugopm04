# Step Detection Using Accelerometer Data  
*A lightweight signal-processing approach for detecting steps with orientation-independent features.*

## Overview
This project explores two step-detection methods based purely on accelerometer data:

1. **Static Threshold Method**  
   Detects steps using fixed thresholds over the acceleration **magnitude** and **characteristic angle**.

2. **Dynamic Threshold Method**  
   Adapts thresholds based on recent signal behavior, achieving higher accuracy during normal and slow walking.

Both methods are designed to be **orientation-agnostic**, making them suitable for phones carried in any position.

The system was tested on multiple datasets containing walking, slow walking, fast walking, jumping, and "crazy movement" patterns.  
It is intentionally lightweight, relying only on fundamental signal operations—no machine learning required.

---

## Features
- **Hill/Valley Detection** on acceleration magnitude  
- **Angle Feature Extraction** from 3-axis accelerations  
- **Step Classification** through static & dynamic thresholds  
- **Performance Evaluation** with accuracy scoring  
- **Visualization Utilities** for debugging and analysis  
- **Phone-orientation independent**

---

## Why This Approach?
Many step counters rely only on vertical acceleration, making them easy to fool with simple up-and-down movements.  
This project focuses on a definition of *steps as full-body forward locomotion*, using:

- **Acceleration magnitude hills**  
- **Angle changes between direction vectors**  

This avoids false positives from simple vertical bouncing.

---

## Results Summary
Both methods achieved solid performance:

- **Static threshold:** ~75% accuracy on mixed movements  
- **Dynamic threshold:** ~94% accuracy on normal/slow walking  
- Resistant to false positives from pure vertical jumps  
- Simple, explainable, and computationally cheap

You can find the full discussion and visual results in the Assignment1_Hugo_Pérez document.

---

## How to Run It
```bash
# Clone the repository
git clone https://github.com/it-teaching-abo-akademi/assignment-1-step-counter-Hugopm04.git

# Run analysis
python main.py

