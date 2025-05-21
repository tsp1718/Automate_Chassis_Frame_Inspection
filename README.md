# Automate Chassis Frame Inspection

This project aims to detect mountings and identify missing ones on ladder-type chassis frames, automating the quality inspection process in an assembly factory located in Indore.

📄 For a detailed understanding of the system, refer to the full report:
📘 [Report.pdf](https://github.com/tsp1718/Automate_Chassis_Frame_Inspection/blob/main/Report.pdf)

## 📁 Repository Structure

```bash Automate_Chassis_Frame_Inspection/
├── App/
│   ├── main.py                # Streamlit app
│   ├── requirements.txt       # App dependencies
│   ├── Nano.pt                # Trained YOLOv8n model
│   ├── Test images/           # Images for testing
│   └── reference images/      # Ground truth/reference images
├── Models/
│   ├── YOLOv8x/
│   │   ├── graphs & metrics           # Training graphs & Evaluation metrics
│   │   ├── training_log.txt
│   │   └── model_kaggle_link.txt
│   ├── YOLOv8s/
│   │   ├── graphs & metrics  # Training graphs & Evaluation metrics
│   │   ├── training_log.txt
│   │   └── model_kaggle_link  # [Add link here]
│   └── YOLOv8n/
│   │   ├── graphs & metrics  # Training graphs & Evaluation metrics
│       ├── training_log.txt
│       └── model_kaggle_link  # [Add link here]
├── training.ipynb             # Notebook to train the models
└── Report.pdf                 # Complete project report
 ```


## 🧠 Model Training

All models were trained on Kaggle using NVIDIA T4 GPUs.
Variants trained: YOLOv8n, YOLOv8s, and YOLOv8x
Evaluation includes training graphs, performance metrics, and logs.

## ⚙️ Setup Instructions

Follow the steps below to run the inspection system locally:

1.  **Clone the repository**

    ```bash
    git clone [https://github.com/tsp1718/Automate_Chassis_Frame_Inspection.git](https://github.com/tsp1718/Automate_Chassis_Frame_Inspection.git)
    cd Automate_Chassis_Frame_Inspection/App
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app**

    ```bash
    streamlit run main.py
    ```

4.  **Test the model**
    Upload images from the `Test images/` folder when prompted.

## ✅ Project Status

We have successfully implemented the following features:

* ✅ Classification of mountings
* ✅ Detection of missing components
* ⚠️ Orientation and position detection — in progress

The current system proves the feasibility of using YOLO-based computer vision for automated quality checks on complex mechanical assemblies like chassis frames.
