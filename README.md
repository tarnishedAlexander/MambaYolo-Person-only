# MambaYolo-Person-Only

Welcome to **MambaYolo-Person-Only**, a specialized project for real-time person detection using the efficient Mamba-YOLO model. This repository provides a streamlined setup to run the model within a Docker container, ensuring a consistent environment with all dependencies pre-installed.

## Overview

This project leverages the Mamba-YOLO architecture, optimized for detecting people in various scenarios such as crowd monitoring, security systems, or pedestrian tracking. By using Docker, we ensure portability and ease of setup across different systems.

## Prerequisites

- **Docker** and **Docker Compose** installed on your system.
- A compatible GPU (recommended for optimal performance with YOLO models).
- Input data (e.g., video or images) for person detection, properly configured for the script.
- note: Assumption, dokcer only works for NVIDIA graphic cards 

## Setup and Running the Project

Follow these steps to set up and run the project:

1. **Build the Docker Image**
   Navigate to the `docker` directory and build the Docker image:

   ```bash
   cd docker
   docker compose build
   
2. **Start the Container** Launch the Docker container in detached mode:

   ```bash
   docker compose up -d

4. **Access the Container** Enter the container's interactive shell to execute commands:

   ```bash
   docker compose exec -it yoloCounter /bin/bash

4.Download prerequisites

   ```bash
   conda create -n mambayolo -y python=3.11 && conda activate mambayolo
   ```

5. Install Dependencies

   ```bash
    pip3 install torch===2.3.0 torchvision torchaudio && pip install seaborn thop timm einops && cd selective_scan && pip install . && cd .. && pip install -v -e .
    ```

6.   **Run the Detection Script** Inside the container, navigate to the Mamba-YOLO directory and run the main script to start person detection:

   ```bash
    cd Mamba-YOLO/
    python yolosComparition.py
    ```
