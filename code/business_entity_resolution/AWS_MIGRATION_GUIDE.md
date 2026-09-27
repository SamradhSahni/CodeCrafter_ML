# AWS Cloud Migration & Execution Guide

This guide provides step-by-step instructions to run the **Business Entity Resolution GPU Pipeline** on an AWS cloud instance with high-end hardware, eliminating all laptop RAM limits, memory crashes, and thermal throttling.

---

## 1. Recommended AWS Hardware & Cost Breakdown

| Instance Type | vCPUs | System RAM | GPU | GPU VRAM | On-Demand Cost | Spot Cost | Est. Pipeline Time | Est. Total Run Cost |
|---|---|---|---|---|---|---|---|---|
| **`g5.4xlarge`** *(Recommended)* | **16** | **64 GB** | **NVIDIA A10G** | **24 GB** | ~$1.62 / hr | **~$0.55 / hr** | **~35 - 45 min** | **~$0.50 - $1.10 USD** (~₹50 - ₹90 INR) |
| **`g5.2xlarge`** | 8 | 32 GB | NVIDIA A10G | 24 GB | ~$1.21 / hr | **~$0.40 / hr** | ~45 - 60 min | **~$0.40 - $0.80 USD** (~₹40 - ₹70 INR) |
| **`r6i.4xlarge`** *(High-RAM CPU)* | 16 | 128 GB | None | - | ~$1.00 / hr | **~$0.30 / hr** | ~60 - 75 min | **~$0.35 - $0.70 USD** |

> **Cost Summary**: One full end-to-end execution of the entire 12.5M record dataset on a spot `g5.4xlarge` costs **under $1.00 USD (less than ₹100 INR)**.

---

## 2. Step-by-Step AWS Setup Walkthrough

### Step 1: Launch the EC2 Instance (AWS Console)
1. Go to **AWS Console** -> **EC2** -> **Launch Instances**.
2. **Name**: `entity-resolution-worker`
3. **Application and OS Images (AMI)**:
   - Select **Ubuntu 22.04 LTS** (or search in Community AMIs for: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)`).
4. **Instance Type**:
   - Choose **`g5.4xlarge`** (or `g5.2xlarge`).
5. **Key pair (login)**:
   - Select an existing key pair or click **Create new key pair** (e.g. `aws-key.pem`).
6. **Network settings**:
   - Check **Allow SSH traffic from Anywhere** (or your IP).
7. **Configure Storage**:
   - Change root volume size from 8 GB to **150 GB** (`gp3` volume type).
8. **Advanced details (To Save 65% on Cost)**:
   - Scroll to **Purchasing option** -> check **Spot instances**.
9. Click **Launch Instance**.

---

### Step 2: Connect to Your Instance via SSH

Open PowerShell or Terminal on your local machine:

```bash
# Set permissions on your key (if on Linux/Mac)
chmod 400 path/to/aws-key.pem

# SSH into the instance (replace with your EC2 Public IP / DNS)
ssh -i "path/to/aws-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
```

---

### Step 3: Clone Code & Transfer Data

On the EC2 instance:

```bash
# Clone the repository
git clone https://github.com/SamradhSahni/CodeCrafter_ML.git
cd CodeCrafter_ML/code/business_entity_resolution
```

#### Transferring Dataset to EC2:

**Option A (Using `scp` from your local machine)**:
Open a **new** PowerShell terminal on your local machine:

```powershell
# From e:\projects\Amazon_ML_Challenge:
scp -i "path\to\aws-key.pem" -r student_resource ubuntu@<YOUR_EC2_PUBLIC_IP>:~/CodeCrafter_ML/
```

*(Optional - Transfer pre-computed embeddings to skip Phase 2 and save 20 minutes)*:
```powershell
scp -i "path\to\aws-key.pem" -r cache ubuntu@<YOUR_EC2_PUBLIC_IP>:~/CodeCrafter_ML/
```

**Option B (Direct S3 Sync if uploaded to S3)**:
```bash
# On EC2:
aws s3 sync s3://your-bucket-name/student_resource/ ~/CodeCrafter_ML/student_resource/
```

---

### Step 4: Run Automated Setup Script

On the EC2 instance terminal:

```bash
cd ~/CodeCrafter_ML/code/business_entity_resolution
chmod +x setup_aws.sh run_aws.sh
./setup_aws.sh
```

This automated script will:
- Install all Ubuntu system packages and C++ compilers.
- Configure Python virtualenv.
- Install PyTorch with CUDA 12 and `faiss-gpu`.
- Install all project dependencies (`lightgbm`, `rapidfuzz`, `sentence-transformers`, `sparse_dot_topn`).
- Run a self-check confirming GPU, 24GB VRAM, and multi-core CPU availability.

---

### Step 5: Run the Full Pipeline

Use `tmux` so that if your internet or SSH connection disconnects, the pipeline continues running uninterrupted in the background:

```bash
# Start a tmux session
tmux new -s pipeline

# Activate virtual environment
source venv/bin/activate

# Launch the automated runner (logs both to screen and to a timestamped file)
./run_aws.sh
```

*(Tip: You can detach from tmux anytime using `Ctrl+B` then `D`. Reattach anytime using `tmux attach -t pipeline`)*.

---

### Step 6: Download Output and Terminate Instance

When the pipeline finishes:
1. The final submission files will be saved in `~/CodeCrafter_ML/output/`.
2. From your local PC PowerShell, download the output:

```powershell
scp -i "path\to\aws-key.pem" -r ubuntu@<YOUR_EC2_PUBLIC_IP>:~/CodeCrafter_ML/output e:\projects\Amazon_ML_Challenge\
```

3. **IMPORTANT**: In the AWS Console, click **Instance State** -> **Terminate Instance** (or **Stop Instance**) to ensure no further charges are incurred.
