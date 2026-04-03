import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import glob
import os
import numpy as np

# Adjust import to run as script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.khaos_kan.data_processor import process_multi_timeframe
from src.khaos_kan.data_loader import create_rolling_datasets
from src.khaos_kan.physics import PhysicsLayer
from src.khaos_kan.kan import KHAOS_KAN
from src.khaos_kan.loss import PhysicsLoss

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 1. Prepare Data (Multi-Timeframe)
    print("Preparing Multi-Timeframe Datasets...")
    train_data_dir = os.path.join(args.data_dir, 'training_ready')
    os.makedirs(train_data_dir, exist_ok=True)
    
    # Check if we already have files in training_ready
    ready_files = glob.glob(os.path.join(train_data_dir, '*.csv'))
    if len(ready_files) > 0:
        print(f"Found {len(ready_files)} existing files in training_ready. Using them.")
        processed_files = ready_files
    else:
        # Process raw files
        processed_files = []
        search_pattern = os.path.join(args.data_dir, '**', '*.csv')
        raw_files = glob.glob(search_pattern, recursive=True)
        print(f"Found {len(raw_files)} raw files.")
        
        for f in raw_files:
            if 'training_ready' in f: continue
            if os.path.getsize(f) < 1024: continue
            
            new_files = process_multi_timeframe(f, train_data_dir)
            processed_files.extend(new_files)
            
    # Filter only target timeframes
    target_suffixes = ['_5m.csv', '_15m.csv', '_1h.csv', '_4h.csv', '_1d.csv']
    final_files = [f for f in processed_files if any(f.endswith(s) for s in target_suffixes)]
    
    # --- DATASET PRIORITIZATION ---
    # User Request: Reduce Forex usage, prioritize Crypto, Index, Commodity.
    # We will filter final_files based on asset class keywords.
    
    priority_assets = [
        'BTC', 'ETH', # Crypto Options
        'SPX', 'NDX', 'DJI', 'UDX', # Index Options
        'XAU', 'WTI', 'GC', 'CL'    # Commodity Options
    ]
    
    # Allow only minimal Forex (e.g. EURUSD only) if needed, but User stressed Options.
    # EURUSD has options too.
    allowed_forex = ['EURUSD']
    
    filtered_files = []
    for f in final_files:
        filename = os.path.basename(f).upper()
        
        # Check priority
        is_priority = any(asset in filename for asset in priority_assets)
        
        # Check allowed forex
        is_allowed_forex = any(fx in filename for fx in allowed_forex)
        
        # Exclude other forex implies: if it looks like forex (contains "USD" but not in priority) 
        # and not in allowed_forex, skip it? 
        # Safer: Just select what we want.
        
        if is_priority or is_allowed_forex:
            filtered_files.append(f)
            
    final_files = filtered_files
    # -----------------------------
    
    # Remove duplicates
    final_files = list(set(final_files))
    
    if not final_files:
        print("No valid multi-timeframe data found. Please ensure data fetcher fetched high-res data.")
        # Fallback to existing files if resampling failed
        # Apply same filter to raw files
        raw_candidates = [f for f in raw_files if os.path.getsize(f) > 1024]
        final_files = []
        for f in raw_candidates:
            name = os.path.basename(f).upper()
            if any(a in name for a in priority_assets) or any(fx in name for fx in allowed_forex):
                final_files.append(f)

    print(f"Training on {len(final_files)} files (Optimized Asset Mix)")
    print(f"Assets included: {[os.path.basename(f) for f in final_files[:5]]} ...")
    
    # Model Init (Once)
    physics = PhysicsLayer().to(device)
    
    # Input Dim: 16 steps * 14 features = 224
    # Physics(5) + Price(5) + SMC(4) = 14
    input_dim = 16 * 14
    
    # Checkpoint logic
    # If checkpoint exists, we load it. BUT we must ensure dimensions match.
    # If dimension mismatch, we should start fresh or partial load.
    
    kan = KHAOS_KAN(
        input_dim=input_dim, 
        hidden_dim=64,
        output_dim=1,
        layers=3, 
        grid_size=args.grid_size
    ).to(device)
    
    optimizer = optim.AdamW(kan.parameters(), lr=args.lr, weight_decay=1e-4)
    
    start_epoch = 0
    
    # Load checkpoint if available
    checkpoint_path = os.path.join(args.save_dir, 'khaos_kan_model_final.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            # Safe load: Allow argparse.Namespace
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Check shape compatibility
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Check input norm running mean
                if 'layers.0.input_norm.running_mean' in state_dict:
                    chk_dim = state_dict['layers.0.input_norm.running_mean'].shape[0]
                    if chk_dim != input_dim:
                        print(f"Dimension mismatch: Checkpoint {chk_dim} vs Current {input_dim}. Starting fresh.")
                    else:
                        kan.load_state_dict(state_dict)
                        print("Loaded successfully.")
                else:
                    # Try partial load? Or strict load?
                    # Strict load might fail if shapes mismatch elsewhere
                    try:
                        kan.load_state_dict(state_dict)
                        print("Loaded successfully (strict).")
                    except RuntimeError as e:
                         print(f"Shape mismatch in layers: {e}. Starting fresh.")
            else:
                 print("Checkpoint missing model_state_dict. Starting fresh.")
                 
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting fresh.")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    # Aggressive Physics Weights: 50.0
    # MSELoss for Volatility Regression
    # Physics Loss weights need adjustment.
    # P1/P5 (Direction) are irrelevant for Volatility magnitude.
    # P3 (Entropy -> Confidence/Vol): High Entropy -> High Volatility? Or Low Predictability?
    # Usually High Entropy = High Volatility.
    # P4 (Residual -> Vol): High Residual -> High Volatility.
    
    # New Physics Loss for Volatility (Z-Score Domain):
    # 1. Entropy-Volatility Correlation: Penalize if (Ent is high AND Pred_Z < 0)
    # 2. Residual-Volatility Correlation: Penalize if (Res is high AND Pred_Z < 0)
    
    # Weights: Aggressive (User Requested)
    # MSE is ~ 1.0 (Unit Variance).
    # Physics Penalty should be significant.
    # If Ent=0.9, Violation = 0.2 * Pred_Err.
    # To steer gradient, we need weight ~ 2.0 to 5.0.
    
    criterion = PhysicsLoss(weights={'p3': 2.0, 'p4': 2.0})
    criterion.main_loss_fn = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # Iterate over files
    for file_idx, data_path in enumerate(final_files):
        print(f"\n[{file_idx+1}/{len(final_files)}] Processing: {os.path.basename(data_path)}")
        
        try:
            train_ds, test_ds = create_rolling_datasets(
                data_path, 
                window_size=args.window_size,
                horizon=args.horizon
            )
            
            if len(train_ds) < args.batch_size:
                print("Dataset too small, skipping.")
                continue
                
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            
            # Standard: Train N epochs per file.
            file_epochs = args.epochs
            
            for epoch in range(file_epochs):
                kan.train()
                total_loss = 0
                
                # print(f"  Starting Epoch {epoch+1} with {len(train_loader)} batches")
                for batch_idx, (batch_x, batch_y, batch_sigma, batch_weights) in enumerate(train_loader):
                    # if batch_idx % 100 == 0:
                    #     print(f"    Batch {batch_idx}...")
                    
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device).unsqueeze(1)
                    batch_weights = batch_weights.to(device).unsqueeze(1)
                    
                    # 1. Physics Layer
                    # Returns [B, T, 15]
                    psi_seq = physics(batch_x)
                    
                    # 2. Temporal Flattening (Last 16 steps)
                    # psi_seq is [B, window_size, 14]. window_size is 32.
                    # We take last 16: [B, 16, 14] -> [B, 224]
                    steps_to_use = 16
                    if psi_seq.shape[1] < steps_to_use:
                        # Should not happen if window_size >= 16
                        steps_to_use = psi_seq.shape[1]
                        
                    features_seq = psi_seq[:, -steps_to_use:, :] # [B, 16, 14]
                    features = features_seq.reshape(features_seq.shape[0], -1) # [B, 224]
                    
                    # Get current physics state for loss (Last step)
                    # We need indices 0-6 at least for loss calculation
                    # 0: h_t, 1: vol, 2: ekf_v, 3: res, 4: pe, 5: price_mom, 6: ema_div, ..., 13: MSS
                    psi_t = psi_seq[:, -1, :]
                    
                    # 4. KAN
                    pred = kan(features)
                    
                    # 5. Loss with Weights
                    loss_unweighted, l_dict = criterion(pred, batch_y, psi_t)
                    
                    # Apply sample weights (Focus on Setups)
                    loss = (loss_unweighted * batch_weights).mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(train_loader)
                
                # Validation
                kan.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y, batch_sigma, batch_weights in test_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device).unsqueeze(1)
                        batch_weights = batch_weights.to(device).unsqueeze(1)
                        
                        psi_seq = physics(batch_x)
                        
                        # Temporal Flattening
                        steps_to_use = 16
                        if psi_seq.shape[1] < steps_to_use:
                            steps_to_use = psi_seq.shape[1]
                            
                        features_seq = psi_seq[:, -steps_to_use:, :]
                        features = features_seq.reshape(features_seq.shape[0], -1)
                        
                        psi_t = psi_seq[:, -1, :]
                        
                        pred = kan(features)
                        loss_unweighted, _ = criterion(pred, batch_y, psi_t)
                        loss = (loss_unweighted * batch_weights).mean()
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(test_loader)
                print(f"  Epoch {epoch+1}/{file_epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = os.path.join(args.save_dir, 'khaos_kan_best.pth')
                    torch.save({
                        'model_state_dict': kan.state_dict(),
                        'args': args,
                        'val_loss': best_val_loss
                    }, save_path)
                    # print(f"  New best model saved to {save_path}")
            
        except Exception as e:
            print(f"Error processing {data_path}: {e}")
            continue

    # Save Final Model
    save_path = os.path.join(args.save_dir, 'khaos_kan_model_final.pth')
    torch.save({
        'model_state_dict': kan.state_dict(),
        'args': args
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'd:\Finance\Finance\data\model_research\data_processed')
    parser.add_argument('--save_dir', type=str, default=r'd:\Finance\Finance\models')
    parser.add_argument('--epochs', type=int, default=2) # Reduced from 5 to 2
    parser.add_argument('--batch_size', type=int, default=128) # Increased from 64 to 128
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--window_size', type=int, default=32) # Optimized to 32
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64) # Optimized to 64
    parser.add_argument('--layers', type=int, default=3) # Optimized to 3
    parser.add_argument('--grid_size', type=int, default=10)
    
    args = parser.parse_args()
    train(args)
